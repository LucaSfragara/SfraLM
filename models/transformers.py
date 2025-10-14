import torch
import torch.nn as nn
import random
from typing import Tuple, Optional
#from models.masks import PadMask, CausalMask
from models.layers.decoder_layers import SelfAttentionDecoderLayer
from models.layers.embedding import TokenEmbedding

class DecoderOnlyTransformer(nn.Module):
    '''
    A Pre-LN Decoder-Only Transformer model.
    '''
    def __init__(
            self, 
            num_layers: int, 
            d_model: int, 
            num_heads: int, 
            d_ff: int, 
            dropout: float, 
            seq_len: int, 
            num_classes: int,
            weight_tying: bool = False,
            layer_drop_rate: float = 0.0,
    ):
        '''
        Initialize the Decoder-Only Transformer model.

        Args:
            num_layers: int, number of decoder layers
            d_model: int, model dimension
            num_heads: int, number of attention heads
            d_ff: int, feed-forward dimension
            dropout: float, dropout rate
            seq_len: int, sequence length
            num_classes: int, number of classes
            weight_tying: bool, whether to use weight tying (default: False)
            layer_drop_rate: float, layer drop rate (default: 0.0)
        '''
        super().__init__()
        
       
        # Initialize the decoder
        self.seq_len         = seq_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes     = num_classes
        self.num_layers      = num_layers
        
        # TODO: Create a ModuleList of decoder layers based on the number of layers
        self.dec_layers     = nn.ModuleList(
            [(SelfAttentionDecoderLayer(d_model, num_heads, d_ff, dropout)) for _ in range(num_layers)]
        ) # ModuleList of decoder layers

        self.target_embedding       = TokenEmbedding(num_classes, d_model)

        self.final_linear           = nn.Linear(d_model, num_classes) # Final linear layer

        nn.init.normal_(self.final_linear.weight, mean=0.0, std=0.02) # Initialize final linear layer

        self.dropout                = nn.Dropout(dropout) # Dropout
        self.norm                   = nn.LayerNorm(d_model) # Layer norm

        # Weight tying (extra form of regularization, read more about it)
        if weight_tying:
            self.target_embedding.embed.weight = self.final_linear.weight

    @torch.no_grad()
    def _make_pos_ids(self, B: int, T: int, device) -> torch.Tensor:
        # positions 0..T-1 for every sequence (change here for packed/chunked)
        return torch.arange(T, device=device).unsqueeze(0).expand(B, T)


    def forward(self, padded_targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        
        '''
        Forward pass for the decoder. Used for Training only. Tokens are assumed to be right-padded.
        Args:
            padded_targets (torch.Tensor): The padded target sequence. shape: (batch_size, seq_len)
        Returns:
            seq_out (torch.Tensor): The output sequence. shape: (batch_size, seq_len, d_model)
            runnint_att (dict): The attention weights. shape: (batch_size, seq_len, seq_len)
        '''
    
        x = self.target_embedding(padded_targets)
    
        x = self.dropout(x)

        runnint_att = {}
        for i in range(self.num_layers):
            # Optionally apply LayerDrop during training (More regularization!)
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
            
            x, attention = self.dec_layers[i](x, pos_ids = self._make_pos_ids(x.size(0), x.size(1), x.device)) # shape (batch_size, seq_len, d_model), (batch_size, seq_len, seq_len)
            
            runnint_att['layer{}_dec_self'.format(i + 1)] = attention #shape (batch_size, seq_len, seq_len) 

        x = self.norm(x)
  
        seq_out = self.final_linear(x)
  
        return seq_out, runnint_att
    
    def score(self, batch_prompts: torch.Tensor) -> torch.Tensor:
        '''
        Score the tokens for the decoder. 
        This is used for scoring the next token for a given prompt.
        Padding mask is not applied so ensure that the prompts are not padded. 
        Can only handle batch_size = 1 or batch with same lengths and no padding. 
        Args:
            prompts (torch.Tensor) : tensor of fixed length token sequences. shape: (batch_size, seq_len)
        Returns:
            logits (torch.Tensor): Batch of next token logits. shape: (batch_size, num_classes)
        '''
        if self.training:
            raise ValueError("score method is not supported during training, use forward method instead")
        # Forward pass with no target lengths
        seq_out, _ = self.forward(batch_prompts)
        # Return the last token's logits for next token prediction    
        logits     = seq_out[:, -1, :]
        return logits
    
