import sentencepiece as spm
import torch
from typing import List, Union, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class TokConfig:
    
    model_type: str = "unigram"   # or "bpe"
    vocab_size: int = 24576
    character_coverage: float = 1.0
    byte_fallback: bool = True
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = -1              # -1 = disabled
    unk_id: int = 0
    user_defined_symbols: Optional[list] = None
    input_sentence_size: Optional[int] = None  # subsample for speed (e.g., 10_000_000)
    shuffle_input_sentence: bool = True
    seed_sentencepiece_size: Optional[int] = None  # leave None

class SfraTokenizer:
    def __init__(self, model_path: str, add_bos: bool = True, add_eos: bool = True, device: str = "cpu"):
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.device = device
        self._model_path = str(model_path)

        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        
    # ---- runtime methods (encode/decode) ----
    def encode(self, text: str, return_tensors: bool = True):
        ids = self.sp.encode(text, out_type=int)
        if self.add_bos and self.bos_id >= 0: ids = [self.bos_id] + ids
        if self.add_eos and self.eos_id >= 0: ids = ids + [self.eos_id]
        return torch.tensor(ids, dtype=torch.long, device=self.device) if return_tensors else ids

    def decode(self, ids) -> str:
        if torch.is_tensor(ids): 
            ids = ids.tolist()
        return self.sp.decode(ids)

    def encode_batch(self, texts: List[str], max_length: Optional[int] = None, pad_to_max: bool = False) -> torch.LongTensor:
        batch = [self.encode(t, return_tensors=False) for t in texts]
        max_length = max_length or max(len(x) for x in batch)
        pad_id = self.pad_id if self.pad_id >= 0 else 0
        out = torch.full((len(batch), max_length), pad_id, dtype=torch.long, device=self.device)
        for i, seq in enumerate(batch):
            L = min(len(seq), max_length)
            out[i, :L] = torch.tensor(seq[:L], dtype=torch.long, device=self.device)
        return out

    def decode_batch(self, batch_ids: torch.Tensor):
        
        return [self.decode(seq) for seq in batch_ids]

    def vocab_size(self) -> int: return self.sp.vocab_size()
    def model_path(self) -> str: return self._model_path
