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

    # ---- factory to train + return a ready-to-use tokenizer ----
    @classmethod
    def train(cls, input_files, out_dir: str, prefix: str = "sfrallm", cfg: TokConfig = TokConfig(),
              add_bos=True, add_eos=True, device="cpu"):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        
        model_path = out / f"{prefix}.model"
        vocab_path  = out / f"{prefix}.vocab"
        cfg_path    = out / f"{prefix}.json"

        if isinstance(input_files, (list, tuple)):
            input_spec = ",".join(map(str, input_files))
        else:
            input_spec = str(input_files)

        args = {
            "input": input_spec,
            "model_prefix": str(out / prefix),
            "model_type": cfg.model_type,
            "vocab_size": cfg.vocab_size,
            "character_coverage": cfg.character_coverage,
            "byte_fallback": cfg.byte_fallback,
            "bos_id": cfg.bos_id,
            "eos_id": cfg.eos_id,
            "pad_id": cfg.pad_id,
            "unk_id": cfg.unk_id,
        }
        if cfg.user_defined_symbols:
            args["user_defined_symbols"] = ",".join(cfg.user_defined_symbols)
        if cfg.input_sentence_size:
            args["input_sentence_size"] = cfg.input_sentence_size
            args["shuffle_input_sentence"] = cfg.shuffle_input_sentence

        spm.SentencePieceTrainer.train(**args)

        # save a tiny sidecar config for reproducibility
        (out / cfg_path.name).write_text(json.dumps(asdict(cfg), indent=2))
        return cls(str(model_path), add_bos=add_bos, add_eos=add_eos, device=device)

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

    def encode_batch(self, texts: List[str], max_length: Optiona[int] = None, pad_to_max: bool = False) -> torch.LongTensor:
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
