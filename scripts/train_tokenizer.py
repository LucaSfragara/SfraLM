from pathlib import Path
import json
import torch
import sentencepiece as spm


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
