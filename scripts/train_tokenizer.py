from pathlib import Path
import json
import torch
import sentencepiece as spm
from typing import List, Optional
from config import config

input_files = config["data"]["download_dir"]

prefix = config["prefix"]
cfg = config["tokenizer"]

out = Path(cfg["out_dir"])
out.mkdir(parents=True, exist_ok=True)

model_path = out / f"{prefix}.model"
vocab_path  = out / f"{prefix}.vocab"
cfg_path    = out / f"{prefix}.json"

if isinstance(input_files, (list, tuple)):
    input_spec = ",".join(map(str, input_files))
else:
    input_spec = str(input_files)
print("input_spec:", input_spec )
args = {
    "input": "m.txt",
    "model_prefix": str(out / prefix),
    "model_type": cfg["model_type"],
    "vocab_size": cfg["vocab_size"],
    "character_coverage": cfg["character_coverage"],
    "byte_fallback": cfg["byte_fallback"],
    "bos_id": cfg["bos_id"],
    "eos_id": cfg["eos_id"],
    "pad_id": cfg["pad_id"],
    "unk_id": cfg["unk_id"]
}
if cfg["user_defined_symbols"]:
    args["user_defined_symbols"] = ",".join(cfg["user_defined_symbols"])
if cfg["input_sentence_size"]:
    args["input_sentence_size"] = cfg["input_sentence_size"]
    args["shuffle_input_sentence"] = cfg["shuffle_input_sentence"]

spm.SentencePieceTrainer.train(**args)

# save a tiny sidecar config for reproducibility
(out / cfg_path.name).write_text(json.dumps(cfg, indent=2))


