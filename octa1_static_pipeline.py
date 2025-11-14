# static_pipeline_hscode_full.py
"""
End-to-end pipeline for fine-tuning an LLM on HS Code chat-style data,
merging adapters, converting to GGUF, and creating an Ollama model.

Expect dataset folder:
  dataset_out/
    train.jsonl   # each line: {"messages": [{"role":"user","content":...}, {"role":"assistant","content":...}]}
    (optional) val.jsonl

Run:
  python3 static_pipeline_hscode_full.py
"""

import os
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import set_seed
from tokenizers import Tokenizer
import sentencepiece as spm

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig

# --------------------
# Config (edit to taste)
# --------------------
# DATA_DIR = "./dataset_out"                  # input folder with train.jsonl (and optional val.jsonl)
# OUT_DIR = "outputs_hscode_lora"             # base output dir
# BASE     = "meta-llama/Llama-3.1-8B-Instruct"      
# MAX_SEQ_LEN = 2048
# PER_DEVICE_BS = 2
# GRAD_ACCUM = 8
# MAX_STEPS = 1000                            # reasonable starting point for 13k data (tune later)
# LR = 2e-4
# DO_MERGE = True                             # set False to skip merge + conversion
# QUANTS = ["f16", "q8_0"]                    # GGUF output types
# CUSTOM_MODEFILE = "custom_Modelfile.txt"    # optional custom modelfile template
# LLAMACPP_DIR = Path(__file__).resolve().parent / "llama.cpp"
# VAL_SPLIT_RATIO = 0.05                      # auto-split val if only train exists (5%)


DATA_DIR = "./dataset_out"           # train.jsonl (+ optional val.jsonl)
OUT_DIR  = "outputs_llama31_8b_hscode"
BASE     = "meta-llama/Llama-3.1-8B-Instruct"   #  new base model
SEED = 42
MAX_SEQ_LEN = 2048            # Llama 3.1 supports up to 8k if you have memory,80 GB GPUs, you can increase it to 4096 or 8192
PER_DEVICE_BS = 2
GRAD_ACCUM = 8
MAX_STEPS = 500
LR = 2e-4
DO_MERGE = True 
QUANTS = ["f16", "q8_0"]                    # GGUF output types
CUSTOM_MODEFILE = "custom_Modelfile.txt"    # optional custom modelfile template
LLAMACPP_DIR = Path(__file__).resolve().parent / "llama.cpp"
VAL_SPLIT_RATIO = 0.05   


# --------------------
# Helpers
# --------------------
def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl_iterable(path: Path, iterable):
    with path.open("w", encoding="utf-8") as f:
        for obj in iterable:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def maybe_create_val_split(data_dir: Path, val_ratio: float = VAL_SPLIT_RATIO) -> Optional[Path]:
#def maybe_create_val_split(data_dir: Path, val_ratio: float = 0.05) -> Optional[Path]:
    """
    If val.jsonl missing and train.jsonl exists, create val.jsonl by splitting.
    Returns path to val.jsonl or None.
    """
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")
    if val_path.exists():
        print(f"[INFO] Found existing validation file: {val_path}")
        return val_path

    # load all entries (may be large but 13k is fine)
    print("[INFO] Creating validation split from train.jsonl...")
    entries = list(read_jsonl(train_path))
    random.Random(SEED).shuffle(entries)
    n_val = max(1, int(len(entries) * val_ratio))
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]

    # overwrite train.jsonl with new train split and write val.jsonl
    write_jsonl_iterable(train_path, train_entries)
    write_jsonl_iterable(val_path, val_entries)
    print(f"[INFO] Created val.jsonl with {len(val_entries)} entries; train.jsonl now {len(train_entries)} entries.")
    return val_path

def load_split(name: str, data_dir: Path) -> Optional[Dataset]:
    p = data_dir / f"{name}.jsonl"
    if not p.exists():
        return None
    return load_dataset("json", data_files=str(p), split="train")

def write_modelfile(custom_modelfile: Path, chosen_gguf: str, out_path: Path):
    """Write Modelfile using custom template if available"""
    if custom_modelfile.exists():
        try:
            lines = custom_modelfile.read_bytes().decode("utf-8", errors="ignore").splitlines()
        except Exception as e:
            print(f"[WARN] Could not read {custom_modelfile}: {e}. Falling back to default.")
            lines = []

        new_lines = []
        for line in lines:
            if line.strip().startswith("FROM "):
                new_lines.append(f"FROM {chosen_gguf}")
            else:
                new_lines.append(line)
        if not new_lines:
            new_lines = [f"FROM {chosen_gguf}", 'TEMPLATE "{{ .Prompt }}"']
        out_path.write_text("\n".join(new_lines), encoding="utf-8")
        print(f"[INFO] Wrote Modelfile from custom template: {out_path}")
    else:
        out_path.write_text(f"FROM {chosen_gguf}\nTEMPLATE \"{{{{ .Prompt }}}}\"", encoding="utf-8")
        print(f"[INFO] Wrote default Modelfile: {out_path}")

def ensure_llama_cpp(llamacpp_dir: Path):
    """Clone and build llama.cpp if missing. Cache build if already exists."""
    if not llamacpp_dir.exists():
        print(f"[INFO] Cloning llama.cpp into {llamacpp_dir}...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(llamacpp_dir)], check=True)
    else:
        print(f"[INFO] llama.cpp already exists at {llamacpp_dir}, pulling latest...")
        subprocess.run(["git", "-C", str(llamacpp_dir), "pull"], check=True)

    build_dir = llamacpp_dir / "build"
    llama_quantize = build_dir / "bin" / "llama-quantize"
    if llama_quantize.exists():
        print("[INFO] llama.cpp build already exists, skipping rebuild.")
        return

    build_dir.mkdir(exist_ok=True)
    print(f"[INFO] Building llama.cpp in {build_dir} (requires cmake and C++ toolchain)...")
    subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
    subprocess.run(["cmake", "--build", ".", "-j"], cwd=build_dir, check=True)
    print("[INFO] llama.cpp build complete.")

def rebuild_tokenizer_model_from_json(tokenizer_json_path: Path, output_path: Path):
    """Attempt to rebuild a tokenizer.model (SentencePiece) from tokenizer.json"""
    print(f"[INFO] Rebuilding tokenizer.model from {tokenizer_json_path} -> {output_path}")
    tok = Tokenizer.from_file(str(tokenizer_json_path))
    vocab = tok.get_vocab()
    vocab_txt = output_path.with_suffix(".vocab.txt")
    with open(vocab_txt, "w", encoding="utf-8") as f:
        for token, idx in vocab.items():
            f.write(f"{token} {idx}\n")
    spm.SentencePieceTrainer.Train(
        f"--input={vocab_txt} "
        f"--model_prefix={str(output_path.with_suffix(''))} "
        f"--vocab_size={len(vocab)} "
        "--character_coverage=1.0 "
        "--model_type=bpe"
    )
    print("[INFO] Rebuilt tokenizer.model")

def ensure_tokenizer_model(tokenizer, merged_dir: Path, base_model_id: str):
    """Ensure tokenizer.model exists in merged_dir, otherwise rebuild from tokenizer.json"""
    tok_path = merged_dir / "tokenizer.model"
    tokenizer.save_pretrained(merged_dir)
    if not tok_path.exists() or tok_path.stat().st_size == 0:
        print("[WARN] tokenizer.model missing/empty; trying to rebuild from tokenizer.json...")
        tok_json = merged_dir / "tokenizer.json"
        if tok_json.exists():
            rebuild_tokenizer_model_from_json(tok_json, tok_path)
    if not tok_path.exists() or tok_path.stat().st_size == 0:
        raise FileNotFoundError("tokenizer.model still missing after attempted rebuild.")

# --------------------
# Main pipeline
# --------------------
def main():
    data_dir = Path(DATA_DIR)
    ensure_dir(data_dir)
    set_seed(SEED)
    random.seed(SEED)

    # 1) Ensure val split exists (create if missing)
    val_path = maybe_create_val_split(data_dir)  # will create val.jsonl if missing (and return its path)

    # 2) Load datasets
    train_ds = load_split("train", data_dir)
    val_ds = load_split("val", data_dir)  # may be None if not present (should not be after split)
    if train_ds is None:
        raise FileNotFoundError(f"Missing {data_dir / 'train.jsonl'}")
    assert "messages" in train_ds.column_names, "Dataset must have 'messages' field!"

    # 3) Load base model + tokenizer
    print(f"[INFO] Loading base model: {BASE}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.float16,   # or torch.float16 if GPU doesn’t support bf16
        load_in_4bit=False,
        full_finetuning=False,
        device_map="auto",
    )
    model = FastLanguageModel.for_training(model)



    # model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name=BASE,
    # max_seq_length=MAX_SEQ_LEN,
    # dtype=torch.bfloat16,     
    # device_map="auto",
    # )


    # 4) Chat template setup
    tokenizer = get_chat_template(tokenizer, chat_template="unsloth")
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5) Convert 'messages' -> 'text' (apply chat template)
    def to_text(batch):
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch["messages"]
        ]
        return {"text": texts}
    print("[INFO] Mapping dataset messages to text with chat template...")
    train_ds = train_ds.map(to_text, batched=True, remove_columns=[c for c in train_ds.column_names if c not in ["messages"]])
    if val_ds is not None:
        val_ds = val_ds.map(to_text, batched=True, remove_columns=[c for c in val_ds.column_names if c not in ["messages"]])

    # 6) Add LoRA (PEFT)
    print("[INFO] Attaching LoRA adapters...")
    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r=32,
    #     target_modules=["q_proj", "v_proj"],
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     bias="none",
    #     use_gradient_checkpointing="unsloth",
    #     random_state=SEED,
    # )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,                        # LoRA rank
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Llama architecture
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )


    # 7) Setup trainer
    print("[INFO] Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=OUT_DIR,
            per_device_train_batch_size=PER_DEVICE_BS,
            gradient_accumulation_steps=GRAD_ACCUM,
            max_steps=MAX_STEPS,
            learning_rate=LR,
            warmup_steps=10,
            logging_steps=10,
            eval_strategy="steps" if val_ds is not None else "no",
            eval_steps=50,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            optim="adamw_torch",
            seed=SEED,
            report_to="none",
            bf16=True,
            packing=False,
            dataset_text_field="text",
        ),
    )

    # 8) Train
    print("[INFO] Starting training...")
    trainer.train(resume_from_checkpoint=False)
    print("[INFO] Training finished. Saving adapter + tokenizer...")
    os.makedirs(OUT_DIR, exist_ok=True)
    # Save LoRA adapter (implementation dependent—FastLanguageModel handles adapter saving)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[OK] LoRA saved to {OUT_DIR}")

    # 9) Merge + convert to GGUF + create Ollama model (optional)
    if DO_MERGE:
        merged_dir = OUT_DIR + "-merged"
        merged_path = Path(merged_dir)
        if merged_path.exists():
            print(f"[INFO] Removing existing merged dir: {merged_dir}")
            shutil.rmtree(merged_dir)

        print(f"[INFO] Merging adapters into {merged_dir} ...")
        if hasattr(FastLanguageModel, "merge_lora"):
            FastLanguageModel.merge_lora(
                model,
                lora_model_dir=OUT_DIR,
                save_dir=merged_dir,
                dtype=torch.bfloat16,
            )
        else:
            # fallback API
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

        # ensure tokenizer.model is present
        ensure_tokenizer_model(tokenizer, merged_path, BASE)

        # ensure llama.cpp is present and built
        try:
            ensure_llama_cpp(LLAMACPP_DIR)
        except Exception as e:
            raise RuntimeError(f"Failed to prepare llama.cpp: {e}")

        # find conversion script in llama.cpp repo
        convert_script = LLAMACPP_DIR / "convert-hf-to-gguf.py"
        if not convert_script.exists():
            convert_script = LLAMACPP_DIR / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            raise FileNotFoundError("Could not find convert-hf-to-gguf.py in llama.cpp repository.")

        ggufs = []
        for outtype in QUANTS:
            gguf_out = f"{merged_dir}-{outtype}.gguf"
            cmd = [
                "python3", str(convert_script),
                merged_dir,
                "--outfile", gguf_out,
                "--outtype", outtype,
            ]
            print(f"[INFO] Converting merged HF model to GGUF ({outtype}) -> {gguf_out}")
            subprocess.run(cmd, check=True)
            ggufs.append(gguf_out)

        # Write Modelfile and create an Ollama model
        chosen_gguf = ggufs[-1]  # pick the last quant produced (e.g., q8_0)
        write_modelfile(Path(CUSTOM_MODEFILE), chosen_gguf, Path("Modelfile"))

        model_name = Path(OUT_DIR).name
        print(f"[INFO] Creating Ollama model '{model_name}' from Modelfile (requires 'ollama' CLI)...")
        subprocess.run(["ollama", "create", model_name, "-f", "Modelfile"], check=True)
        print(f"[OK] Ollama model created: {model_name}")

    print("[DONE] Pipeline complete.")

if __name__ == "__main__":
    main()
