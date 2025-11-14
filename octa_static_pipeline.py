#!/usr/bin/env python3
"""
Complete end-to-end pipeline: fine-tune (LoRA) -> merge -> convert to GGUF -> create Ollama model
Designed for chat-style JSONL datasets where each line is an object like:
{"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

Usage (example):
python static_pipeline_full_pipeline.py --data-dir ./dataset_out --out-dir outputs_hscode_full \
    --base-model unsloth/mistral-7b-bf16 --max-steps 800 --per-device-batch 2 --grad-accum 8 \
    --do-merge --quants f16 q8_0

Notes:
- Requires Python packages: torch, datasets, transformers, trl, tokenizers, sentencepiece, unsloth
- Requires system tools for merging/conversion: git, cmake, a C++ toolchain, and optionally 'ollama' CLI for creating Ollama models
- This script attempts to be robust and logs key steps. Adjust hyperparameters as needed.
"""

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

import torch
from datasets import load_dataset, Dataset
from transformers import set_seed

# Optional/project-specific imports - the script expects these to be available in the environment
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
except Exception:
    FastLanguageModel = None
    get_chat_template = None

try:
    from trl import SFTTrainer, SFTConfig
except Exception:
    SFTTrainer = None
    SFTConfig = None

try:
    from tokenizers import Tokenizer
    import sentencepiece as spm
except Exception:
    Tokenizer = None
    spm = None


# --------------------
# Logging config
# --------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("static_pipeline")


# --------------------
# Helpers
# --------------------

def load_split(data_dir: Path, name: str) -> Optional[Dataset]:
    path = data_dir / f"{name}.jsonl"
    if not path.exists():
        return None
    logger.info(f"Loading dataset split: {path}")
    return load_dataset("json", data_files=str(path), split="train")


def write_modelfile(custom_modelfile: Path, chosen_gguf: str, out_path: Path) -> None:
    """Write Modelfile using custom template if available"""
    if custom_modelfile and custom_modelfile.exists():
        try:
            lines = custom_modelfile.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as e:
            logger.warning(f"Could not read {custom_modelfile} cleanly: {e}, falling back to default.")
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
        logger.info(f"Using custom Modelfile with updated FROM: {chosen_gguf}")
    else:
        out_path.write_text(f"FROM {chosen_gguf}\nTEMPLATE \"{{{{ .Prompt }}}}\"", encoding="utf-8")
        logger.info(f"Default Modelfile written (FROM {chosen_gguf})")


def ensure_llama_cpp(llamacpp_dir: Path) -> None:
    """Clone and build llama.cpp if missing. Cache build if already exists."""
    if not llamacpp_dir.exists():
        logger.info(f"Cloning llama.cpp into {llamacpp_dir}...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(llamacpp_dir)], check=True)
    else:
        logger.info(f"llama.cpp already exists at {llamacpp_dir}, pulling latest...")
        subprocess.run(["git", "-C", str(llamacpp_dir), "pull"], check=True)

    build_dir = llamacpp_dir / "build"
    llama_quantize = build_dir / "bin" / "llama-quantize"
    if llama_quantize.exists():
        logger.info("llama.cpp build already exists, skipping rebuild")
        return

    build_dir.mkdir(exist_ok=True)
    logger.info(f"Building llama.cpp in {build_dir} (this may take a while)...")
    subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
    subprocess.run(["cmake", "--build", ".", "-j"], cwd=build_dir, check=True)
    logger.info("llama.cpp build complete")


def rebuild_tokenizer_model_from_json(tokenizer_json_path: Path, output_path: Path) -> None:
    """Rebuild tokenizer.model from tokenizer.json using tokenizers + sentencepiece"""
    if Tokenizer is None or spm is None:
        raise RuntimeError("tokenizers and sentencepiece are required to rebuild tokenizer.model")

    logger.info(f"Rebuilding tokenizer.model from {tokenizer_json_path}")
    tok = Tokenizer.from_file(str(tokenizer_json_path))
    vocab = tok.get_vocab()

    vocab_txt = output_path.with_suffix(".vocab.txt")
    with open(vocab_txt, "w", encoding="utf-8") as f:
        for token, idx in vocab.items():
            f.write(f"{token} {idx}\n")

    # Train SentencePiece model
    spm.SentencePieceTrainer.Train(
        f"--input={vocab_txt} "
        f"--model_prefix={output_path.with_suffix('').as_posix()} "
        f"--vocab_size={len(vocab)} "
        "--character_coverage=1.0 "
        "--model_type=bpe"
    )
    logger.info(f"Rebuilt tokenizer.model at {output_path}")


def ensure_tokenizer_model(tokenizer, merged_dir: Path, base_model_id: str) -> None:
    """Ensure tokenizer.model exists, otherwise rebuild from tokenizer.json"""
    tok_path = merged_dir / "tokenizer.model"
    tokenizer.save_pretrained(merged_dir)

    # If missing or empty, rebuild from tokenizer.json
    if not tok_path.exists() or tok_path.stat().st_size == 0:
        logger.warning("tokenizer.model missing or empty, attempting rebuild...")
        tok_json = merged_dir / "tokenizer.json"
        if tok_json.exists():
            try:
                rebuild_tokenizer_model_from_json(tok_json, tok_path)
            except Exception as e:
                logger.error(f"Failed to rebuild tokenizer.model: {e}")

    if not tok_path.exists() or tok_path.stat().st_size == 0:
        raise FileNotFoundError(
            "tokenizer.model is still missing/empty! "
            f"Tried saving and rebuilding from tokenizer.json ({base_model_id})."
        )


def convert_to_gguf(llamacpp_dir: Path, merged_dir: str, quants: List[str]) -> List[str]:
    """Convert merged HF model to GGUF using convert script in llama.cpp"""
    convert_script = llamacpp_dir / "convert-hf-to-gguf.py"
    if not convert_script.exists():
        convert_script = llamacpp_dir / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        raise FileNotFoundError("convert-hf-to-gguf.py / convert_hf_to_gguf.py not found in llama.cpp")

    ggufs = []
    for outtype in quants:
        gguf_out = f"{merged_dir}-{outtype}.gguf"
        cmd = ["python3", str(convert_script), merged_dir, "--outfile", gguf_out, "--outtype", outtype]
        logger.info(f"Converting to GGUF ({outtype}) -> {gguf_out}")
        subprocess.run(cmd, check=True)
        ggufs.append(gguf_out)
    return ggufs


# --------------------
# Main pipeline
# --------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Full LLM fine-tuning -> merge -> GGUF -> Ollama pipeline")

    # Paths & dataset
    parser.add_argument("--data-dir", type=Path, default=Path("./dataset_out"), help="Folder with train.jsonl (and optional val.jsonl)")
    parser.add_argument("--out-dir", type=str, default="outputs_hscode_full", help="Output directory for adapters and merged model")
    parser.add_argument("--base-model", type=str, default="unsloth/gpt-oss-20b-BF16", help="Base model identifier to load via FastLanguageModel")

    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--per-device-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)

    # LoRA params
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Merge & conversion
    parser.add_argument("--do-merge", action="store_true", help="After training, merge LoRA into base and convert to GGUF + create Ollama model")
    parser.add_argument("--llama-cpp-dir", type=Path, default=Path("./llama.cpp"), help="Path to clone/build llama.cpp for conversion")
    parser.add_argument("--quants", nargs="+", default=["f16", "q8_0"], help="Quantization targets for GGUF conversion")
    parser.add_argument("--custom-modelfile", type=Path, default=Path("custom_Modelfile.txt"), help="Optional custom modelfile template")

    # Misc
    parser.add_argument("--no-bf16", dest="bf16", action="store_false", help="Disable BF16 (default enabled)")
    parser.add_argument("--val-split-size", type=float, default=0.0, help="If >0, split this fraction of train into validation set")
    parser.add_argument("--device-map", type=str, default="auto", help="Device map for model loading (e.g., 'auto')")

    args = parser.parse_args(argv)

    # Basic validations
    data_dir: Path = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # reproducibility
    set_seed(args.seed)
    random.seed(args.seed)

    # Load datasets
    train_ds = load_split(data_dir, "train")
    val_ds = load_split(data_dir, "val")

    # Optionally split validation from train if a fraction is specified and val.jsonl absent
    if val_ds is None and args.val_split_size > 0.0:
        logger.info(f"Splitting {args.val_split_size*100:.1f}% of train into validation set")
        # HuggingFace datasets take a fraction in .train_test_split
        split = train_ds.train_test_split(test_size=args.val_split_size, seed=args.seed)
        train_ds = split["train"]
        val_ds = split["test"]

    if train_ds is None:
        raise FileNotFoundError(f"Missing {data_dir}/train.jsonl")

    if "messages" not in train_ds.column_names:
        raise AssertionError("Dataset must have 'messages' field!")

    # Load base model + tokenizer
    if FastLanguageModel is None:
        raise RuntimeError("unsloth.FastLanguageModel not available in the environment")
    if SFTTrainer is None or SFTConfig is None:
        raise RuntimeError("trl.SFTTrainer / SFTConfig not available in the environment")

    logger.info(f"Loading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        load_in_4bit=False,
        full_finetuning=False,
        device_map=args.device_map,
    )
    model = FastLanguageModel.for_training(model)

    # Chat template
    if get_chat_template is None:
        raise RuntimeError("unsloth.chat_templates.get_chat_template not available")

    tokenizer = get_chat_template(tokenizer, chat_template="unsloth")
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Convert messages -> text (apply chat template)
    def to_text(batch):
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch["messages"]
        ]
        return {"text": texts}

    logger.info("Formatting datasets with chat template...")
    train_ds = train_ds.map(to_text, batched=True, remove_columns=[c for c in train_ds.column_names if c != "messages"])
    if val_ds is not None:
        val_ds = val_ds.map(to_text, batched=True, remove_columns=[c for c in val_ds.column_names if c != "messages"])

    # Add LoRA/PEFT
    logger.info("Applying LoRA (PEFT) adapters to the model")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # Trainer config
    logger.info("Configuring trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(out_dir),
            per_device_train_batch_size=args.per_device_batch,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            warmup_steps=10,
            logging_steps=10,
            eval_strategy="steps" if val_ds is not None else "no",
            eval_steps=50,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            optim="adamw_torch",
            seed=args.seed,
            report_to="none",
            bf16=args.bf16,
            packing=False,
            dataset_text_field="text",
        ),
    )

    # Train
    logger.info("Starting training... (this may take a while)")
    trainer.train(resume_from_checkpoint=False)

    # Save LoRA adapter + tokenizer
    logger.info("Saving adapter + tokenizer...")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info(f"[OK] LoRA saved: {out_dir}")

    # Merge + convert + ollama
    if args.do_merge:
        merged_dir = f"{out_dir}-merged"
        if Path(merged_dir).exists():
            logger.info(f"Removing existing merged dir: {merged_dir}")
            shutil.rmtree(merged_dir)

        logger.info(f"Merging adapters into: {merged_dir}")
        if hasattr(FastLanguageModel, "merge_lora"):
            FastLanguageModel.merge_lora(
                model,
                lora_model_dir=out_dir,
                save_dir=merged_dir,
                dtype=torch.bfloat16 if args.bf16 else torch.float32,
            )
        else:
            # fallback
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

        # Ensure tokenizer.model present
        ensure_tokenizer_model(tokenizer, Path(merged_dir), args.base_model)

        # Ensure llama.cpp is available
        ensure_llama_cpp(args.llama_cpp_dir)

        # Convert to GGUF
        ggufs = convert_to_gguf(args.llama_cpp_dir, merged_dir, args.quants)

        # Write Modelfile and create Ollama model
        chosen_gguf = ggufs[-1]
        write_modelfile(args.custom_modelfile, chosen_gguf, Path("Modelfile"))

        model_name = Path(out_dir).name
        logger.info(f"Creating Ollama model '{model_name}' from Modelfile (requires ollama CLI)")
        subprocess.run(["ollama", "create", model_name, "-f", "Modelfile"], check=True)
        logger.info(f"[OK] Ollama model created: {model_name}")

    logger.info("[DONE] Static pipeline complete.")


if __name__ == "__main__":
    main()
