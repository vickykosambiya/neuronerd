"""
06_train_runpod.py - High-Performance Training on RunPod

Optimized for NVIDIA A100 / H100 GPUs.
Features:
- Unsloth Llama 3.1 8B (Can be upgraded to 70B if VRAM allows)
- Bfloat16 Precision (Native A100 support)
- 4-bit Loading (False) or True depending on VRAM. 
  For 8B on A100, we can do 16-bit LoRA for max quality.
- Higher LoRA Rank (r=64) for better knowledge retention
"""

import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B"  # Or "unsloth/Meta-Llama-3.1-70B-bnB-4bit" for 70B
NEW_MODEL_NAME = "neuronerd-llama-3.1-8b-v1"

# RunPod A100/H100 Settings
MAX_SEQ_LENGTH = 4096   # Longer context
DTYPE = None            # Auto-detect (Float16 or Bfloat16)
LOAD_IN_4BIT = True     # Keep True for speed, set False for max quality (requires >16GB VRAM for 8B)

# LoRA Settings (Higher quality)
LORA_R = 64             # Higher rank = more parameters to train
LORA_ALPHA = 16

# Training Settings
BATCH_SIZE = 8          # Higher batch size for A100
GRAD_ACCUM = 2          # Adjust based on VRAM
LEARNING_RATE = 2e-4
MAX_STEPS = 1200        # ~2 epochs for 8k dataset
# ------------------------------------------------------------------------

def train():
    print(f"🚀 Initializing Unsloth with {MODEL_NAME}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_R,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = LORA_ALPHA,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    # Load Data (Assumes you cloned the repo)
    print("📂 Loading dataset...")
    dataset_files = {
        "train": "output/final/train.jsonl",
        "test": "output/final/test.jsonl"
    }
    # Fallback to relative path if running from root
    if not os.path.exists(dataset_files["train"]):
        print("⚠️ Local files not found, trying GitHub URL...")
        # Note: You usually just clone the repo on RunPod, so local paths should work.
        # But let's error out if missing.
        raise FileNotFoundError("Run 'python run_pipeline.py --from 4' or clone the repo first!")

    dataset = load_dataset("json", data_files=dataset_files)

    # Formatting Prompt
    alpaca_prompt = """Below is an instruction that describes a task, backed by an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    dataset = dataset.map(formatting_prompts_func, batched = True)

    print("🏋️ Starting Training...")
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_steps = 10,
            max_steps = MAX_STEPS,
            learning_rate = LEARNING_RATE,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use 'wandb' if you want charts
        ),
    )

    trainer.train()

    print("💾 Saving Model...")
    # Save GGUF (q8_0 is best balance for high quality)
    model.save_pretrained_gguf("neuronerd_model_q8", tokenizer, quantization_method = "q8_0")
    # Save 16bit GGUF (f16) for absolute max quality (large file)
    # model.save_pretrained_gguf("neuronerd_model_f16", tokenizer, quantization_method = "f16")
    
    print("✅ Done! Download 'neuronerd_model_q8.gguf' from the directory.")

if __name__ == "__main__":
    train()
