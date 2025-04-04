import os
import json
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login

# 1. Set up cache directories
os.environ["HF_DATASETS_CACHE"] = "/ephemeral/hf_cache"
os.environ["HF_HOME"] = "/ephemeral/transformers_cache"
os.environ["TMPDIR"] = "/ephemeral/tmp"

os.makedirs("/ephemeral/hf_cache", exist_ok=True)
os.makedirs("/ephemeral/transformers_cache", exist_ok=True)
os.makedirs("/ephemeral/tmp", exist_ok=True)

# 2. Authentication -----------------------------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Make sure these are set in your environment
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

login(token=HF_API_TOKEN)
wandb.login(key=WANDB_API_KEY)

# 3. Load Pretrained Tokenizer & Model for Qwen -----------------------------
# Replace with the correct Qwen model name on Hugging Face Hub.
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Example; adjust if needed.
DATASET_PATH = "../qwen_training_data.json"  # Path to your JSON training data

# Load tokenizer for Qwen (assumes a causal model tokenizer)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load Qwen causal language model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto",)


# 4. Apply LoRA Adapters (for causal LM) --------------------------------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Use CAUSAL_LM for Qwen
    r=32,                        # LoRA rank
    lora_alpha=64,               # Scaling factor
    lora_dropout=0.1,            # Dropout for regularization
    target_modules=["q_proj", "v_proj"],  # Adjust these target modules if needed for Qwen's architecture
)

model = get_peft_model(model, lora_config)

# Ensure LoRA parameters are trainable
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params} || Total params: {total_params} || Trainable%: {100 * trainable_params / total_params:.4f}")

# 5. Load & Preprocess Dataset (from JSON) -------------------------------------
# If you have a single JSON file, load it and split into train/validation.
raw_dataset = load_dataset("json", data_files=DATASET_PATH)["train"]
dataset = raw_dataset.train_test_split(test_size=0.05)
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})

def build_prompt(example):
    return (
        f"Instruction: {example['instruction']}\n"
        f"Input: {example['input']}\n"
        f"Output: {example['output']}"
    )

def preprocess_function(examples):
    prompts = []
    # examples is a dict of lists; iterate over indices
    for i in range(len(examples["instruction"])):
        # Construct a dictionary for each sample
        sample = {
            "instruction": examples["instruction"][i],
            "input": examples["input"][i],
            "output": examples["output"][i]
        }
        prompts.append(build_prompt(sample))
    
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=1024,  # Adjust max_length as needed
        padding="max_length"
    )
    return tokenized

train_dataset = dataset["train"].map(
    preprocess_function,
    batched=True,
    num_proc=20,
    remove_columns=dataset["train"].column_names  # Remove original string columns
)
eval_dataset = dataset["validation"].map(
    preprocess_function,
    batched=True,
    num_proc=20,
    remove_columns=dataset["validation"].column_names
)

processed_dataset = DatasetDict({
    "train": train_dataset,
    "validation": eval_dataset
})

print("Example tokenized input:", processed_dataset["train"][0])

# 6. Data Collator for Causal LM ------------------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 7. Training Arguments ---------------------------------------------------
training_args = TrainingArguments(
    output_dir="/ephemeral/qwen_finetuned",
    run_name="qwen_mn_en_translation",
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=10,
    per_device_train_batch_size=4,  # Adjust according to GPU memory; Qwen models tend to be heavy
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Adjust to simulate a larger batch size if needed
    lr_scheduler_type="cosine",
    warmup_steps=100,
    learning_rate=1e-4,
    num_train_epochs=2,
    weight_decay=0.01,
    fp16=True,
    logging_dir="/workspace/logs",
    logging_steps=10,
    report_to="wandb",
    remove_unused_columns=False,
    optim="adamw_torch",
    ddp_find_unused_parameters=False,
    
    # Dataloader config
    dataloader_drop_last=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=8,
    
    # Push to Hugging Face Hub (optional)
    push_to_hub=True,
    hub_model_id="Billyyy/qwen_mn_en_finetuned",
    hub_strategy="every_save",
)

# 8. Trainer Initialization ------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Disable cache to avoid warnings during training
model.config.use_cache = False

# 9. Train & Save Model --------------------------------------------------
trainer.train()
trainer.save_model("/ephemeral/qwen_mn_en_lora_trained")

print("✅ LoRA fine-tuning complete! Model saved at `/ephemeral/qwen_mn_en_lora_trained`")
