import os

# Set new cache directories
os.environ["HF_DATASETS_CACHE"] = "/ephemeral/hf_cache"
os.environ["HF_HOME"] = "/ephemeral/transformers_cache"
os.environ["TMPDIR"] = "/ephemeral/tmp"

# Ensure the directories exist
os.makedirs("/ephemeral/hf_cache", exist_ok=True)
os.makedirs("/ephemeral/transformers_cache", exist_ok=True)
os.makedirs("/ephemeral/tmp", exist_ok=True)

import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
from transformers import BitsAndBytesConfig
from transformers import get_cosine_schedule_with_warmup, AdamW

# 1. Authentication -----------------------------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Ensure token is set in environment
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Login to Hugging Face Hub & Weights & Biases
login(token=HF_API_TOKEN)
wandb.login(key=WANDB_API_KEY)

# 2. Load Pretrained Tokenizer & Model -----------------------------
MODEL_NAME = "Billyyy/llama3_mn_embed"  # Change if using another model
TOKENIZER_PATH = "Billyyy/llama_8K_extended"  # Your trained tokenizer
DATASET_NAME = "Billyyy/cleaned-mongolian-dataset"  # Dataset on HF Hub

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding is defined

subset_size = 505_000  # Define max dataset size

# Load dataset
dataset = load_dataset(DATASET_NAME)

# Tokenize dataset function
def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=1024
    )

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=27)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])  # Keep only tokenized data

# Convert to PyTorch format
tokenized_dataset.set_format("torch")

tokenized_dataset["train"] = tokenized_dataset["train"].select(range(subset_size))

dataset = tokenized_dataset["train"].shuffle(seed=42).train_test_split(test_size=0.01, seed=42)

total_samples = len(dataset["train"])

# 3. Load QLoRA Model with 4-bit Quantization ----------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Use FP16 computations
    bnb_4bit_quant_type="nf4",  # Normalized 4-bit Quantization (best for LLaMA)
    bnb_4bit_use_double_quant=True,  # Extra compression
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare model for training with LoRA
model = prepare_model_for_kbit_training(model)

# Get layer names for last 5 layers
total_layers = len(model.model.layers)
target_modules = [
    f"model.layers.{i}.self_attn.{proj}"
    for i in range(total_layers-5, total_layers)
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
]

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,  # Directly specify modules here
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap model with LoRA
lora_model = get_peft_model(model, lora_config)

# Print trainable parameters
def print_trainable_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

print_trainable_params(lora_model)
# 5. Training Arguments -------------------------------------------

# Compute total training steps
num_train_epochs = 1
gradient_accumulation_steps = 4
batch_size=24  # Adjust based on GPU memory

# Calculate total steps
num_training_steps = (len(dataset["train"]) // batch_size) // gradient_accumulation_steps * num_train_epochs
num_warmup_steps = int(0.05 * num_training_steps)  # 10% warmup

optimizer = AdamW(lora_model.parameters(), lr=2e-4, weight_decay=0.01)

# Define cosine learning rate scheduler
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer, 
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# 5. Training Arguments -------------------------------------------
training_args = TrainingArguments(
    output_dir="/ephemeral/llama_qlora_4bit",  
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=100,
    per_device_train_batch_size=batch_size,  # Adjusted batch size
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=2e-4,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    bf16=True,
    logging_dir="/workspace/logs",
    logging_steps=10,
    report_to="wandb",
    remove_unused_columns=False,
    group_by_length=False,

    # A100-specific optimizations
    gradient_checkpointing=True,
    optim="adamw_torch",
    ddp_find_unused_parameters=False,

    # Data config
    dataloader_drop_last=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=8,

    # Push to Hugging Face Hub
    push_to_hub=True,
    hub_model_id="Billyyy/llama3_qlora_trained",
    hub_strategy="every_save",
)
# 6. Data Collator for Language Modeling -------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 7. Trainer Initialization with Cosine LR Scheduler -----------------------
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    optimizers=(optimizer, lr_scheduler),  
)

# 8. Train & Save -------------------------------------------------
trainer.train()
trainer.save_model("/ephemeral/llama3_qlora_trained")
trainer.push_to_hub()
print("QLoRA training complete! Model saved.")
