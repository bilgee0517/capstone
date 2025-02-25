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
    DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login

# 1. Authentication -----------------------------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Ensure token is set in environment
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Login to Hugging Face Hub & Weights & Biases
login(token=HF_API_TOKEN)
wandb.login(key=WANDB_API_KEY)

# 2. Load Pretrained Tokenizer & Model -----------------------------
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Change if using another model
TOKENIZER_PATH = "Billyyy/llama_8K_extended" # Path to your trained tokenizer
DATASET_PATH = "/workspace/labeled_corpus.txt"  # TXT dataset path


def merge_sequences_with_text(dataset, tokenizer, max_length=1024):
    """
    Merge sequences in a dataset while keeping 'text' and 'input_ids' aligned.
    
    Args:
        dataset (Dataset): Hugging Face Dataset with 'text' and 'input_ids'.
        tokenizer: Tokenizer used to tokenize the text.
        max_length (int): Maximum length for merged sequences.

    Returns:
        dict: A dictionary with merged 'text' and 'input_ids'.
    """
    merged_texts = []
    merged_input_ids = []
    current_text = []
    current_input_ids = []

    for text, input_ids in zip(dataset["text"], dataset["input_ids"]):
        # Check if adding the current sequence exceeds the max length
        if len(current_input_ids) + len(input_ids) > max_length:
            # Append the merged sequences
            merged_texts.append(" ".join(current_text))
            merged_input_ids.append(current_input_ids[:max_length])
            # Reset for the next sequence
            current_text = []
            current_input_ids = []
        
        # Extend the current sequence
        current_text.append(text)
        current_input_ids.extend(input_ids)
    
    # Add the final batch if it exists
    if current_input_ids:
        merged_texts.append(" ".join(current_text))
        merged_input_ids.append(current_input_ids[:max_length])

    return {"text": merged_texts, "input_ids": merged_input_ids}


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
vocab_size = len(tokenizer)

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 3. Resize Model's Embedding Layer -------------------------------
old_vocab_size = model.get_input_embeddings().num_embeddings

if vocab_size > old_vocab_size:
    print(f"Resizing model embeddings from {old_vocab_size} → {vocab_size}")
    model.resize_token_embeddings(vocab_size)

# Freeze all layers except embedding
for param in model.parameters():
    param.requires_grad = False  # Freeze entire model

# Enable training for embedding layer only
model.get_input_embeddings().weight.requires_grad = True

# 4. Tokenize & Convert TXT Dataset --------------------------------
def load_txt_dataset(txt_file):
    """Reads a text file and converts it into a Hugging Face Dataset."""
    with open(txt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines

    dataset = Dataset.from_dict({"text": lines})  # Convert to HF dataset
    return dataset

def tokenize_function(examples):
    """Tokenizes each line in the dataset."""
    return tokenizer(
        examples["text"], truncation=True, max_length=256
    )

# Load and tokenize dataset
dataset = load_txt_dataset(DATASET_PATH)
# Split into train (90%) and evaluation (10%)
dataset_split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=27)
eval_dataset = eval_dataset.map(tokenize_function, batched=True, num_proc=27)

# Convert into DatasetDict for Trainer API
dataset = DatasetDict({
    "train": train_dataset,
    "eval": eval_dataset
})

# 5. Training Arguments -------------------------------------------
training_args = TrainingArguments(
    output_dir="/ephemeral/llama_3B_translation",  
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    weight_decay=0.01,
    fp16=True,
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
    dataloader_num_workers=8
)

# 6. Data Collator for Masked Language Modeling -------------------
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 7. Trainer Initialization ---------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"], 
    data_collator=data_collator
)

# 8. Train & Save -------------------------------------------------
trainer.train()
trainer.save_model("/workspace/llama3_embedding_trained")

print("Training complete! Model saved.")
