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
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login

# 1. Authentication -----------------------------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Ensure token is set in environment
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Login to Hugging Face Hub & Weights & Biases
login(token=HF_API_TOKEN)
wandb.login(key=WANDB_API_KEY)

# 2. Load Pretrained Tokenizer & Model -----------------------------
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
DATASET_PATH = "Billyyy/mn-en-parallel"  # Preprocessed dataset path

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="khk_Cyrl", tgt_lang="eng_Latn")

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, device_map="auto")

# 3. Apply LoRA Adapters --------------------------------------------
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=32,  # Rank of LoRA matrix
    lora_alpha=64,  # Scaling factor
    lora_dropout=0.1,  # Dropout for regularization
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)

# Ensure LoRA parameters require gradients
for name, param in model.named_parameters():
    if "lora" in name:  # Ensure LoRA layers are trainable
        param.requires_grad = True

# Print trainable parameters for debugging
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params} || Total params: {total_params} || Trainable%: {100 * trainable_params / total_params:.4f}")

# 4. Load & Tokenize Dataset ----------------------------------------
dataset = load_dataset(DATASET_PATH)

def preprocess_function(examples):
    """Ensure the dataset is properly tokenized."""
    tokenizer.tgt_lang = "eng_Latn"  # Ensure the target language is set

    # Tokenize both source and target in one call
    model_inputs = tokenizer(
        examples["src"],
        text_target=examples["tgt"], 
        truncation=True,
        max_length=256,
        padding="max_length"
    )

    return model_inputs

# Tokenize dataset
train_dataset = dataset["train"].map(preprocess_function, batched=True, num_proc=20,remove_columns=["src", "tgt"])
eval_dataset = dataset["validation"].map(preprocess_function, batched=True, num_proc=20,remove_columns=["src", "tgt"])

# Convert into DatasetDict for Trainer API
dataset = DatasetDict({
    "train": train_dataset,
    "validation": eval_dataset
})

print(f" Printing dataset {dataset['train'][0]}")

model.config.use_cache = False

# 5. Training Arguments -------------------------------------------
training_args = TrainingArguments(
    output_dir="/ephemeral/nllb_mn_en_lora_r32",
    run_name="nllb_mongolian_translation",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=10,
    per_device_train_batch_size=40,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    lr_scheduler_type="cosine", 
    warmup_steps=500,  
    learning_rate=1e-4,
    num_train_epochs=2,
    weight_decay=0.01,
    fp16=True,
    logging_dir="/workspace/logs",
    logging_steps=10,
    report_to="wandb",
    remove_unused_columns=False,
    label_names=["labels"],

    optim="adamw_torch",
    ddp_find_unused_parameters=False,

    # Data config
    dataloader_drop_last=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=8,
    
    # Push to Hugging Face Hub
    push_to_hub=True,
    hub_model_id="Billyyy/mon_nllb_3B_r32",
    hub_strategy="every_save",

    
)

# 6. Data Collator for Seq2Seq -----------------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id = 1)

# 7. Trainer Initialization ---------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"], 
    data_collator=data_collator,
    tokenizer=tokenizer
)

model.train()

# 8. Train & Save -------------------------------------------------
trainer.train()
trainer.save_model("/ephemeral/nllb_mn_en_lora_trained")

print("✅ LoRA Fine-tuning complete! Model saved at `/workspace/nllb_mn_en_lora_trained`")
