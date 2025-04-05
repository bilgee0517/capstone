import os

# Set new cache directories
os.environ["HF_DATASETS_CACHE"] = "/ephemeral/hf_cache"
os.environ["HF_HOME"] = "/ephemeral/transformers_cache"
os.environ["TMPDIR"] = "/ephemeral/tmp"

# Ensure the directories exist
os.makedirs("/ephemeral/hf_cache", exist_ok=True)
os.makedirs("/ephemeral/transformers_cache", exist_ok=True)
os.makedirs("/ephemeral/tmp", exist_ok=True)

from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
import wandb

# 1. Authentication -----------------------------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Ensure token is set in environment
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Login to Hugging Face Hub & Weights & Biases
login(token=HF_API_TOKEN)
wandb.login(key=WANDB_API_KEY)

# Load base model and adapter
base_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")
model = PeftModel.from_pretrained(base_model, "Billyyy/mon_nllb_3.3B")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", 
                                            src_lang="khk_Cyrl", tgt_lang="eng_Latn")
 # Ensure LoRA parameters require gradients
for name, param in model.named_parameters():
    if "lora" in name:  # Ensure LoRA layers are trainable
        param.requires_grad = True

# Load your new dataset (JSON format) with 10K examples
# Note: This JSON should contain "input" and "output" keys.
dataset = load_dataset("json", data_files={"train": "../extracted_input_output.json"})

# Load evaluation dataset (ensure it's preprocessed or process it similarly)
eval_dataset = load_dataset("Billyyy/mn-en-parallel", split="eval")

# Define a preprocessing function that uses the correct keys:
def preprocess_function(examples):
    tokenizer.tgt_lang = "eng_Latn"
    # Here we assume the JSON keys are "input" and "output"
    model_inputs = tokenizer(
        examples["input"],
        text_target=examples["output"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    return model_inputs


def preprocess_function_eval(examples):
    tokenizer.tgt_lang = "eng_Latn"
    # Here we assume the JSON keys are "input" and "output"
    model_inputs = tokenizer(
        examples["src"],
        text_target=examples["tgt"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    return model_inputs


# Apply preprocessing on the train split from the JSON dataset
dataset["train"] = dataset["train"].map(preprocess_function, batched=True, num_proc=20, remove_columns=["input", "output"])

# (Optional) You might want to preprocess your eval_dataset similarly if needed.
# For example:
eval_dataset = eval_dataset.map(preprocess_function_eval, batched=True, num_proc=20, remove_columns=['src','tgt'])

# Create a DatasetDict with train and eval splits
dataset = DatasetDict({
    "train": dataset["train"],
    "eval": eval_dataset,  # Ensure eval_dataset is preprocessed or matches your training format
})

# Define training arguments
training_args = TrainingArguments(
    output_dir="/ephemeral/nllb_mn_en_lora_resume",
    run_name="nllb_mongolian_translation_resume",
    evaluation_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    lr_scheduler_type="cosine", 
    warmup_steps=10,  
    learning_rate=5e-5,
    num_train_epochs=4,
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
    hub_model_id="Billyyy/mn_nllb_3.3B_continue",
    hub_strategy="every_save",
)

# Initialize the Trainer (note the comma added after eval_dataset)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    tokenizer=tokenizer,
)

# Resume training from checkpoint
trainer.train()
trainer.save_model("/ephemeral/nllb_mn_en_lora_resume_trained")
