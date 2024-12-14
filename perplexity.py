import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_from_disk


# Collate function for batching
def collate_fn(batch):
    return {
        "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
        "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
    }


# Perplexity Calculation Function
def compute_perplexity(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    model.to(device)
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Compute loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Accumulate loss and number of tokens
            total_loss += loss.item() * input_ids.shape[0]
            total_tokens += torch.sum(attention_mask).item()
    
    # Compute perplexity
    perplexity = np.exp(total_loss / total_tokens)
    return perplexity


# Main Function
def main():
    parser = argparse.ArgumentParser(description="Compute Perplexity for a CausalLM Model")
    
    # Add arguments
    parser.add_argument("--model_name", type=str, required=True, help="Path or name of the pretrained model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the tokenized dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16 if args.device == "cuda" else torch.float32)
    
    # Load dataset
    print("Loading dataset...")
    tokenized_dataset = load_from_disk(args.dataset_path)
    
    # DataLoader setup
    print("Setting up DataLoader...")
    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Compute Perplexity
    print("Calculating perplexity...")
    perplexity = compute_perplexity(model, dataloader, args.device)
    print(f"Perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    main()
