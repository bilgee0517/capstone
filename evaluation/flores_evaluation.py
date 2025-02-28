import os
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
from tqdm import tqdm
from datasets import load_dataset
from nltk.translate.meteor_score import meteor_score
from peft import PeftModel
import random
import numpy as np
from nltk.tokenize import word_tokenize  # Import tokenizer
import nltk

nltk.download('punkt_tab')
nltk.download('wordnet')

# Fix randomness for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load the FLORES-200 dataset
dataset = load_dataset("facebook/flores", "eng_Latn-khk_Cyrl", split="devtest")

def translate_batch(model, tokenizer, texts, max_length, device):
    """
    Translate a batch of texts using the specified model and tokenizer.
    Ensures deterministic output by disabling sampling.
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_length=max_length,
            use_cache=True  # Enables faster decoding
        )
    return tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

def evaluate_model(args):
    """
    Evaluate the translation model on the provided dataset and compute BLEU, chrF++, and METEOR scores.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading NLLB model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, src_lang=args.src_lang)
    model = PeftModel.from_pretrained(base_model, args.model_name).to(device)
    
    batch_size = args.batch_size
    num_samples = len(dataset)

    predictions = []
    references = []

    for i in tqdm(range(0, num_samples, batch_size), desc="Translating in Batches"):
        batch = dataset.select(range(i, min(i + batch_size, num_samples)))  # ✅ Proper batch selection

        src_texts = batch["sentence_khk_Cyrl"]  # Extract source texts correctly
        ref_texts = batch["sentence_eng_Latn"]  # Extract references correctly

        translated_texts = translate_batch(model, tokenizer, src_texts, args.max_length, device)

        predictions.extend(translated_texts)
        references.extend([[ref] for ref in ref_texts])  # sacrebleu expects a list of lists

    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(predictions, references, tokenize="intl")
    print(f"BLEU Score: {bleu.score:.2f}")

    # Compute chrF++ score
    chrf = sacrebleu.corpus_chrf(predictions, references, beta=2)
    print(f"chrF++ Score: {chrf.score:.2f}")
    
    #Compute METEOR score

    # Tokenize both references and predictions
    meteor_scores = [
        meteor_score([word_tokenize(ref[0])], word_tokenize(pred))  # Convert to tokenized lists
        for ref, pred in zip(references, predictions)
    ]

    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    print(f"METEOR Score: {avg_meteor_score:.4f}")

    results = {
        "Model Name": args.model_name,
        "BLEU": bleu.score,
        "chrF++": chrf.score,
        "METEOR": avg_meteor_score,
        "Predictions": predictions
    }

    output_file = "flores_results.json"
    
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_data = json.load(f)
        existing_data.append(results)
    else:
        existing_data = [results]

    with open(output_file, "w") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a translation model using the FLORES-200 dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the pre-trained model.")
    parser.add_argument("--src_lang", type=str, default="khk_Cyrl", help="Source language code (e.g., 'eng_Latn').")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum length for the generated translations.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for translation speed optimization.")
    args = parser.parse_args()

    evaluate_model(args)

if __name__ == "__main__":
    main()
