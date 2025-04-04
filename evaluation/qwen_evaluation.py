import os
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sacrebleu
from tqdm import tqdm
from datasets import load_dataset
from nltk.translate.meteor_score import meteor_score
import random
import numpy as np
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("punkt_tab")

# Fix randomness for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load the FLORES-200 dataset
dataset = load_dataset("facebook/flores", "eng_Latn-khk_Cyrl", split="devtest", trust_remote_code=True)

def parse_translation(text):
    """
    Extract the English translation from the generated text.
    Assumes output format:
      "Instruction: Translate the following Mongolian sentence into English, and then explain your translation choices.
       Input: <source text>
       Output: English Translation: <translation>
       Explanation: <explanation>"
    """
    marker = "English Translation:"
    explanation_marker = "Explanation:"
    start = text.find(marker)
    if start == -1:
        return text.strip()
    start += len(marker)
    end = text.find(explanation_marker, start)
    if end == -1:
        translation = text[start:].strip()
    else:
        translation = text[start:end].strip()
    return translation

def build_prompt(src_text):
    """
    Build the prompt for Qwen evaluation.
    """
    prompt = (
        "Instruction: Translate the following Mongolian sentence into English, and then explain your translation choices.\n"
        f"Input: {src_text}\n"
        "Output:"
    )
    return prompt

def translate_batch(args, model, tokenizer, texts, max_length, device):
    """
    Translate a batch of texts by building prompts and generating responses.
    Then, extract only the English translation from each response.
    """
    # Build prompts for each source text
    prompts = [build_prompt(text) for text in texts]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_length=max_length,
            use_cache=True  # Enables faster decoding
        )
    outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    # Parse and extract only the English translation
    parsed_outputs = [parse_translation(text) for text in outputs]
    return parsed_outputs

def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 Loading fine-tuned Qwen model...")
    model = AutoModelForCausalLM.from_pretrained("Billyyy/qwen_mn_en_finetuned").to(device)
    tokenizer = AutoTokenizer.from_pretrained("Billyyy/qwen_mn_en_finetuned")
    
    batch_size = args.batch_size
    num_samples = len(dataset)

    predictions = []
    references = []

    for i in tqdm(range(0, num_samples, batch_size), desc="Translating in Batches"):
        batch = dataset.select(range(i, min(i + batch_size, num_samples)))
        src_texts = batch["sentence_khk_Cyrl"]   # Source Mongolian sentences
        ref_texts = batch["sentence_eng_Latn"]     # Reference English translations
    
        translated_texts = translate_batch(args, model, tokenizer, src_texts, args.max_length, device)
        print(translated_texts)
        predictions.extend(translated_texts)
        references.extend([[ref] for ref in ref_texts])  # sacrebleu expects a list of lists

    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(predictions, references, tokenize="intl")
    print(f"🎯 BLEU Score: {bleu.score:.2f}")

    # Compute chrF++ score
    chrf = sacrebleu.corpus_chrf(predictions, references, beta=2)
    print(f"📊 chrF++ Score: {chrf.score:.2f}")
    
    # Compute METEOR score
    meteor_scores = [
        meteor_score([word_tokenize(ref[0])], word_tokenize(pred))
        for ref, pred in zip(references, predictions)
    ]
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    print(f"📌 METEOR Score: {avg_meteor_score:.4f}")

    results = {
        "Model Name": "Billyyy/qwen_mn_en_finetuned",
        "BLEU": bleu.score,
        "chrF++": chrf.score,
        "METEOR": avg_meteor_score,
        "Predictions": predictions
    }

    output_file = "results/layer_wise_flores_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_data = json.load(f)
        existing_data.append(results)
    else:
        existing_data = [results]

    with open(output_file, "w") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    
    print(f"📂 Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned Qwen translation model on FLORES-200.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length for generated translations.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for translation.")
    args = parser.parse_args()
    evaluate_model(args)

if __name__ == "__main__":
    main()
