import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import PeftModel
import json
from tqdm import tqdm
import re
import nltk
from huggingface_hub import login

nltk.download('punkt')  # Needed for sentence tokenization

def translate_text(text, tokenizer, model, src_lang, device):
    """
    Translates text from the source language using the NLLB model.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            max_length=256
        )
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

def process_with_llama(prompt, tokenizer, model, device):
    """
    Processes the prompt using LLaMA and ensures only the answer is returned.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        response_tokens = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )
    
    response_text = tokenizer.decode(response_tokens[0], skip_special_tokens=True)

    # Remove prompt if it's included in the response
    response_text = response_text.replace(prompt, "").strip()
    
    return response_text

def split_long_question(question, max_length=80):
    """
    Splits long questions into sentences and returns a list of sentences.
    """
    sentences = nltk.sent_tokenize(question)
    return sentences if sum(len(s) for s in sentences) > max_length else [question]

def clean_numeric_prediction(prediction):
    """
    Extracts only the numeric answer from the LLaMA response.
    """
    match = re.search(r"\b\d+\b", prediction)  # Extracts the first number in the response
    return match.group(0) if match else "INVALID"

def load_questions(filename):
    """
    Loads mathematical word problems from a JSON file.
    """
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Authentication -----------------------------------------------
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    login(token=HF_API_TOKEN)

    # Load NLLB model
    print("Loading NLLB model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").to(device)
    nllb_tokenizer = AutoTokenizer.from_pretrained(args.nllb_model_name, src_lang=args.src_lang)
    nllb_model = PeftModel.from_pretrained(base_model, args.nllb_model_name).to(device)
    
    # Load LLaMA model
    print("Loading LLaMA model...")
    llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model_name)
    llama_model = AutoModelForCausalLM.from_pretrained(args.llama_model_name).to(device)

    # Load questions
    questions = load_questions(args.input_file)

    predictions = []
    correct_predictions = 0

    for i, question_data in tqdm(enumerate(questions), total=len(questions)):
        original_question = question_data['question']
        correct_answer = question_data['answer']

        # Step 1: **Break down long questions into sentences**
        question_sentences = split_long_question(original_question)
        
        # Step 2: **Translate each sentence separately**
        translated_sentences = [
            translate_text(sentence, nllb_tokenizer, nllb_model, args.src_lang, device) 
            for sentence in question_sentences
        ]

        # Step 3: **Reconstruct the full translated question**
        translated_question = " ".join(translated_sentences)

        instruction = "Solve the following math problem which is translated from Mongolian with the original. Think step by step. Break down the problem before solving. Then provide only the numeric answer, DO NOT INCLUDE ANYTHING ELSE."

        # Construct LLaMA prompt
        prompt = f"{instruction}\n\nProblem: {translated_question}\n\nAnswer: "

        # Get LLaMA's response
        llama_response = process_with_llama(prompt, llama_tokenizer, llama_model, device).strip()

        # Extract only the numeric part
        predicted_answer = clean_numeric_prediction(llama_response)

        # Determine if prediction is correct
        is_correct = (predicted_answer == correct_answer)
        if is_correct:
            correct_predictions += 1

        print(correct_predictions)
        predictions.append({
            "original_question": original_question,
            "split_sentences": question_sentences,
            "translated_question": translated_question,
            "ground_truth": correct_answer,
            "prediction": predicted_answer,
            "is_correct": is_correct
        })

    # Calculate accuracy
    total_questions = len(questions)
    accuracy = (correct_predictions / total_questions) * 100

    print(f"Total Questions: {total_questions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Save predictions
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve mathematical word problems using NLLB and LLaMA.")
    parser.add_argument('--nllb_model_name', type=str, required=True, help="Name or path of the NLLB model.")
    parser.add_argument('--llama_model_name', type=str, required=True, help="Name or path of the LLaMA model.")
    parser.add_argument('--src_lang', type=str, required=True, help="Source language code (e.g., 'khk_Cyrl').")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file containing questions.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output JSON file with predictions.")
    args = parser.parse_args()

    main(args)
