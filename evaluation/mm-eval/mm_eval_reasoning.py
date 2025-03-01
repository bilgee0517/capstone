import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import PeftModel
import json
from tqdm import tqdm
from huggingface_hub import login
import re

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
    Processes the prompt using the LLaMA model to generate a response.
    Extracts the answer inside \boxed{} while ensuring the response does not include the prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        response_tokens = model.generate(
            **inputs,
            max_new_tokens=1024,  # Allow space for full reasoning
            do_sample=False
        )
    
    response_text = tokenizer.decode(response_tokens[0], skip_special_tokens=True)

    # Remove the prompt if it's included in the response
    if response_text.startswith(prompt):
        response_text = response_text[len(prompt):].strip()

    # Extract only the answer inside \boxed{}
    match = re.search(r"\\boxed\{(.*?)\}", response_text, re.DOTALL)
    extracted_answer = match.group(1).strip() if match else "INVALID"  # Return extracted answer if found

    return response_text, extracted_answer

def load_questions(filename):
    """
    Load open-ended questions from a JSON file.
    """
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Authentication -----------------------------------------------
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    login(token=HF_API_TOKEN)

    correct_predictions = 0  # Initialize counter


    # Load NLLB model
    print("Loading NLLB model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").to(device)
    nllb_tokenizer = AutoTokenizer.from_pretrained(args.nllb_model_name, src_lang=args.src_lang)
    nllb_model = PeftModel.from_pretrained(base_model, args.nllb_model_name).to(device)
    
    # Load DeepSeek model
    print("Loading DeepSeek LLaMA model...")
    llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model_name)
    llama_model = AutoModelForCausalLM.from_pretrained(args.llama_model_name).to(device)

    # Load questions
    questions = load_questions(args.input_file)

    predictions = []

    for i, question_data in tqdm(enumerate(questions), total=len(questions)):
        original_question = question_data['question']
        ground_truth = question_data.get('answer', 'N/A')  # Ensure there's an answer key

        # Step 1: **Translate question**
        translated_question = translate_text(original_question, nllb_tokenizer, nllb_model, args.src_lang, device)

        # Step 2: **Construct the prompt**
        instruction = (
            "You are a AI assistant specialized in solving translated math problems. Expect bad translation but solve it with best assumptions"
            "Solve the problem by first rewriting the question to yourself. Then, explicitly format ONLY YOUR FINAL ANSWER as follows:\n"
            "Answer:\\boxed{YOUR_ANSWER}"
        )

        prompt = f"{instruction}\n\nTranslated Problem: {translated_question}"

        # Step 3: **Generate response**
        llama_response, extracted_answer = process_with_llama(prompt, llama_tokenizer, llama_model, device)
        
        print(f"{llama_response}, EXTRACTED_ANSWER:{extracted_answer}")
        if not extracted_answer.isdigit():
            print(f"Skipping question {i+1}: Extracted answer is not an integer -> {extracted_answer}")
            continue  # Skip this iteration

            # Compare extracted answer with ground truth
        is_correct = (int(extracted_answer) == int(ground_truth))
        if is_correct:
            correct_predictions += 1

        print(correct_predictions)
        predictions.append({
            "model_name": args.llama_model_name,
            "original_question": original_question,
            "translated_question": translated_question,
            "raw_llama_response": llama_response,
            "prediction": extracted_answer,  # Extracted \boxed{} answer
        })
    
    # Save predictions
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve open-ended math problems using NLLB and DeepSeek-R1.")
    parser.add_argument('--nllb_model_name', type=str, required=True, help="Name or path of the NLLB model.")
    parser.add_argument('--llama_model_name', type=str, required=True, help="Name or path of the DeepSeek-R1 model.")
    parser.add_argument('--src_lang', type=str, required=True, help="Source language code (e.g., 'khk_Cyrl').")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file containing questions.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output JSON file with predictions.")
    args = parser.parse_args()

    main(args)
