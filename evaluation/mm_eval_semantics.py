import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import PeftModel
import json
from tqdm import tqdm
import re
from huggingface_hub import login

def translate_text(text, tokenizer, model, src_lang, device):
    """
    Translates text using the NLLB model.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            max_length=256
        )
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

def process_with_llama(prompt, tokenizer, model, device):
    """
    Processes the prompt using LLaMA and ensures only the generated output is returned.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        response_tokens = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )
    
    response_text = tokenizer.decode(response_tokens[0], skip_special_tokens=True)

    # Remove prompt if included in the response
    response_text = response_text.replace(prompt, "").strip()
    
    return response_text

def clean_predicted_label(prediction):
    """
    Cleans the predicted label, ensuring it is only A, B, C, or D.
    """
    match = re.search(r"\b([A-D])\b", prediction)  # Extracts only A, B, C, or D
    return match.group(1) if match else "INVALID"

def load_questions(filename):
    """
    Load multiple-choice cloze questions from a JSON file.
    """
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def fill_in_the_blank(question, choice_text):
    """
    Replaces the blank ('_') in the question with a given choice.
    """
    return question.replace("_", choice_text)

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

    for i, question_data in tqdm(enumerate(questions[:10]), total=len(questions[:10])):
        original_question = question_data['question']
        choices = question_data['choices']
        answer_key = question_data['answerKey']

        # Create cloze-style questions by filling in the blank with each choice
        filled_questions = {choice['label']: fill_in_the_blank(original_question, choice['text']) for choice in choices}

        # Translate each filled-in question and its choice
        translated_filled_questions = {label: translate_text(filled_q, nllb_tokenizer, nllb_model, args.src_lang, device) 
                                       for label, filled_q in filled_questions.items()}

        instruction = "You are a helpful AI assistant. Read the multiple-choice cloze question and select the best option. Answer with only the correct letter (A, B, C, or D)."

        # Construct LLaMA prompt
        prompt = f"{instruction}\n\n"
        for label, translated_filled_q in translated_filled_questions.items():
            prompt += f"{label}: {translated_filled_q}\n"

        prompt += "Answer (A, B, C, or D) ONLY: "

        # Get LLaMA's response
        llama_response = process_with_llama(prompt, llama_tokenizer, llama_model, device).strip()
        
        # Extract the predicted answer label
        predicted_label = clean_predicted_label(llama_response)

        # Determine if prediction is correct
        is_correct = (predicted_label == answer_key)
        if is_correct:
            correct_predictions += 1
        print(correct_predictions)

        predictions.append({
            "question": original_question,
            "filled_questions": filled_questions,
            "translated_questions": translated_filled_questions,
            "choices": {choice["label"]: choice["text"] for choice in choices},
            "ground_truth": answer_key,
            "prediction": predicted_label,
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
    parser = argparse.ArgumentParser(description="Process multiple-choice cloze questions using NLLB and LLaMA models.")
    parser.add_argument('--nllb_model_name', type=str, required=True, help="Name or path of the NLLB model.")
    parser.add_argument('--llama_model_name', type=str, required=True, help="Name or path of the LLaMA model.")
    parser.add_argument('--src_lang', type=str, required=True, help="Source language code (e.g., 'khk_Cyrl').")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file containing questions.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output JSON file with predictions.")
    args = parser.parse_args()

    main(args)
