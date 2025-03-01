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
    Translates text from the source language to the target language using the NLLB model.
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
    Processes the prompt using the LLaMA model to generate a response.
    Ensures that only the generated output is returned.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        response_tokens = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )
    
    response_text = tokenizer.decode(response_tokens[0], skip_special_tokens=True)

    # Remove prompt if it's included in the response
    response_text = response_text.replace(prompt, "").strip()
    
    return response_text

def load_questions(filename):
    """
    Load multiple-choice questions from a JSON file.

    Args:
        filename (str): The path to the JSON file containing questions.

    Returns:
        list: A list of dictionaries, each representing a question.
    """
    with open(filename, 'r', encoding='utf-8') as file:
        questions = json.load(file)
    return questions

def extract_questions_answers_choices(questions):
    """
    Extract questions, their corresponding answer keys, and choices with labels.

    Args:
        questions (list): A list of dictionaries, each containing a question and its details.

    Returns:
        tuple: Three lists - questions, answer keys, and choices with labels.
    """
    question_texts = []
    answer_keys = []
    choices_with_labels = []

    for q in questions:
        question_texts.append(q['question'])
        answer_keys.append(q['answerKey'])
        choices = {choice['label']: choice['text'] for choice in q['choices']}
        choices_with_labels.append(choices)

    return question_texts, answer_keys, choices_with_labels

def clean_predicted_label(prediction):
    """
    Cleans the predicted label by keeping only a single uppercase letter (A, B, C, or D).
    """
    match = re.search(r"\b([A-D])\b", prediction)  # Looks for a standalone A, B, C, or D
    return match.group(1) if match else "INVALID"  # Returns the matched letter or "INVALID" if not found


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Authentication -----------------------------------------------
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Ensure token is set in environment
    # Login to Hugging Face Hub & Weights & Biases
    login(token=HF_API_TOKEN)

    # Load NLLB model and tokenizer
    print("Loading NLLB model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").to(device)
    nllb_tokenizer = AutoTokenizer.from_pretrained(args.nllb_model_name, src_lang=args.src_lang)
    nllb_model = PeftModel.from_pretrained(base_model, args.nllb_model_name).to(device)
    
    # Load LLaMA model and tokenizer
    print("Loading LLaMA model...")
    llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model_name)
    llama_model = AutoModelForCausalLM.from_pretrained(args.llama_model_name).to(device)

    # Load questions from the JSON file
    questions = load_questions(args.input_file)

    # Extract questions, answer keys, and choices with labels
    question_texts, answer_keys, choices_with_labels = extract_questions_answers_choices(questions)

    predictions = []
    correct_predictions = 0

    for i, question in tqdm(enumerate(question_texts), total=len(question_texts)):
        # Translate question
        translated_question = translate_text(
            question,
            nllb_tokenizer,
            nllb_model,
            args.src_lang,
            device
        )

        # Translate choices
        translated_choices = {}
        for label, choice in choices_with_labels[i].items():
            translated_choice = translate_text(
                choice,
                nllb_tokenizer,
                nllb_model,
                args.src_lang,
                device
            )
            translated_choices[label] = translated_choice


        instruction = "You are a helpful AI assistant. Read the question, which is translated from Mongolian and also provided in its original Mongolian version. Then answer the multiple-choice question with a single letter only."

        # Construct prompt for LLaMA
        prompt = f"{instruction}\n\nTranslated Question: {translated_question}\nOriginal Question: {question}\n\n"

        prompt += "Choices:\n"
        for label, choice in choices_with_labels[i].items():  # Iterate over original choices
            translated_choice = translated_choices[label]  # Get corresponding translated choice
            prompt += f"{label}: {choice} (Translated: {translated_choice})\n"

        prompt += "Answer (A, B, C, or D) ONLY: "


        llama_response = process_with_llama(
            prompt,
            llama_tokenizer,
            llama_model,
            device
        ).strip()
        
        print(prompt, llama_response)

        predicted_label = clean_predicted_label(llama_response)

        # Extract the predicted answer label
        is_correct = (predicted_label == answer_keys[i])
        if is_correct:
            correct_predictions += 1

        predictions.append({
            "question": question,
            "choices": choices_with_labels[i],
            "ground_truth": answer_keys[i],
            "prediction": predicted_label,
            "is_correct": is_correct
        })

    # Calculate accuracy
    total_questions = len(question_texts)
    accuracy = (correct_predictions / total_questions) * 100

    # Print accuracy
    print(f"Total Questions: {total_questions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Save predictions and ground truth to a JSON file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple-choice questions using NLLB and LLaMA models.")
    parser.add_argument('--nllb_model_name', type=str, required=True, help="Name or path of the NLLB model.")
    parser.add_argument('--llama_model_name', type=str, required=True, help="Name or path of the LLaMA model.")
    parser.add_argument('--src_lang', type=str, required=True, help="Source language code (e.g., 'khk_Cyrl').")
    parser.add_argument('--tgt_lang', type=str, required=True, help="Target language code (e.g., 'eng_Latn').")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file containing questions.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output JSON file with predictions.")
    args = parser.parse_args()

    main(args)
