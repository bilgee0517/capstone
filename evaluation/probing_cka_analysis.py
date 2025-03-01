import torch
import torch.nn.functional as F
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


# 🔹 Compute CKA (Centered Kernel Alignment)
def compute_cka(X, Y):
    """
    Compute Centered Kernel Alignment (CKA) between two sets of representations.
    
    :param X: Tensor of hidden states from the baseline model
    :param Y: Tensor of hidden states from the fine-tuned model
    :return: CKA similarity score (float)
    """
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)
    X_norm = F.normalize(X, p=2, dim=1)
    Y_norm = F.normalize(Y, p=2, dim=1)
    return torch.mm(X_norm, Y_norm.T).mean().item()


# 🔹 Extract Hidden States from a Specific Decoder Layer
def extract_hidden_states(model, tokenizer, sentence, layer_idx):
    """
    Extract decoder hidden states from a specific layer.
    
    :param model: Transformer model (NLLB)
    :param tokenizer: Tokenizer for input processing
    :param sentence: Input sentence
    :param layer_idx: Decoder layer index to analyze
    :return: Hidden states from the specified layer
    """
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.decoder_hidden_states[layer_idx].squeeze(0).cpu().numpy()
    return hidden_states


# 🔹 Compute Layer-wise CKA Similarity
def compare_layerwise_cka(model_baseline, model_finetuned, tokenizer, sentence, num_layers):
    """
    Compute CKA between baseline and fine-tuned model's hidden states for all decoder layers.
    
    :param model_baseline: Original NLLB model
    :param model_finetuned: LoRA fine-tuned NLLB model
    :param tokenizer: Tokenizer for text input
    :param sentence: Input sentence for probing
    :param num_layers: Total number of decoder layers
    :return: Dictionary of layer-wise CKA scores
    """
    cka_scores = {}
    for layer_idx in range(num_layers):
        cka_score = compute_cka(
            torch.tensor(extract_hidden_states(model_baseline, tokenizer, sentence, layer_idx)),
            torch.tensor(extract_hidden_states(model_finetuned, tokenizer, sentence, layer_idx))
        )
        cka_scores[f"Layer_{layer_idx}"] = cka_score
        print(f"Layer {layer_idx}: CKA Similarity = {cka_score:.4f}")

    return cka_scores


# 🔹 Save Results to JSON File
def save_results(results, filename="result/cka_results.json"):
    """
    Saves CKA similarity scores to a JSON file.

    :param results: Dictionary containing CKA scores
    :param filename: Name of the JSON file
    """
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_data = json.load(f)
        existing_data.update(results)
    else:
        existing_data = results

    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"📂 CKA results saved to {filename}")


# 🔹 Main Execution
if __name__ == "__main__":
    # Load Models
    print("🚀 Loading models...")
    model_baseline = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")

    model_finetuned = PeftModel.from_pretrained(model_baseline, "Billyyy/mon_nllb_1.3B").to(device)

    # Example sentence for probing
    sentence = "Жош байшингийн засварын ажлыг туршиж үзэхээр шийджээ. Тэрээр байшин худалдаж авахад 80,000 доллар зарцуулж, дараа нь засвар хийхэд 50,000 доллар зарцуулжээ. Ингэснээр байшингийн үнэ цэнийг 150% -иар нэмэгдүүлсэн. Тэр хэр их ашиг олсон бэ?."

    # Number of decoder layers
    num_decoder_layers = len(model_baseline.model.decoder.layers)

    # Compute CKA for all layers
    print("\n🔍 Computing CKA for full LoRA model...")
    cka_scores = compare_layerwise_cka(model_baseline, model_finetuned, tokenizer, sentence, num_decoder_layers)

    # Save results
    save_results(cka_scores)
