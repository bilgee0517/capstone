import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import argparse
import os
import json

class LogitLensAnalyzer:
    def __init__(self, peft_model=None, device="cpu", top_k=5, num_tokens=20):
        """
        Initializes the Logit Lens analysis for NLLB models.
        """
        self.device = device
        self.top_k = top_k  # Number of top tokens to extract per layer
        self.num_tokens = num_tokens  # Number of tokens to analyze
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang="khk_Cyrl")
        self.hidden_states_per_step = []  # Store hidden states at each step
        self.peft_model = peft_model

        if self.peft_model:
            # Load the base model and LoRA-adapted model
            base_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").to(device)
            self.model = PeftModel.from_pretrained(base_model, peft_model).to(device)
        else:
            # Load standard NLLB model
            self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").to(device)

        # Automatically detect where the decoder layers are located
        self.decoder_layers = self._find_decoder_layers()
        self.lm_head = self.model.lm_head  # The final layer that maps hidden states to logits

    def _find_decoder_layers(self):
        """Finds the correct path to the decoder layers dynamically."""
        if self.peft_model:
            print("LoRA model detected: Using `self.model.base_model.model.decoder.layers`")
            return self.model.model.model.decoder.layers  # LoRA-adapted model
        else:
            print("Base model detected: Using `self.model.model.decoder.layers`")
            return self.model.model.decoder.layers  # Standard model

    def hook_fn(self, module, input, output):
        """Captures hidden states from all decoder layers."""
        self.hidden_states_per_step[-1].append(output[0])  # Extract the main hidden state

    def register_hooks(self):
        """Registers hooks for ALL decoder layers."""
        self.hooks = []
        for layer in self.decoder_layers:
            hook = layer.register_forward_hook(self.hook_fn)
            self.hooks.append(hook)

    def remove_hooks(self):
        """Removes hooks after execution."""
        for hook in self.hooks:
            hook.remove()

    def extract_logits_autoregressive(self, input_text):
        """
        Runs input text through the model and extracts hidden states while generating tokens autoregressively.
        """
        self.hidden_states_per_step = []
        self.register_hooks()

        # Tokenize the input and move to device
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Prepare encoder outputs (do this once to avoid redundant computation)
        with torch.no_grad():
            encoder_outputs = self.model.get_encoder()(**inputs, return_dict=True)
        
        # Initialize decoder with the correct start token
        if self.model.config.decoder_start_token_id is not None:
            decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id, 256047]], device=self.device) ##starting with decoder token and then eng_Latn token

        past_key_values = None  # Store cached past key values for efficient decoding
        generated_tokens = []

        print(f"\n🔹 **Starting autoregressive generation for input:** '{input_text}'\n")

        with torch.no_grad():
            for step in range(self.num_tokens):
                self.hidden_states_per_step.append([])  # Prepare storage for this step

            # Forward pass with past key values
                outputs = self.model(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,  # Enables faster decoding
                    output_hidden_states=True,  # Ensures hidden states are returned
                    return_dict=True
                )


                logits = outputs.logits[:, -1, :]  # Logits for the last generated token
                past_key_values = outputs.past_key_values  # Update cache for next step

                # Store hidden states for this step
                self.hidden_states_per_step[-1] = outputs.decoder_hidden_states  # Capture hidden states for this step

                # Select next token (greedy decoding)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated_tokens.append(next_token.item())

                # Convert token ID to text and print debug info
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print(f"🔹 **Step {step+1}: Generated Token ID:** {next_token.item()} -> '{self.tokenizer.decode(next_token.item())}'")
                print(f"   **Current Generated Text:** '{generated_text}'\n")

                # Append new token to decoder input for next step
                decoder_input_ids = next_token  # Keep only the latest token

        self.remove_hooks()
        return self.hidden_states_per_step, generated_text  # Return all recorded hidden states and generated tokens

    def get_top_k_predictions(self, hidden_states):
        """Extracts top-k predictions for each layer at each step."""
        top_k_predictions = []

        for layer_states in hidden_states:  # Iterate over decoder layers
            logits = self.lm_head(layer_states[:, -1, :])  # Compute logits
            top_k_values, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
            decoded_tokens = [self.tokenizer.decode(idx.item()) for idx in top_k_indices[0]]
            top_k_predictions.append(decoded_tokens)

        return top_k_predictions  # List of [layer][top-k words]

    def compare_models(self, base_hidden_states, lora_hidden_states):
        """
        Compares hidden states from base and LoRA models and extracts logit predictions.
        """
        min_steps = min(len(base_hidden_states), len(lora_hidden_states))  # Ensure matching lengths
        similarities = []
        base_top_k_tokens = []
        lora_top_k_tokens = []

        for step in range(min_steps):  # Step-wise comparison
            base_step_states = base_hidden_states[step]
            lora_step_states = lora_hidden_states[step]

            step_similarities = []

            # Compute top-k token predictions
            base_top_k_tokens.append(self.get_top_k_predictions(base_step_states))
            lora_top_k_tokens.append(self.get_top_k_predictions(lora_step_states))

            for base_h, lora_h in zip(base_step_states, lora_step_states):
                # Compute cosine similarity
                cos_sim = F.cosine_similarity(base_h, lora_h, dim=-1).mean().item()
                step_similarities.append(cos_sim)

            similarities.append(step_similarities)

        return similarities, base_top_k_tokens, lora_top_k_tokens
    
def save_comparison_results(similarities, base_model_text, lora_text,  base_top_k, lora_top_k, input_text, filename):
    """Save model comparison results to a JSON file."""
    results = {
        "input_text": input_text,
        "base_model_text":base_model_text,
        "lora_text": lora_text,
        "similarities": similarities,
        "base_top_k": base_top_k,
        "lora_top_k": lora_top_k
    }

    filepath = os.path.join(filename)
    with open(filename, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Comparison results saved to {filepath}")

# Example usage
def main(args):
    base_model_analyzer = LogitLensAnalyzer() 
    lora_model_analyzer = LogitLensAnalyzer(peft_model=args.model_name)
    input_text = "Жош байшингийн засварын ажлыг туршиж үзэхээр шийджээ. Тэрээр байшин худалдаж авахад 80,000 доллар зарцуулж, дараа нь засвар хийхэд 50,000 доллар зарцуулжээ. Ингэснээр байшингийн үнэ цэнийг 150% -иар нэмэгдүүлсэн. Тэр хэр их ашиг олсон бэ?"
    # Extract hidden states for both models while preserving autoregressive behavior
    base_hidden_states, base_generated_text = base_model_analyzer.extract_logits_autoregressive(input_text)
    lora_hidden_states, lora_generated_text = lora_model_analyzer.extract_logits_autoregressive(input_text)

    # Compare models
    similarities, base_top_k, lora_top_k = base_model_analyzer.compare_models(base_hidden_states, lora_hidden_states)
    save_comparison_results(similarities, base_generated_text, lora_generated_text, base_top_k, lora_top_k, input_text, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple-choice questions using NLLB and LLaMA models.")
    parser.add_argument('--model_name', type=str, required=True, help="Name or path of the peft model.")
    # parser.add_argument('--input_text', type=str, required=True, help="Enter input text")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output JSON file with predictions.")
    args = parser.parse_args()

    main(args)