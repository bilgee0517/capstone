from datasets import load_dataset
import sentencepiece as spm
from transformers import PreTrainedTokenizerFast
import os
import re
import unicodedata
from huggingface_hub import login
from transformers import BloomTokenizerFast

# 1. Load Dataset -------------------------------------------------
dataset = load_dataset("Billyyy/cleaned-mongolian-dataset")

# 2. Text Cleaning and Preprocessing ------------------------------
def clean_text(text):
    """Ensure valid UTF-8 and normalize Cyrillic characters"""
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = unicodedata.normalize('NFC', text)  # Normalize Cyrillic
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # Remove control chars
    return text

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # Hugging Face token
CYRILLIC_SUFFIXES = [
    "ийг", "ыг", "д", "т", "аас", "ээс", "ын", "ийн", "тай", "той",
    "ууд", "үүд", "чууд", "ж", "ч", "сан", "жээ", "лаа", "лээ", "аар"
]

def add_morph_boundaries(batch):
    """Insert morphological markers for agglutinative suffixes"""
    texts = []
    for text in batch["text"]:
        text = clean_text(text)  # Ensure clean UTF-8
        
        # Add space between words for SentencePiece
        text = re.sub(r"(\w)([^\w⨎])", r"\1 \2", text)
        text = re.sub(r"([^\w⨎])(\w)", r"\1 \2", text)
        
        # Mark morphological boundaries
        for suffix in CYRILLIC_SUFFIXES:
            text = re.sub(fr"(\w)({suffix})(\W|$)", r"\1⨎\2\3", text)
        
        texts.append(text)
    return {"processed_text": texts}

# Process in batches for efficiency
processed_dataset = dataset.map(
    add_morph_boundaries,
    batched=True,
    batch_size=1000,
    num_proc=os.cpu_count()-1,
    remove_columns=dataset["train"].column_names
)

# 3. Train SentencePiece Model ------------------------------------
def train_sp_tokenizer():
    # Save processed text to temporary file
    with open("mongolian_temp.txt", "w", encoding='utf-8') as f:
        for text in processed_dataset['train']["processed_text"]:
            f.write(text + "\n")

    # SentencePiece configuration for Cyrillic Mongolian
    spm.SentencePieceTrainer.train(
        input="mongolian_temp.txt",
        model_prefix="mongolian-cyrillic",
        vocab_size=8000,
        model_type="unigram",
        character_coverage=0.9999,
        num_threads=os.cpu_count()-1,
        max_sentencepiece_length=8,
        split_by_whitespace=True,
        user_defined_symbols=CYRILLIC_SUFFIXES + ["⨎"],
        control_symbols=["<|endoftext|>"],
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
        normalization_rule_name="nfkc",  # Use strict Unicode normalization
        allow_whitespace_only_pieces=False,
        remove_extra_whitespaces=True,
        split_by_unicode_script=True,
        max_sentence_length=8000
    )

def create_hf_tokenizer():
    # Ensure the SentencePiece model file exists
    if not os.path.exists("mongolian-cyrillic.model"):
        raise FileNotFoundError("SentencePiece model file not found!")
    
    # Verify the model file is valid
    try:
        sp = spm.SentencePieceProcessor()
        sp.Load("mongolian-cyrillic.model")
        print("SentencePiece model loaded successfully!")
    except Exception as e:
        raise ValueError(f"Invalid SentencePiece model: {e}")

    # Load the BLOOMZ tokenizer
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloomz-3b")

    # Extract tokens from SentencePiece model
    spm_vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

    # Remove duplicates (some might already exist in BLOOMZ tokenizer)
    new_tokens = list(set(spm_vocab) - set(tokenizer.get_vocab().keys()))

    # Add new tokens to the BLOOMZ tokenizer
    tokenizer.add_tokens(new_tokens)

    print(f"Added {len(new_tokens)} new tokens to the tokenizer.")

    return tokenizer

# 5. Quality Validation -------------------------------------------
def validate_tokenizer(tokenizer):
    test_samples = [
        ("номынхандаа", ["ном", "⨎ын", "⨎хан", "⨎даа"]),
        ("сургуулиудтайгаа", ["сургууль", "⨎ууд", "⨎тай", "⨎гаа"]),
        ("хичээлдээ", ["хичээл", "⨎дээ"])
    ]
    
    passed = 0
    for word, expected in test_samples:
        tokens = tokenizer.tokenize(word)
        if tokens == expected:
            passed += 1
        else:
            print(f"Failed: {word} → {tokens} (expected {expected})")
    
    print(f"Validation: {passed}/{len(test_samples)} passed")

# Main Execution --------------------------------------------------
if __name__ == "__main__":
    # Train and save tokenizer
    train_sp_tokenizer()
    tokenizer = create_hf_tokenizer()
    
    # Validate
    validate_tokenizer(tokenizer)
    
    login(token=HF_API_TOKEN)
    
    # Save to Hub
    tokenizer.push_to_hub(
        "Billyyy/bloomz_mn",
        commit_message="Add agglutination-optimized tokenizer"
    )