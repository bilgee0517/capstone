import os
import asyncio
import openai
import aiofiles
import json
from datasets import load_dataset
from openai import OpenAI
from tqdm.asyncio import tqdm
from collections import deque
import time

# Global deque to store timestamps of requests
last_call_times = deque()
RATE_LIMIT = 500
TIME_WINDOW = 60  # seconds

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

# Model name (replace with actual one if different)
MODEL_NAME = "gpt-4o-mini-2024-07-18"  # Assuming you're using GPT-4o mini, otherwise adjust


semaphore = asyncio.Semaphore(500)  # Limit to 500 concurrent requests
async def translate_sentence(sentence: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        # Rate limiting
        while True:
            now = time.time()
            # Remove timestamps older than 60 seconds
            while last_call_times and now - last_call_times[0] > TIME_WINDOW:
                last_call_times.popleft()

            if len(last_call_times) < RATE_LIMIT:
                last_call_times.append(now)
                break
            else:
                await asyncio.sleep(0.05)  # wait a bit and try again

        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a translator. Translate the following Mongolian sentence into English. "
                            "Only output the translated English sentence without any explanation or formatting."
                        )
                    },
                    {"role": "user", "content": sentence}
                ],
                temperature=0.0
            )
            translation = response.choices[0].message.content.strip()
            return translation
        except Exception as e:
            print(f"Error translating sentence: {sentence}\nError: {e}")
            return ""


async def translate_sentences(sentences: list, concurrency: int = 5) -> list:
    """
    Translate a list of sentences concurrently with a given concurrency limit.
    Displays a progress bar to monitor the translation progress.
    """
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [translate_sentence(sentence, semaphore) for sentence in sentences]

    translations = []
    for future in tqdm.as_completed(tasks, total=len(tasks), desc="Translating"):
        translation = await future
        translations.append(translation)

    return translations


async def main():
    # Load the HF dataset
    dataset = load_dataset("Billyyy/mn-en-parallel", split="train")

    # Shuffle and sample 100K or fewer if dataset is smaller
    dataset = dataset.shuffle(seed=42)
    num_samples = min(50_000, len(dataset))
    dataset = dataset.select(range(num_samples))

    # Extract source sentences
    src_sentences = dataset["src"]
    print(f"🚀 Translating {len(src_sentences)} sentences...")

    # Translate
    translations = await translate_sentences(src_sentences, concurrency=10)

    # Save as a {"src": [...], "tgt": [...]} dictionary
    output_data = {"src": src_sentences, "tgt": translations}
    output_file = "updated_billyyy_mn_en_parallel.json"

    async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
        await f.write(json.dumps(output_data, ensure_ascii=False, indent=2))

    print(f"✅ Translation complete. Saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
