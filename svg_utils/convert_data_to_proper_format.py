from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel

import json
import os
from typing import List, Dict, Any
import glob,re
import pandas as pd
from tqdm import tqdm

def create_instruction_variants() -> List[str]:
    """Create varied instruction prompts to improve model generalization"""
    return [
        "Generate an SVG image based on the following description",
        "Create an SVG illustration for this description",
        "Convert this text description into SVG code",
        "Generate SVG markup that represents this description",
        "Create an SVG graphic based on this text",
        "Design an SVG image for the following description"
    ]

def round_to_k_digits(svg: str, k: int = 0) -> str:
    """Rounds numbers in an SVG string to k decimal places."""
    def _round_match(match):
        return f"{float(match.group()):.{k}f}"
    return re.sub(r"\d+\.\d+", _round_match, svg)

def filter_svg_by_token_length(svg: str, tokenizer, max_tokens: int = 1000) -> bool:
    """Checks if the token count of an SVG is within the specified limit."""
    length = tokenizer(svg,return_tensors='pt')['input_ids'].shape[1]
    return length <= max_tokens


def filter_svg_by_text_length(svg: str, max_tokens: int = 1000) -> bool:
    """
    Based on some heurisitic, I've found that the ratio betbween tokens and text is token_length = text_length*2.3034459836975407 using BERT tokenizer.
    The tokenization process takes too long, so I'm just checking the text length.
     ."""
    return len(svg) <= max_tokens*2.3034459836975407

    


def convert_to_alpaca_format(input_files, chunk_size=1000,max_tokens = 1024):
    """Convert multiple MMSVG datasets to LLaMA-Factory Alpaca format"""
    instructions = create_instruction_variants()
    alpaca_data = []
    included_files = []
    excluded_files = []
    unused_indices = []

    tokenizer = AutoTokenizer.from_pretrained('apple/DiffuCoder-7B-cpGRPO',trust_remote_code=True)
    for file_path in tqdm(input_files, desc="Processing files"):
        print(f"Processing {file_path}...")
        df = pd.read_json(file_path)
        for i, item in tqdm(df.iterrows(), desc=f"Processing {os.path.basename(file_path)} - Alpaca samples: {len(alpaca_data)}", total=len(df), leave=False):
            try:
                description = item.get('description', '').strip()
                svg_code = item.get('svg', '').strip()

                if not description or not svg_code:
                    unused_indices.append((file_path, i))
                    if file_path not in excluded_files:
                        excluded_files.append(file_path)
                    continue

                processed_svg = round_to_k_digits(svg_code)
                instruction = instructions[len(alpaca_data) % len(instructions)]

                full_prompt = f"{instruction}\n\n{description}\n\n{processed_svg}"
                if filter_svg_by_token_length(full_prompt, tokenizer,max_tokens = max_tokens):
                    alpaca_data.append({
                        "instruction": instruction,
                        "input": description,
                        "output": processed_svg
                    })
                    if len(alpaca_data) % 100 == 0:
                        print(f'The length of alpaca data is now {len(alpaca_data)}')

            except Exception as e:
                print(f"Error processing {file_path}, index {i}: {e}")
                print(f"Item: {item}")
                print("")
                unused_indices.append((file_path, i))
                if file_path not in excluded_files:
                    excluded_files.append(file_path)

    print(f"\nSkipped {len(unused_indices)} items due to errors or missing data.")
    if unused_indices:
        print("Skipped indices (file, index):")
        for file_path, index in unused_indices:
            print(f"- {file_path}, index {index}")

    return alpaca_data, included_files, excluded_files

def main():
    # Find all JSON files matching the pattern
    # check the path under data/omnisvg_data
    json_files = glob.glob("/workspace/code/data/omnisvg_data/*.json")
    # json_files = glob.glob("*SVG*.json") or glob.glob("MMSVG*.json") or ["MMSVG-Illustration.json"]

    print(f"Found files: {json_files}")
    existing_files = [f for f in json_files if os.path.exists(f)]

    if not existing_files:
        print("No SVG JSON files found! Place your files in the current directory.")
        return
    max_tokens = 230
    print(f"Converting {len(existing_files)} files...")
    alpaca_data, included_files, excluded_files = convert_to_alpaca_format(existing_files,max_tokens=max_tokens)

    # Save results
    with open(f"svg_generation_dataset_smaller_than_{max_tokens}_tokens.json", 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)

    with open("dataset_info_entry.json", 'w', encoding='-utf-8') as f:
        json.dump({
            "svg_generation": {
                "file_name": "svg_generation_dataset.json",
                "columns": {"prompt": "instruction", "query": "input", "response": "output"}
            }
        }, f, indent=2)


    print(f"✅ Complete! {len(alpaca_data)} samples from {len(existing_files)} files")
    print(f"Included files logged in included_files.txt")
    print(f"Excluded files logged in excluded_files.txt")
    print("Next: Copy dataset_info_entry.json content to data/dataset_info.json")

if __name__ == "__main__":
    main() 