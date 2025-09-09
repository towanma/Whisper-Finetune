
import argparse
import json
from transformers import WhisperProcessor
from tqdm import tqdm

def filter_long_audio(input_file, output_file, model_path, max_tokens=448):
    """
    Filters out data samples where the tokenized sentence is too long.
    """
    print(f"Loading tokenizer for model: {model_path}...")
    # 使用您计划训练的模型来加载对应的tokenizer
    processor = WhisperProcessor.from_pretrained(model_path)
    
    print(f"Processing file: {input_file}...")
    
    original_lines = 0
    filtered_lines = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, desc="Filtering data"):
            original_lines += 1
            try:
                data = json.loads(line)
                sentence = data.get("sentence")
                
                if not sentence:
                    continue
                    
                tokens = processor.tokenizer(sentence, add_special_tokens=True).input_ids
                
                if len(tokens) <= max_tokens:
                    outfile.write(line)
                    filtered_lines += 1
                    
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line: {line.strip()}")

    removed_count = original_lines - filtered_lines
    print("\nFiltering complete.")
    print(f"Original number of lines: {original_lines}")
    print(f"Number of lines after filtering: {filtered_lines}")
    print(f"Number of lines removed: {removed_count}")
    print(f"Filtered data saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter long audio data based on token length.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Whisper model/tokenizer (e.g., ./whisper-large-v3).")
    parser.add_argument("--train_input", type=str, default="dataset/train.json", help="Input training data JSON file.")
    parser.add_argument("--train_output", type=str, default="dataset/train_filtered.json", help="Output filtered training data JSON file.")
    parser.add_argument("--test_input", type=str, default="dataset/test.json", help="Input test data JSON file.")
    parser.add_argument("--test_output", type=str, default="dataset/test_filtered.json", help="Output filtered test data JSON file.")
    parser.add_argument("--max_tokens", type=int, default=448, help="Maximum number of tokens allowed for a sentence.")
    
    args = parser.parse_args()
    
    # Filter training data
    filter_long_audio(args.train_input, args.train_output, args.model_path, args.max_tokens)
    
    print("-" * 20)
    
    # Filter test data
    filter_long_audio(args.test_input, args.test_output, args.model_path, args.max_tokens)
