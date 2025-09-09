
import argparse
import os
import json
from tokenizers import trainers, Tokenizer, models, pre_tokenizers
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

def create_corpus_from_json(json_files, corpus_path):
    """
    Reads one or more JSON lines files and extracts the 'sentence' field 
    to create a temporary text corpus file for tokenizer training.
    """
    print(f"Creating text corpus from {json_files} at {corpus_path}...")
    with open(corpus_path, 'w', encoding='utf-8') as corpus_file:
        for json_file_path in json_files:
            print(f"Processing {json_file_path}...")
            with open(json_file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Extracting text from {os.path.basename(json_file_path)}"):
                    try:
                        data = json.loads(line)
                        sentence = data.get("sentence")
                        if sentence:
                            corpus_file.write(sentence + "\n")
                    except json.JSONDecodeError:
                        print(f"Skipping malformed JSON line: {line.strip()}")
                        continue

def main(args):
    temp_corpus_path = "temp_corpus_for_tokenizer.txt"
    try:
        # --- 1. 从JSON数据文件创建临时语料库 ---
        create_corpus_from_json(args.json_files, temp_corpus_path)

        # --- 2. 加载 Whisper 的原始 Processor ---
        print(f"Loading original processor from {args.base_model}...")
        original_processor = WhisperProcessor.from_pretrained(args.base_model)
        original_tokenizer = original_processor.tokenizer

        # --- 3. 训练一个新的藏语 Tokenizer ---
        print(f"Training a new tokenizer from the generated corpus...")
        tibetan_tokenizer = Tokenizer(models.BPE())
        tibetan_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(vocab_size=args.new_vocab_size, special_tokens=["<|endoftext|>", "<unk>"])
        
        tibetan_tokenizer.train(files=[temp_corpus_path], trainer=trainer)
        
        # --- 4. 提取新的藏语 tokens ---
        new_tokens = []
        for i in range(tibetan_tokenizer.get_vocab_size()):
            token = tibetan_tokenizer.id_to_token(i)
            if token not in original_tokenizer.get_vocab():
                new_tokens.append(token)

        print(f"Found {len(new_tokens)} new unique tokens to add.")

        if not new_tokens:
            print("No new tokens to add. The existing tokenizer already covers the vocabulary in the provided files.")
            return

        # --- 5. 将新 tokens 添加到原始 tokenizer 中 ---
        original_tokenizer.add_tokens(new_tokens)
        print(f"Tokenizer vocabulary size after adding new tokens: {len(original_tokenizer)}")

        # --- 6. 加载原始模型并调整 Embedding 层大小 ---
        print(f"Loading original model from {args.base_model} to resize embeddings...")
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
        model.resize_token_embeddings(len(original_tokenizer))
        
        print(f"Model embedding layer resized to: {model.get_input_embeddings().weight.shape[0]}")

        # --- 7. 保存新的 Processor 和调整后的模型 ---
        os.makedirs(args.output_path, exist_ok=True)
        print(f"Saving extended processor and resized model to {args.output_path}...")
        original_processor.save_pretrained(args.output_path)
        model.save_pretrained(args.output_path)
        
        print("\nProcess complete!")
        print(f"Next steps: Use '{args.output_path}' as the --base_model for your fine-tuning command.")

    finally:
        # --- 8. 清理临时文件 ---
        if os.path.exists(temp_corpus_path):
            print(f"Cleaning up temporary corpus file: {temp_corpus_path}")
            os.remove(temp_corpus_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand Whisper tokenizer and model for a new language using JSON data files.")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base Whisper model (e.g., ./whisper-large-v3).")
    parser.add_argument("--json_files", type=str, nargs='+', required=True, help="List of JSON data files to build corpus from (e.g., dataset/train_filtered.json dataset/test_filtered.json).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the resized model and extended processor.")
    parser.add_argument("--new_vocab_size", type=int, default=1000, help="Vocabulary size for the new language tokenizer.")
    
    args = parser.parse_args()
    main(args)
