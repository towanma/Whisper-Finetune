import argparse

from transformers import WhisperProcessor


def main():
    parser = argparse.ArgumentParser(description="Inspect tokenization and decoding for a given text.")
    parser.add_argument("--tokenizer_path", required=True, help="Path to tokenizer/processor directory.")
    parser.add_argument("--text", required=True, help="Input text to tokenize.")
    parser.add_argument(
        "--add_special_tokens",
        action="store_true",
        help="Include special tokens during tokenization.",
    )
    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained(args.tokenizer_path)
    tokenizer = processor.tokenizer

    ids = tokenizer.encode(args.text, add_special_tokens=args.add_special_tokens)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    decoded = tokenizer.decode(ids)

    print("IDs:", ids)
    print("Tokens:", tokens)
    print("Decoded:", decoded)


if __name__ == "__main__":
    main()
