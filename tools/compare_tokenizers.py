import argparse
from typing import List, Sequence, Tuple

from transformers import WhisperProcessor


def encode_text(tokenizer, text: str, add_special_tokens: bool, special_tokens: Sequence[str]) -> Tuple[List[int], List[str]]:
    if add_special_tokens:
        ids = tokenizer.encode(text, add_special_tokens=True)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        return ids, tokens

    ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    special_set = set(special_tokens)
    filtered_ids: List[int] = []
    filtered_tokens: List[str] = []
    for token_id, token in zip(ids, tokens):
        if token in special_set:
            continue
        filtered_ids.append(token_id)
        filtered_tokens.append(token)
    return filtered_ids, filtered_tokens


def format_tokens(ids: Sequence[int], tokens: Sequence[str]) -> List[str]:
    return [f"{token_id}:{token}" for token_id, token in zip(ids, tokens)]


def compare_tokenizers(base_path: str, expanded_path: str, text: str, add_special_tokens: bool) -> None:
    print(f"Loading base tokenizer from {base_path}")
    base_processor = WhisperProcessor.from_pretrained(base_path)
    print(f"Loading expanded tokenizer from {expanded_path}")
    expanded_processor = WhisperProcessor.from_pretrained(expanded_path)

    base_ids, base_tokens = encode_text(
        base_processor.tokenizer,
        text,
        add_special_tokens=add_special_tokens,
        special_tokens=base_processor.tokenizer.all_special_tokens,
    )
    expanded_ids, expanded_tokens = encode_text(
        expanded_processor.tokenizer,
        text,
        add_special_tokens=add_special_tokens,
        special_tokens=expanded_processor.tokenizer.all_special_tokens,
    )

    print("\n=== Input Text ===")
    print(text)
    print("\n=== Base Tokenizer ===")
    print(f"Token count: {len(base_ids)}")
    print(format_tokens(base_ids, base_tokens))

    print("\n=== Expanded Tokenizer ===")
    print(f"Token count: {len(expanded_ids)}")
    print(format_tokens(expanded_ids, expanded_tokens))


def main():
    parser = argparse.ArgumentParser(description="Compare tokenization between base and expanded Whisper tokenizers.")
    parser.add_argument("--base_tokenizer", required=True, help="Path to the base tokenizer/processor directory.")
    parser.add_argument("--expanded_tokenizer", required=True, help="Path to the expanded tokenizer/processor directory.")
    parser.add_argument("--text", required=True, help="Input text to tokenize.")
    parser.add_argument(
        "--no_special_tokens",
        action="store_true",
        help="Exclude special tokens from the comparison output.",
    )
    args = parser.parse_args()

    compare_tokenizers(
        base_path=args.base_tokenizer,
        expanded_path=args.expanded_tokenizer,
        text=args.text,
        add_special_tokens=not args.no_special_tokens,
    )


if __name__ == "__main__":
    main()
