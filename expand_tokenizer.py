
import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

from tokenizers import Tokenizer, models, trainers
from transformers import WhisperProcessor
from tqdm import tqdm


TIBETAN_SYLLABLE_DELIMITERS = [
    "\u0F0B",  # ་ tsek
    "\u0F0C",  # shad tsek
    "\u0F0D",  # shad
    "\u0F0E",  # nyis shad
    "\u0F0F",  # rin chen spungs shad
    "\u0F11",  # rgya gram shad
]


def normalize_tibetan_text(text: str, add_syllable_spaces: bool) -> str:
    """Optionally inserts spaces after Tibetan syllable delimiters to help tokenizer learn syllable-level tokens."""
    if not add_syllable_spaces:
        return text
    for delimiter in TIBETAN_SYLLABLE_DELIMITERS:
        text = text.replace(delimiter, f"{delimiter} ")
    # Collapse multi-space sequences introduced by replacements
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def create_corpus_from_json(json_files: Iterable[str], corpus_path: str, add_syllable_spaces: bool) -> None:
    """
    Reads one or more JSON lines files and extracts the 'sentence' field
    to create a temporary text corpus file for tokenizer training.
    """
    print(f"Creating text corpus from {json_files} at {corpus_path}...")
    with open(corpus_path, "w", encoding="utf-8") as corpus_file:
        for json_file_path in json_files:
            print(f"Processing {json_file_path}...")
            with open(json_file_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"Extracting text from {os.path.basename(json_file_path)}"):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Skipping malformed JSON line: {line.strip()}")
                        continue
                    sentence = data.get("sentence")
                    if not sentence:
                        continue
                    normalized_sentence = normalize_tibetan_text(sentence, add_syllable_spaces=add_syllable_spaces)
                    if normalized_sentence:
                        corpus_file.write(normalized_sentence + "\n")


def main(args):
    temp_corpus_path = "temp_corpus_for_tokenizer.txt"
    try:
        create_corpus_from_json(args.json_files, temp_corpus_path, add_syllable_spaces=args.add_syllable_spaces)

        print(f"Loading original processor from {args.base_model}...")
        original_processor = WhisperProcessor.from_pretrained(args.base_model)
        original_tokenizer = original_processor.tokenizer

        print("Loading tokenizer.json to collect base merges/vocab...")
        tokenizer_json_path = getattr(original_tokenizer, "init_kwargs", {}).get("tokenizer_file")
        if tokenizer_json_path is None or not os.path.isfile(tokenizer_json_path):
            candidate_path = os.path.join(args.base_model, "tokenizer.json")
            if os.path.isfile(candidate_path):
                tokenizer_json_path = candidate_path
        if tokenizer_json_path is None or not os.path.isfile(tokenizer_json_path):
            raise FileNotFoundError(
                "Cannot locate tokenizer.json for the base model. "
                "Please ensure the base model directory contains tokenizer.json."
            )

        base_tokenizer_impl = Tokenizer.from_file(tokenizer_json_path)
        base_tokenizer_json = json.loads(base_tokenizer_impl.to_str())
        base_model_cfg = base_tokenizer_json["model"]
        bpe_vocab: Dict[str, int] = base_model_cfg["vocab"]
        base_merges: List[List[str]] = base_model_cfg["merges"]

        added_vocab = original_tokenizer.get_added_vocab()
        special_tokens = list(added_vocab.keys())
        special_token_set = set(special_tokens)

        max_bpe_id = max(bpe_vocab.values()) if bpe_vocab else -1
        max_added_id = max(added_vocab.values()) if added_vocab else -1
        next_available_id = max(max_bpe_id, max_added_id) + 1

        print("Training auxiliary tokenizer to mine new merges...")
        auxiliary_tokenizer = Tokenizer(models.BPE(**{k: v for k, v in {
            "unk_token": base_model_cfg.get("unk_token"),
            "continuing_subword_prefix": base_model_cfg.get("continuing_subword_prefix"),
            "end_of_word_suffix": base_model_cfg.get("end_of_word_suffix"),
            "fuse_unk": base_model_cfg.get("fuse_unk"),
            "byte_fallback": base_model_cfg.get("byte_fallback"),
            "dropout": base_model_cfg.get("dropout"),
            "ignore_merges": base_model_cfg.get("ignore_merges"),
        }.items() if v is not None}))
        auxiliary_tokenizer.normalizer = base_tokenizer_impl.normalizer
        auxiliary_tokenizer.pre_tokenizer = base_tokenizer_impl.pre_tokenizer
        auxiliary_tokenizer.decoder = base_tokenizer_impl.decoder
        base_vocab_size = len(bpe_vocab)
        target_vocab_size = (
            base_vocab_size + args.new_vocab_size
            if args.new_vocab_size <= base_vocab_size
            else args.new_vocab_size
        )
        trainer = trainers.BpeTrainer(
            vocab_size=target_vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=special_tokens,
        )
        auxiliary_tokenizer.train(files=[temp_corpus_path], trainer=trainer)
        auxiliary_json = json.loads(auxiliary_tokenizer.to_str())
        aux_vocab = auxiliary_json["model"]["vocab"]
        aux_merges: List[List[str]] = auxiliary_json["model"]["merges"]

        print("Integrating auxiliary results...")
        existing_tokens = set(bpe_vocab.keys()) | special_token_set
        new_tokens: List[Tuple[int, str]] = []
        for token, idx in sorted(aux_vocab.items(), key=lambda item: item[1]):
            if token in existing_tokens:
                continue
            bpe_vocab[token] = next_available_id
            new_tokens.append((next_available_id, token))
            existing_tokens.add(token)
            next_available_id += 1

        new_token_strings = {token for _, token in new_tokens}
        merge_set = {tuple(pair) for pair in base_merges}
        for merge in aux_merges:
            merge_tuple = tuple(merge)
            result_token = "".join(merge)
            if result_token not in new_token_strings:
                continue
            if merge_tuple not in merge_set:
                base_merges.append(merge)
                merge_set.add(merge_tuple)

        updated_tokenizer = Tokenizer.from_str(json.dumps(base_tokenizer_json))

        os.makedirs(args.output_dir, exist_ok=True)
        tmp_tokenizer_path = os.path.join(args.output_dir, "_tokenizer_tmp.json")
        updated_tokenizer.save(tmp_tokenizer_path)

        # ensure special tokens occupy their original ids in vocab to avoid holes
        for token, idx in added_vocab.items():
            if token not in bpe_vocab:
                bpe_vocab[token] = idx

        added_tokens_path = os.path.join(args.output_dir, "added_tokens.txt")
        with open(added_tokens_path, "w", encoding="utf-8") as f:
            for idx, token in new_tokens:
                f.write(f"{idx}\t{token}\n")
        print(f"Saved list of newly assigned token IDs to {added_tokens_path}")

        print("Saving feature extractor and tokenizer configs...")
        original_processor.save_pretrained(args.output_dir)

        final_tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
        os.replace(tmp_tokenizer_path, final_tokenizer_path)

        vocab_sorted = sorted(bpe_vocab.items(), key=lambda item: item[1])
        with open(os.path.join(args.output_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump({token: idx for token, idx in vocab_sorted}, f, ensure_ascii=False)

        merges_path = os.path.join(args.output_dir, "merges.txt")
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for merge in base_merges:
                f.write(" ".join(merge) + "\n")
        print(f"Updated tokenizer assets written to {args.output_dir}")

        print("\nTokenizer expansion complete.")
        print(f"Next steps: run the model-resizing script with --tokenizer_dir {args.output_dir} to update model embeddings.")
    finally:
        if os.path.exists(temp_corpus_path):
            print(f"Cleaning up temporary corpus file: {temp_corpus_path}")
            os.remove(temp_corpus_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a tokenizer on Tibetan text and extend a Whisper tokenizer without touching model weights."
    )
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base Whisper model (e.g., ./whisper-large-v3).")
    parser.add_argument(
        "--json_files",
        type=str,
        nargs="+",
        required=True,
        help="List of JSONL files containing a 'sentence' field (e.g., dataset/train_filtered.json dataset/test_filtered.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the extended tokenizer/processor will be saved.",
    )
    parser.add_argument(
        "--new_vocab_size",
        type=int,
        default=2000,
        help="If <= base vocab size, interpreted as number of extra tokens to mine; otherwise treated as absolute vocab size for auxiliary training.",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for a token candidate to be kept during auxiliary tokenizer training.",
    )
    parser.add_argument(
        "--add_syllable_spaces",
        action="store_true",
        help="Insert spaces after Tibetan syllable delimiters (་, །, …) when building the corpus to encourage syllable-level tokens.",
    )

    args = parser.parse_args()
    main(args)
