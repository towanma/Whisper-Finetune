import argparse
from transformers import WhisperProcessor

def main():
    """
    An interactive script to visualize the output of the Whisper tokenizer.
    """
    parser = argparse.ArgumentParser(description="Visualize Whisper tokenizer output for a given text.")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the Whisper model or tokenizer repository (e.g., ./whisper-large-v3).")
    args = parser.parse_args()

    try:
        print(f"Loading tokenizer from: {args.model_path}...")
        processor = WhisperProcessor.from_pretrained(args.model_path)
        tokenizer = processor.tokenizer
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print("\nEnter text to see the tokenization. Type 'exit' or press Ctrl+C to quit.")
    print("-