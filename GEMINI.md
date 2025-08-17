
# GEMINI.md

## Project Overview

This project is designed for fine-tuning the OpenAI Whisper model for speech recognition tasks. It leverages Python and the Hugging Face ecosystem, including the `transformers` and `peft` libraries, to enable efficient fine-tuning with techniques like LoRA and AdaLora. The project supports various Whisper model sizes, from `tiny` to `large-v3`. Key functionalities include preparing custom datasets, training the model, merging fine-tuned weights, evaluating performance with CER and WER metrics, and running inference through a CLI, GUI, or a web server. Additionally, it offers acceleration features like FlashAttention2, BetterTransformer, and CTranslate2, and provides client applications for Android and Windows.

## Building and Running

### 1. Environment Setup

First, install the necessary dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Prepare your training and testing data in the JSON format specified in the `README.md` file. The project includes a script `aishell.py` to help with generating data from the AIShell dataset.

### 3. Fine-tuning

To fine-tune a Whisper model, use the `finetune.py` script. You can specify the base model, output directory, and other hyperparameters as command-line arguments. For example, to fine-tune the `whisper-tiny` model:

```bash
python finetune.py --base_model openai/whisper-tiny --output_dir output/
```

The `run.sh` script provides examples of how to fine-tune different model sizes with various configurations.

### 4. Merging LoRA Models

After fine-tuning, merge the LoRA weights with the base model using `merge_lora.py`:

```bash
python merge_lora.py --lora_model output/whisper-tiny/checkpoint-best/ --output_dir models/
```

### 5. Evaluation

Evaluate the performance of the fine-tuned model using `evaluation.py`. You can choose between CER and WER as the evaluation metric:

```bash
python evaluation.py --model_path models/whisper-tiny-finetune --metric cer
```

### 6. Inference

The project provides several ways to run inference with the fine-tuned model:

- **Command-line Interface:** Use `infer.py` to transcribe an audio file from the command line.

  ```bash
  python infer.py --audio_path dataset/test.wav --model_path models/whisper-tiny-finetune
  ```

- **GUI Application:** The `infer_gui.py` script launches a graphical user interface for interactive transcription.

- **Web Server:** The `infer_server.py` script starts a web server that exposes a transcription API.

## Development Conventions

The project follows standard Python development practices. The code is organized into several scripts with clear responsibilities. The use of command-line arguments for configuration makes the scripts flexible and easy to use. The `README.md` file is comprehensive and provides detailed instructions for all aspects of the project. The project also includes support for both single and multi-GPU training.
