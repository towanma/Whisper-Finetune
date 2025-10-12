# Repository Guidelines

## Project Structure & Module Organization
- Core training scripts live at the root (`finetune.py`, `finetune_full.py`, `merge_lora.py`, `evaluation.py`, `infer*.py`). Treat them as the entry points for model tuning, merging, scoring, and inference flows.
- `configs/` holds YAML templates for LoRA and Whisper variants; copy and adjust these instead of editing defaults in-place.
- Datasets, sample audio, and generated artifacts belong in `dataset/`, `output_full/`, and any run-specific subfolders under `output/`.
- Assets for deployed clients sit in `AndroidDemo/`, `WhisperDesktop/`, `static/`, and `templates/`. Keep these synchronized with server API changes.
- Utility helpers and reproducibility tools are in `tools/` and `utils/`; prefer extending these modules before adding new top-level scripts.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` installs training and inference dependencies; run inside a fresh Python 3.8+ environment.
- `CUDA_VISIBLE_DEVICES=0 python finetune.py --base_model=openai/whisper-small --output_dir=output/whisper-small` launches a single-GPU LoRA fine-tune using configs inferred from CLI flags.
- `python merge_lora.py --lora_model=output/whisper-small/checkpoint-best/ --output_dir=models/` fuses LoRA adapters with the base model for deployment.
- `python evaluation.py --model_path=models/whisper-small-finetune --metric=cer` reports CER/WER with references stored in `dataset/metadata.csv`.
- `python infer.py --audio_path=dataset/test.wav --model_path=models/whisper-small-finetune` performs quick regression checks; `bash run.sh` drives the end-to-end demo server.

## Coding Style & Naming Conventions
- Follow standard Python style: 4-space indents, snake_case functions, and UpperCamelCase classes. Mirror existing argument names when extending CLI interfaces.
- Keep configuration keys lowercase with hyphen-separated YAML names (e.g., `learning-rate`), matching `configs/*.yaml`.
- Use docstrings for new public functions in `utils/` or top-level scripts, and prefer type hints for complex return values.

## Testing Guidelines
- There is no dedicated unit test harness; rely on `evaluation.py` for quantitative checks and `infer.py` or `infer_gui.py` for qualitative validation.
- Store comparison transcripts under `metrics/` and update README tables only after rerunning evaluations on shared datasets.
- For new datasets, add manifests in `dataset/` and share exact preprocessing steps via `tools/`; align file names with the pattern `<split>_<language>.tsv`.

## Commit & Pull Request Guidelines
- Recent history favors short, imperative subjects (e.g., “add filter”); follow that format and keep commits scoped to one feature or fix.
- Reference issues as `#123` when relevant, and note the model size, dataset, and hardware in the commit body when changes affect training.
- Pull requests should include: summary of intent, key commands run (fine-tune, merge, evaluation), metrics before/after, and screenshots for GUI or deployment-facing updates.
