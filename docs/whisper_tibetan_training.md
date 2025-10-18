Whisper 藏语训练流程指南
========================

本文档整理了两套常用流程，帮助你快速复现藏语识别模型的训练：

1. 直接使用原始 Whisper 词表做全量微调（适合字符集已经覆盖的情况）。
2. 先扩展词表再训练（适合需要新增藏文 token、降低序列长度的情况）。

无论哪一种方案，建议在**同一台机器上使用 Python 3.8+、CUDA 驱动可见的环境**，并提前执行：

```
python -m pip install -r requirements.txt
```

准备数据
--------

训练脚本假设音频与文本已经整理成 JSON Lines，常见字段如下：

```
{
  "audio": {"path": "/abs/path/to/audio.wav"},
  "sentence": "藏文或其它语言文本",
  "duration": 6.64
}
```

如果包含时间戳段落（`sentences` 列表），过滤脚本同样支持。训练前推荐运行一次长度过滤：

```
python filter_long_data.py \
  --model_path ./whisper-large-v3 \
  --train_input ./tib_word_data/output_transcription_part1.jsonl \
  --train_output ./tib_word_data/output_train.jsonl \
  --test_input ./tib_word_data/output_transcription_part2.jsonl \
  --test_output ./tib_word_data/output_test.jsonl \
  --max_tokens 440 \
  --language bo \
  --task transcribe
```

> 说明：`--language bo` 对应 `<|bo|>`，保持与训练、推理参数一致即可。

方案 A：直接微调原词表
---------------------

1. **准备命令行**  
   以两张 GPU 为例，执行：

   ```
   torchrun --nproc_per_node=2 finetune_full.py \
     --base_model ./whisper-large-v3 \
     --train_data ./tib_word_data/output_train.jsonl \
     --test_data ./tib_word_data/output_test.jsonl \
     --output_dir output_full/ \
     --per_device_train_batch_size 2 \
     --per_device_eval_batch_size 2 \
     --learning_rate 1e-5 \
     --num_train_epochs 3 \
     --logging_steps 100 \
     --eval_steps 1000 \
     --save_steps 1000 \
     --num_workers 8 \
     --fp16 True \
     --language bo \
     --task transcribe
   ```

   训练日志中的 `loss`、`eval_loss`、`grad_norm` 可用于监控收敛；完成后模型保存在 `output_full/whisper-large-v3/checkpoint-final`。

2. **推理验证**

   首次推理前，将基座模型中的 tokenizer 相关文件拷贝到 `checkpoint-final`，然后运行：

   ```
   python infer.py \
     --audio_path /path/to/test.wav \
     --model_path output_full/whisper-large-v3/checkpoint-final \
     --use_gpu True \
     --language bo \
     --task transcribe \
     --num_beams 1 \
     --batch_size 1 \
     --local_files_only True
   ```

3. **正式评估（可选）**

   训练完成后，可使用 `evaluation.py` 对独立测试集计算 CER/WER：

   ```
   python evaluation.py \
     --model_path output_full/whisper-large-v3/checkpoint-final \
     --metric cer
   ```

方案 B：先扩展词表再训练
-------------------------

当原始词表无法很好覆盖藏文音节、导致 token 序列过长时，推荐先扩展 tokenizer，再进行训练。

1. **训练并扩展 tokenizer**

   ```
   python expand_tokenizer.py \
     --base_model ./whisper-large-v3 \
     --json_files ./tib_word_data/output_train.jsonl ./tib_word_data/output_test.jsonl \
     --output_dir ./whisper-large-v3-tokenizer-tibetan \
     --new_vocab_size 2000 \
     --min_frequency 2 \
     --add_syllable_spaces
   ```

   - 输出目录会包含扩展后的 processor 以及 `added_tokens.txt`（新增 token 列表），可手动检查是否符合预期。
   - `--new_vocab_size` 小于现有词表规模时表示“新增大约这么多 token”；如果希望直接设定总词表大小，可给一个更大的数值（例如 `60000`）。
   - 如需调整词表大小或过滤低频 token，修改 `--new_vocab_size`、`--min_frequency` 即可。

2. **根据新词表调整模型 embedding**

   ```
   python prepare_expanded_model.py \
     --base_model ./whisper-large-v3 \
     --tokenizer_dir ./whisper-large-v3-tokenizer-tibetan \
     --output_dir ./whisper-large-v3-tibetan
   ```

   新目录（如 `./whisper-large-v3-tibetan`）同时包含扩展后的模型与 processor，可以视作新的基座模型。

3. **重新过滤数据（建议）**

   词表变化会影响 token 长度，重新运行过滤脚本确保所有样本满足长度约束：

   ```
   python filter_long_data.py \
     --model_path ./whisper-large-v3-tibetan \
     --train_input ./tib_word_data/output_transcription_part1.jsonl \
     --train_output ./tib_word_data/output_train.jsonl \
     --test_input ./tib_word_data/output_transcription_part2.jsonl \
     --test_output ./tib_word_data/output_test.jsonl \
     --max_tokens 440 \
     --language bo \
     --task transcribe
   ```

4. **使用扩展模型训练**

   `````
   torchrun --nproc_per_node=2 finetune_full.py \
     --base_model ./whisper-large-v3-tibetan \
     --train_data ./tib_word_data/output_train.jsonl \
     --test_data ./tib_word_data/output_test.jsonl \
     --output_dir output_full/ \
     --per_device_train_batch_size 2 \
     --per_device_eval_batch_size 2 \
     --learning_rate 1e-5 \
     --num_train_epochs 3 \
     --logging_steps 100 \
     --eval_steps 1000 \
     --save_steps 1000 \
     --num_workers 8 \
     --fp16 True \
     --language bo \
     --task transcribe
   ```
   `````

   训练结束后，同样可以复制 tokenizer 文件、运行 `infer.py` 与 `evaluation.py` 进行验证。

常见问题
--------

- **仍然出现 “Labels sequence length 449 cannot exceed 448 tokens” 报错？**  
  说明过滤脚本未对最新的 tokenizer 与语言前缀生效。确认 `filter_long_data.py` 使用的 `--model_path`、`--language`、`--task` 与训练时完全一致，必要时将 `--max_tokens` 调低（例如 440）。

- **中文等其它语言识别变差？**  
  全量微调会覆盖掉原有语言能力，属于灾难性遗忘。若希望兼顾多语言，可在训练集里混入其它语言样本，或改用 LoRA 等参数高效微调方式。

- **新增 token 合理性如何判断？**  
  检查 `added_tokens.txt` 是否以藏文音节为主，若仍有大量碎片，可提高 `--min_frequency` 或调整语料预处理，使训练语料更偏向 syllable 单位。

附：推理快捷命令
----------------

```
python infer.py \
  --audio_path /path/to/audio.wav \
  --model_path output_full/whisper-large-v3/checkpoint-final \
  --use_gpu True \
  --language bo \
  --task transcribe \
  --num_beams 1 \
  --batch_size 1 \
  --local_files_only True
```

若需要批量推理，可编写简单的 Shell/Python 脚本循环调用上述命令，并将输出保存到文件中。保持语言/任务参数与训练一致，能有效提升识别质量。

附：比较词表分词效果
--------------------

扩展词表后，希望直观查看 token 数量是否下降、分割是否满足预期，可以使用比较脚本：

```
python tools/compare_tokenizers.py \
  --base_tokenizer ./whisper-large-v3 \
  --expanded_tokenizer ./whisper-large-v3-tokenizer-tibetan \
  --text "ཅན་གྱི་ན་པུ་ཏའོ་ཡུང་ཧྲུའུ་གའི་ནུའུ་གཡ་གས་ང"
```

输出会展示两套 tokenizer 的 token 数量与具体字符串（`token_id:token`），便于对比音节粒度的改善情况。如不需要特殊 token，可加 `--no_special_tokens`。

附：调试单条文本的分词
----------------------

若需要核对某段藏文在新的 tokenizer 下的编码、token 列表以及 decode 结果，可使用调试脚本：

```
python tools/debug_tokenizer.py \
  --tokenizer_path ./whisper-large-v3-tokenizer-tibetan \
  --text "ཅན་གྱི་ན་པུ་ཏའོ་ཡུང་ཧྲུའུ་གའི་ནུའུ་གཡ་གས་ང" \
  --add_special_tokens
```

命令会打印 `IDs`、`Tokens` 以及 `Decoded` 三部分，方便确认 token 序列与原始文本的一致性。若要排除特殊 token，将 `--add_special_tokens` 去掉即可。
