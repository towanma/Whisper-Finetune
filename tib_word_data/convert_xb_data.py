
import os
import json
import wave
import argparse
from tqdm import tqdm

def create_transcription_json(audio_folder_path, transcription_txt_path, output_json_path):
    """
    根据音频文件和转录文本文件，构建语音识别的转录JSON Lines文件。
    此版本新增了音频时长的计算和tqdm进度条。

    Args:
        audio_folder_path (str): 存放 .wav 音频文件的根文件夹路径。
        transcription_txt_path (str): 包含转录文本的 .txt 文件路径。
        output_json_path (str): 输出的 .json 文件路径。
    """
    # --- 步骤 1: 读取转录文本文件并存入字典 ---
    print(f"正在读取转录文件: {transcription_txt_path}...")
    transcriptions = {}
    try:
        with open(transcription_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    key, sentence = parts
                    transcriptions[key] = sentence
                else:
                    print(f"警告：跳过格式不正确的行 -> {line}")

    except FileNotFoundError:
        print(f"错误：找不到转录文件 '{transcription_txt_path}'。请检查路径是否正确。")
        return
    
    print(f"成功读取 {len(transcriptions)} 条转录数据。")

    # --- 步骤 2: 遍历音频文件，匹配、处理并写入JSON ---
    print(f"正在遍历音频文件夹: {audio_folder_path}... (这可能需要一些时间)")
    
    match_count = 0
    wav_files_to_process = []
    for root, _, files in os.walk(audio_folder_path):
        for filename in files:
            if filename.lower().endswith('.wav'):
                wav_files_to_process.append(os.path.join(root, filename))

    print(f"找到 {len(wav_files_to_process)} 个 .wav 文件，开始处理...")

    try:
        with open(output_json_path, 'w', encoding='utf-8') as out_f:
            for full_audio_path in tqdm(wav_files_to_process, desc="处理音频文件"):
                filename = os.path.basename(full_audio_path)
                base_name = os.path.splitext(filename)[0]
                
                if '-' in base_name:
                    key = base_name.split('-')[-1].strip()
                    
                    if key in transcriptions:
                        match_count += 1
                        
                        original_sentence = transcriptions[key]
                        modified_sentence = original_sentence.replace(' ', '་')
                        
                        duration = 0.0
                        try:
                            with wave.open(full_audio_path, 'rb') as wav_file:
                                frames = wav_file.getnframes()
                                rate = wav_file.getframerate()
                                if rate > 0:
                                    duration = frames / float(rate)
                        except Exception as e:
                            print(f"警告：无法读取音频文件 '{filename}' 的时长。错误: {e}")
                        
                        duration = round(duration, 2)
                        
                        data = {
                            "audio": {"path": os.path.abspath(full_audio_path)},
                            "sentence": modified_sentence,
                            "duration": duration 
                        }
                        
                        json_line = json.dumps(data, ensure_ascii=False)
                        out_f.write(json_line + '\n')
    
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return

    print("=" * 30)
    print("处理完成！")
    print(f"总共找到 {match_count} 个匹配的音频文件并已生成JSON。")
    print(f"结果已保存至: {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="将您的藏语语音数据转换成Whisper训练所需的JSON格式。" )
    parser.add_argument("--audio_path", type=str, required=True, help="存放 .wav 音频文件的根文件夹路径。" )
    parser.add_argument("--transcript_path", type=str, required=True, help="包含转录文本的 .txt 文件路径。" )
    parser.add_argument("--output_path", type=str, required=True, help="输出的 .json 文件路径。" )
    args = parser.parse_args()

    create_transcription_json(args.audio_path, args.transcript_path, args.output_path)

if __name__ == "__main__":
    main()
