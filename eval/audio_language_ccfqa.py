# SPDX-License-Identifier: Apache-2.0
import json
import os
import time
import librosa
from dataclasses import asdict
from typing import Any, NamedTuple, Optional
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from vllm.assets.audio import AudioAsset

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.lora.request import LoRARequest
import argparse
import torch
import json

LANGUAGE_MAPPING = {
    'ara': 'Arabic', 'arz': 'Arabic', 'ben': 'Bengali', 'ces': 'Czech',
    'deu': 'German', 'eng': 'English', 'spa': 'Spanish', 'fas': 'Persian',
    'pes': 'Persian', 'fra': 'French', 'heb': 'Hebrew', 'hin': 'Hindi',
    'ind': 'Indonesian', 'ita': 'Italian', 'jpn': 'Japanese', 'khm': 'Khmer',
    'kor': 'Korean', 'lao': 'Lao', 'msa': 'Malay', 'zsm': 'Malay',
    'mya': 'Burmese', 'nld': 'Dutch', 'pol': 'Polish', 'por': 'Portuguese',
    'rus': 'Russian', 'tha': 'Thai', 'tgl': 'Tagalog', 'tur': 'Turkish',
    'urd': 'Urdu', 'vie': 'Vietnamese', 'zho': 'Chinese', 'cmn': 'Chinese',
    'yue': 'Traditional Chinese', 'ceb': 'Cebuan', 'oci': 'Occitan', 'mon': 'Mongolian',
    'khk': 'Mongolian',
}
class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: Optional[str] = None
    prompt_token_ids: Optional[dict[str, list[int]]] = None
    multi_modal_data: Optional[dict[str, Any]] = None
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None

from transformers import AutoTokenizer

audio_assets = [AudioAsset("winning_call"),AudioAsset("mary_had_lamb")]

def get_visible_gpu_count():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0
print(f"可见显卡数量: {get_visible_gpu_count()}")


def load_and_filter_data(file_path, src_langs, tgt_langs, task, id):
    lines = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)

            id_num = data.get("id", "")
            if id_num <= id:
                continue

            src_lang = data.get("src_lang", "")
            if src_lang == "":
                src_lang = data.get("lang", "")

            tgt_lang = data.get("tgt_lang", "")

            if src_lang == "" and tgt_lang == "":
                prompt = data.get("prompt","")
                src_lang = prompt[2:5]
                tgt_lang = prompt[9:12]

            # 如果是 sqa 任务，只保留 src_lang == tgt_lang
            if task == "sqa" and src_lang != tgt_lang:
                continue

            if tgt_lang != "":
                if src_lang in src_langs and tgt_lang in tgt_langs:
                    lines.append(data)
            else:
                if src_lang in src_langs:
                    lines.append(data)

    # 按照 src_langs 和 tgt_langs 的顺序排序
    def sort_key(item):
        src_lang = item.get("src_lang", "")
        if src_lang == "":
            src_lang = item.get("lang", "")
        tgt_lang = item.get("tgt_lang", "")

        src_priority = src_langs.index(src_lang) if src_lang in src_langs else len(src_langs)
        tgt_priority = tgt_langs.index(tgt_lang) if tgt_lang in tgt_langs else len(tgt_langs)

        return (src_priority, tgt_priority)

    lines.sort(key=sort_key)
    return lines




# Qwen2.5-Omni
def run_qwen2_5_omni(question: str, audio_count: int):
    """准备Qwen2.5-Omni模型的请求数据"""
    engine_args = EngineArgs(model="../models/Qwen2.5-Omni-7B", max_model_len=4096, max_num_seqs=5, limit_mm_per_prompt={"audio": audio_count})
    audio_in_prompt = "".join(["<|audio_bos|><|AUDIO|><|audio_eos|>\n" for idx in range(audio_count)])

    default_system = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
        "Group, capable of perceiving auditory and visual inputs, as well as "
        "generating text and speech.")

    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{audio_in_prompt}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return ModelRequestData(engine_args, prompt)


# Qwen2-Audio
def run_qwen2_audio(question: str, audio_count: int) -> ModelRequestData:
    model_name = "../models/Qwen2-Audio-7B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": audio_count},
    )

    audio_in_prompt = "".join(
        [
            f"Audio {idx + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
            for idx in range(audio_count)
        ]
    )

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{audio_in_prompt}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
    )



question_per_audio_count = {
    0: "What is 1+1?",
    1: "What is recited in the audio?",
    2: "What sport and what nursery rhyme are referenced?",
}

model_example_map = {
    "qwen2_audio": run_qwen2_audio,
}

def prepare_batch(batch_data, file_dir, mode,task,model_type):

    """准备批量处理的数据"""
    templates, src_texts, sources, audios_out = [], [], [], []
    
    for data in batch_data:
        audio_path = data.get("src_audio","")
        if audio_path == "":
            audio_path = data.get("audio","")
        audio_path = os.path.join(file_dir, audio_path) if mode == "local" else None
        multi_modal_data = librosa.load(audio_path, sr=16000) if audio_path else None
        
        src_lang = data.get("src_lang","")
        tgt_lang = data.get("tgt_lang","")
        src_text = data.get(f"{src_lang}_q")
        
        if model_type == "qwen2_5_omni" or model_type == "qwen2_5_omni_3b" or model_type == "qwen2_audio" or model_type == "qwen3_omni":
            if task == "xsqa" or task=="sqa":
                prompt_template = f'''<|im_start|>system\nYou are a speech question answering assistant. Do not transcribe unless explicitly asked. Provide only the specific year, date, number, location, or name being asked for, return only an entity.<|im_end|>\n<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>\nAnswer this {LANGUAGE_MAPPING[src_lang]} speech question in {LANGUAGE_MAPPING[tgt_lang]}. Answer this Question.<|im_end|>\n<|im_start|>assistant\n'''


        templates.append(prompt_template)
        src_texts.append(src_text)
        sources.append(data['source'])
        audios_out.append(multi_modal_data)

    return templates, src_texts, sources, audios_out

def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('-m', '--model-type', default="qwen2_5_omni", 
                        choices=model_example_map.keys())
    parser.add_argument('--task', default="asr")
    parser.add_argument('--input-file', default="input.jsonl")
    parser.add_argument('--output-dir', default="./")
    parser.add_argument('--num-audios', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--seed', type=int, default=443)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument(
        "--num-prompts", type=int, default=1, help="Number of prompts to run."
    )
    return parser.parse_args()

model_example_map = {
    "qwen2_audio": run_qwen2_audio,
}

def main(args):

    #MLLM load
    req_data = model_example_map[args.model_type](
        question=question_per_audio_count[args.num_audios],
        audio_count=args.num_audios
    )
    
    engine_args = asdict(req_data.engine_args)
    engine_args.update({
        "seed": args.seed,
        "trust_remote_code": True,
        "tensor_parallel_size": get_visible_gpu_count(),
        "pipeline_parallel_size": 1,
        "dtype": "float16",
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 1} 
    })
    print(engine_args)
    llm = LLM(**engine_args)
    sampling_params = SamplingParams(temperature=0, max_tokens=256, best_of=1, top_k=1,stop_token_ids=req_data.stop_token_ids)
    

    #Data load
    file_path = args.input_file
    file_name = Path(file_path).name
    src_langs = ['ara', 'ben', 'ces', 'deu', 'eng', 'fas', 'fra', 'heb', 'hin', 'ind', 'ita', 'jpn', 'khm', 'kor', 'lao', 'msa', 'mya', 'nld', 'pol', 'por', 'rus', 'spa', 'tha', 'tgl', 'tur', 'urd', 'vie', 'yue','cmn']

    tgt_langs = ['ara', 'ben', 'ces', 'deu', 'eng', 'fas', 'fra', 'heb', 'hin', 'ind', 'ita', 'jpn', 'khm', 'kor', 'lao', 'msa', 'mya', 'nld', 'pol', 'por', 'rus', 'spa', 'tha', 'tgl', 'tur', 'urd', 'vie', 'yue','cmn']
    task = args.task

    id = 0


    data = load_and_filter_data(file_path, src_langs, tgt_langs,task,id = 0)




    mode = "local"
    
    
    model_type = args.model_type
    
    output_file = args.output_dir+f"{model_type}_{task}_{file_name}"
    batch_size = args.batch

    print(model_type)

    
    global_idx = 0
    # 记录开始时间
    start_time = time.time()
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            templates, src_texts, sources, audios_out = prepare_batch(batch, os.path.dirname(file_path),mode,task,model_type)
            
            inputs = [{"prompt": t, "multi_modal_data": {"audio": a}} for t, a in zip(templates, audios_out)]

            outputs = llm.generate(inputs, sampling_params=sampling_params,lora_request=req_data.lora_requests * args.num_prompts if req_data.lora_requests else None)
            
            for j, output in enumerate(outputs):
                result = output.outputs[0].text.split("</think>")[-1].replace("\n", "").split(": ")[-1]

                if task == "sqa" or task == "xsqa":
                    entry = {
                        **batch[j],
                        "lang_a": result
                    }
                f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                global_idx +=1
                print(f"{global_idx}: {result}")
            print("-" * 50)
    # 计算并打印总耗时
    end_time = time.time()
    total_time = end_time - start_time
    print(f"总耗时: {total_time:.2f}秒")

if __name__ == "__main__":
    main(parse_args())