import json
import os
import time
import csv
from collections import defaultdict

import openai
from openai import OpenAI
import requests
import pandas as pd
import openpyxl
import torch

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LogitsProcessor
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.generation.logits_process import _calc_banned_ngram_tokens
from evaluate import load

normalizer = BasicTextNormalizer()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

src_langs = ['ara', 'ben', 'ces', 'deu', 'eng', 'fas', 'fra', 'heb', 'hin', 'ind', 'ita', 'jpn', 'khm', 'kor', 'lao', 'msa', 'mya', 'nld', 'pol', 'por', 'rus', 'spa', 'tha', 'tgl', 'tur', 'urd', 'vie', 'zho','yue','cmn']

tgt_langs = ['ara', 'ben', 'ces', 'deu', 'eng', 'fas', 'fra', 'heb', 'hin', 'ind', 'ita', 'jpn', 'khm', 'kor', 'lao', 'msa', 'mya', 'nld', 'pol', 'por', 'rus', 'spa', 'tha', 'tgl', 'tur', 'urd', 'vie', 'zho','yue','cmn']

model_vllm = "/data_a100/models/gemma-3-27b-it"
input_file = "./qwen2_5_omni_sqa_test_x.jsonl"

LANGUAGE_MAPPING = {
    'ara': 'Arabic', 'arz': 'Arabic', 'ben': 'Bengali', 'ces': 'Czech',
    'deu': 'German', 'eng': 'English', 'spa': 'Spanish', 'fas': 'Persian',
    'pes': 'Persian', 'fra': 'French', 'heb': 'Hebrew', 'hin': 'Hindi',
    'ind': 'Indonesian', 'ita': 'Italian', 'jpn': 'Japanese', 'khm': 'Khmer',
    'kor': 'Korean', 'lao': 'Lao', 'msa': 'Malay', 'zsm': 'Malay',
    'mya': 'Burmese', 'nld': 'Dutch', 'pol': 'Polish', 'por': 'Portuguese',
    'rus': 'Russian', 'tha': 'Thai', 'tgl': 'Tagalog', 'tur': 'Turkish',
    'urd': 'Urdu', 'vie': 'Vietnamese', 'zho': 'Chinese', 'cmn': 'Chinese',
    'yue': 'Yue Chinese', 'ceb': 'Cebuan', 'oci': 'Occitan', 'mon': 'Mongolian',
    'khk': 'Mongolian',
}

def process_jsonl(input_file, output_file, llm, model_vllm):
    batch_size = 64000
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        lines = []
        for line in infile:
            data = json.loads(line)
            lines.append(line)
        count = 1

        sampling_params = SamplingParams(max_tokens=64, best_of=1, top_k=1, top_p=1, temperature=0)
        results = []

        for i in range(0, len(lines), batch_size):
            batch_lines = lines[i:i + batch_size]
            batch_data = [json.loads(line) for line in batch_lines]
            
            templates = []
            src_texts = []
            sources = []
            for data in batch_data:
                gt = data.get('gt', '')
                src_lang = data.get('src_lang', '')
                tgt_lang = data.get('tgt_lang', '')
                source = data.get('source', '')
                question = data.get("src_q")
                answer = data.get("tgt_a")
                response = data.get("lang_a")

                if "gemma-3" in model_vllm:
                    qa_prompt = (f"""<bos><start_of_turn>user
                        Based on the {LANGUAGE_MAPPING[src_lang]} question and {LANGUAGE_MAPPING[tgt_lang]} reference and response, 
                        determine if the response correctly answers the question in {LANGUAGE_MAPPING[tgt_lang]}. 
                        If correct, return "yes"; otherwise, return "no", without any additional explanation.

                        \n
                        Question: {question}\n
                        {LANGUAGE_MAPPING[tgt_lang]} reference: {answer}\
                        {LANGUAGE_MAPPING[tgt_lang]} response: {response}
                         <end_of_turn><start_of_turn>model""")
                

                
                templates.append(qa_prompt)
                src_texts.append(question)
                sources.append(source)
            
            outputs = llm.generate(templates, sampling_params)
            
            for j, output in enumerate(outputs):
                output_text = output.outputs[0].text.split("</think>")[-1].replace("\n","")
                original_data = batch_data[j]
                
                # 保留原始所有字段，新增acc字段
                new_data = {**original_data, "acc": output_text}
                
                results.append(new_data)
                count += 1
        
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
gpu_count = torch.cuda.device_count()
print(gpu_count)

llm = LLM(model_vllm, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=gpu_count,gpu_memory_utilization=0.7,max_model_len=1024)



base_name = os.path.splitext(os.path.basename(input_file))[0]  
output_file = f"{base_name}_eval_{model_vllm.split('/')[-1]}.jsonl"  

print(f"Input file: {input_file}")
print(f"Output file: {output_file}")
process_jsonl(input_file, output_file,llm,model_vllm)









# -----------------------------------------------------------------------------------------------------------



import json
from collections import defaultdict
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter
import os

def calculate_f1_score(target_text, prediction_text):
    """
    计算两个字符串的字符级F1分数。
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    Precision = TP / (TP + FP)  (匹配字符数 / 预测文本总字符数)
    Recall = TP / (TP + FN)     (匹配字符数 / 目标文本总字符数)
    这里我们将匹配字符数定义为两个字符串的共同字符数量 (交集)。
    """
    if not target_text and not prediction_text:
        return 100.0  # 两个都为空，F1为100%

    target_chars = set(list(target_text))
    prediction_chars = set(list(prediction_text))

    common_chars = target_chars.intersection(prediction_chars)
    
    tp = len(common_chars)
    
    # If one of the texts is empty but the other is not, and there are no common chars, F1 is 0.
    # This handles cases like target="a", prediction="" or target="", prediction="b"
    if (not target_text and prediction_text) or (target_text and not prediction_text):
        return 0.0

    precision = tp / len(prediction_chars) if len(prediction_chars) > 0 else 0
    recall = tp / len(target_chars) if len(target_chars) > 0 else 0

    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1 * 100, 1) # 返回百分比形式，保留一位小数


def calculate_lang_scores(jsonl_path):
    """
    计算每个语言对的acc得分（yes数量占比）和字符级F1得分。
    同时计算每个源语言到所有目标语言（包括同语种）的平均分，以及所有语言对的整体平均分。
    
    参数:
        jsonl_path: JSONL文件路径
    
    返回:
        dict: {(src_lang, tgt_lang): {"yes": count, "no": count, "other": count, "score": float, "f1_total": float, "f1_count": int, "f1_avg": float}} 的统计字典
        int: 总记录数
        int: 包含acc、src_lang和tgt_lang字段的有效记录数
        int: 包含tgt_a和lang_a字段的有效F1计算记录数
        dict: {src_lang: {"f1_avg_all": float, "acc_avg_all": float}} 每个源语言到所有目标语言的平均分 (包括同语种)
        dict: {"overall_f1_avg": float, "overall_acc_avg": float} 所有语言对的整体平均分
        dict: {"sqa_f1_avg": float, "sqa_acc_avg": float} SQA (同语种) 平均分
    """
    lang_pair_stats = defaultdict(lambda: {"yes": 0, "no": 0, "other": 0, "f1_total": 0.0, "f1_count": 0})
    total_count = 0
    valid_acc_fields_count = 0 # Records with acc, src_lang, tgt_lang
    valid_f1_fields_count = 0 # Records with tgt_a, lang_a

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            try:
                data = json.loads(line.strip())
                src_lang = data.get('src_lang')
                tgt_lang = data.get('tgt_lang')
                acc_value = str(data.get('acc', '')).lower().strip() # Use .get() for safer access

                # Check for fields necessary for ACC calculation
                if src_lang and tgt_lang and 'acc' in data:
                    valid_acc_fields_count += 1
                    # Classify ACC
                    if "yes" in acc_value:
                        lang_pair_stats[(src_lang, tgt_lang)]["yes"] += 1
                    elif "no" in acc_value:
                        lang_pair_stats[(src_lang, tgt_lang)]["no"] += 1
                    else:
                        lang_pair_stats[(src_lang, tgt_lang)]["other"] += 1

                # Calculate F1 for ALL records that have 'tgt_a' and 'lang_a'
                if src_lang and tgt_lang and 'tgt_a' in data and 'lang_a' in data:
                    valid_f1_fields_count += 1
                    tgt_a = str(data['tgt_a'])
                    lang_a = str(data['lang_a'])
                    f1_score = calculate_f1_score(tgt_a, lang_a)
                    lang_pair_stats[(src_lang, tgt_lang)]["f1_total"] += f1_score
                    lang_pair_stats[(src_lang, tgt_lang)]["f1_count"] += 1
                
            except json.JSONDecodeError:
                print(f"警告：跳过无法解析的行（行号: {total_count}）")
    
    # Calculate ACC score (yes ratio) and average F1 score for each language pair
    for lang_pair in lang_pair_stats:
        total_acc_for_pair = lang_pair_stats[lang_pair]["yes"] + lang_pair_stats[lang_pair]["no"] + lang_pair_stats[lang_pair]["other"]
        lang_pair_stats[lang_pair]["score"] = round(lang_pair_stats[lang_pair]["yes"] / max(total_acc_for_pair, 1)*100, 1)  # Percentage form

        if lang_pair_stats[lang_pair]["f1_count"] > 0:
            lang_pair_stats[lang_pair]["f1_avg"] = round(lang_pair_stats[lang_pair]["f1_total"] / lang_pair_stats[lang_pair]["f1_count"], 1)
        else:
            lang_pair_stats[lang_pair]["f1_avg"] = 0.0

    # Define language order for consistent processing
    lang_order = ['cmn','eng', 'fra', 'jpn', 'kor', 'rus', 'spa', 'yue']

    # Calculate average score for each source language to all target languages (including same-language)
    src_all_lang_averages = defaultdict(lambda: {"f1_total_all": 0.0, "f1_count_all": 0,
                                                  "acc_total_all": 0.0, "acc_count_all": 0,
                                                  "f1_avg_all": 0.0, "acc_avg_all": 0.0})
    
    # Calculate overall average score for all language pairs
    overall_f1_total = 0.0
    overall_f1_count = 0
    overall_acc_total = 0.0
    overall_acc_count = 0

    # Calculate SQA (Same-Question Answering) averages
    sqa_f1_total = 0.0
    sqa_acc_total = 0.0
    sqa_count = 0

    for src_lang in lang_order:
        for tgt_lang in lang_order:
            key = (src_lang, tgt_lang)
            if key in lang_pair_stats:
                # Accumulate for overall average
                overall_f1_total += lang_pair_stats[key]["f1_avg"]
                overall_f1_count += 1
                overall_acc_total += lang_pair_stats[key]["score"]
                overall_acc_count += 1

                # Accumulate for source-based average (all target languages)
                src_all_lang_averages[src_lang]["f1_total_all"] += lang_pair_stats[key]["f1_avg"]
                src_all_lang_averages[src_lang]["f1_count_all"] += 1
                src_all_lang_averages[src_lang]["acc_total_all"] += lang_pair_stats[key]["score"]
                src_all_lang_averages[src_lang]["acc_count_all"] += 1
                
                # Accumulate for SQA average (same language)
                if src_lang == tgt_lang:
                    sqa_f1_total += lang_pair_stats[key]["f1_avg"]
                    sqa_acc_total += lang_pair_stats[key]["score"]
                    sqa_count += 1
    
    # Finalize source-based averages (all target languages)
    for src_lang in src_all_lang_averages:
        if src_all_lang_averages[src_lang]["f1_count_all"] > 0:
            src_all_lang_averages[src_lang]["f1_avg_all"] = round(
                src_all_lang_averages[src_lang]["f1_total_all"] / src_all_lang_averages[src_lang]["f1_count_all"], 1)
        if src_all_lang_averages[src_lang]["acc_count_all"] > 0:
            src_all_lang_averages[src_lang]["acc_avg_all"] = round(
                src_all_lang_averages[src_lang]["acc_total_all"] / src_all_lang_averages[src_lang]["acc_count_all"], 1)

    # Finalize overall averages (all pairs)
    overall_averages = {
        "overall_f1_avg": round(overall_f1_total / max(overall_f1_count, 1), 1),
        "overall_acc_avg": round(overall_acc_total / max(overall_acc_count, 1), 1)
    }

    # Finalize SQA averages (same-language pairs)
    sqa_averages = {
        "sqa_f1_avg": round(sqa_f1_total / max(sqa_count, 1), 1),
        "sqa_acc_avg": round(sqa_acc_total / max(sqa_count, 1), 1)
    }

    return dict(lang_pair_stats), total_count, valid_acc_fields_count, valid_f1_fields_count, \
           dict(src_all_lang_averages), overall_averages, sqa_averages


def save_to_excel(lang_pair_stats, output_file, src_all_lang_averages, overall_averages, sqa_averages):
    """
    将统计结果保存为Excel文件（按语言对排列），合并F1和Acc得分。
    增加一列行平均分，以及SQA总结行和整体平均行。
    格式为 "F1 / Acc"。
    """
    # Define language order
    lang_order = ['cmn','eng', 'fra', 'jpn', 'kor', 'rus', 'spa', 'yue']
    
    # Prepare combined data for the main grid and summary rows
    combined_data = []
    
    # Create header row, including the "行平均" column
    header = ["Src/Tgt"] + lang_order + ["XSQA (F1/Acc)"]
    combined_data.append(header)
    
    # Populate main language pair data grid
    for src_lang in lang_order:
        row = [src_lang]
        for tgt_lang in lang_order:
            key = (src_lang, tgt_lang)
            if key in lang_pair_stats:
                f1_avg = lang_pair_stats[key]["f1_avg"]
                acc_score = lang_pair_stats[key]["score"]
                row.append(f"{f1_avg:.1f} / {acc_score:.1f}")
            else:
                row.append("- / -")  # If data for this language pair does not exist
        
        # Add the row-wise average (src_all_lang_averages) to the last column
        if src_lang in src_all_lang_averages:
            f1_avg_all = src_all_lang_averages[src_lang]["f1_avg_all"]
            acc_avg_all = src_all_lang_averages[src_lang]["acc_avg_all"]
            row.append(f"{f1_avg_all:.1f} / {acc_avg_all:.1f}")
        else:
            row.append("- / -") # If no data for this source language
        
        combined_data.append(row)
    
    # --- Add SQA (Same-Question Answering) Row ---
    sqa_row = ["SQA (F1/Acc)"]
    sqa_f1_total_for_row = 0.0
    sqa_acc_total_for_row = 0.0
    sqa_count_for_row = 0

    for lang in lang_order:
        key = (lang, lang) # SQA means src_lang == tgt_lang
        if key in lang_pair_stats:
            f1_avg = lang_pair_stats[key]["f1_avg"]
            acc_score = lang_pair_stats[key]["score"]
            sqa_row.append(f"{f1_avg:.1f} / {acc_score:.1f}")
            sqa_f1_total_for_row += f1_avg
            sqa_acc_total_for_row += acc_score
            sqa_count_for_row += 1
        else:
            sqa_row.append("- / -")
    
    # Add SQA row's own average to its last cell (average of all SQA pairs)
    sqa_row_f1_avg = round(sqa_f1_total_for_row / max(sqa_count_for_row, 1), 1)
    sqa_row_acc_avg = round(sqa_acc_total_for_row / max(sqa_count_for_row, 1), 1)
    sqa_row.append(f"{sqa_row_f1_avg:.1f} / {sqa_row_acc_avg:.1f}")
    combined_data.append(sqa_row)

    # --- Add Overall Average Row (New) ---
    overall_f1_avg = overall_averages["overall_f1_avg"]
    overall_acc_avg = overall_averages["overall_acc_avg"]
    overall_avg_row = ["Overall (F1/Acc)"] + [""] * len(lang_order) + [f"{overall_f1_avg:.1f} / {overall_acc_avg:.1f}"]
    combined_data.append(overall_avg_row)


    # Create DataFrame
    df = pd.DataFrame(combined_data[1:], columns=combined_data[0])
    
    # Save to Excel
    writer = pd.ExcelWriter(output_file, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='F1_Acc_Combined') # Single sheet name
    
    # Get worksheet object for formatting
    workbook = writer.book
    worksheet = writer.sheets['F1_Acc_Combined']
    
    # Set column width and font
    for col_idx, col in enumerate(worksheet.columns):
        # Calculate max length more robustly, considering string representation
        max_length = 0
        for cell in col:
            cell_value = str(cell.value)
            if len(cell_value) > max_length:
                max_length = len(cell_value)
        worksheet.column_dimensions[get_column_letter(col[0].column)].width = max_length + 2
        
        for cell in col:
            cell.alignment = Alignment(horizontal='center')
            # Bold header row and the summary rows (SQA, Overall Average)
            if cell.row == 1 or \
               cell.row == len(combined_data) - 1 or \
               cell.row == len(combined_data):
                cell.font = Font(bold=True)
            # Make the labels for "Src/Tgt", "SQA", "整体平均" bold
            if (cell.row == len(combined_data) - 1 and col_idx == 0) or \
               (cell.row == len(combined_data) and col_idx == 0):
                 cell.font = Font(bold=True)
    
    # Save the file
    writer.close()
    print(f"\nExcel文件已保存至: {output_file}")


# 使用示例
import os

# 假设 output_file 是完整的文件路径或文件名（如 "output.jsonl"）
jsonl_file = output_file
name = os.path.splitext(os.path.basename(jsonl_file))[0]  # 获取不带扩展名的文件名
excel_file = f"{name}.xlsx"  # 输出Excel文件名

stats, total, valid_acc, valid_f1, src_all_lang_averages, overall_averages, sqa_averages = calculate_lang_scores(jsonl_file)
save_to_excel(stats, excel_file, src_all_lang_averages, overall_averages, sqa_averages)


