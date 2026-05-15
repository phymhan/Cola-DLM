# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
import difflib
import csv
import sys
import tempfile
import shutil

# ==========================================
# 1. 用户配置区域
# ==========================================

# 评测根目录（脚本会在该目录下自动搜索符合标准的子目录）
ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_output")

# 符合评测标准的子目录名前缀
EVAL_DIR_PREFIX = "tasks_"

# 是否递归搜索 ROOT_DIR 下的所有层级子目录
RECURSIVE_SEARCH = True

# 是否为每个任务输出 *_correct.jsonl 和 *_wrong.jsonl
WRITE_DETAIL_FILES = True

# 汇总 CSV 输出文件名与路径
SUMMARY_CSV_NAME = "accuracy_summary.csv"
SUMMARY_CSV_PATH = os.path.join(ROOT_DIR, SUMMARY_CSV_NAME)

# 任务配置：文件名 -> 合格阈值 (0.0 - 1.0)
# 8 tasks: MMLU, RACE, Story Cloze, LAMBADA, OBQA, HellaSwag, SIQA, SQuAD
TASK_CONFIG = {    
    "lambada.jsonl": 1.0,   
    "mmlu.jsonl": 1.0,
    "obqa.jsonl": 1.0,
    "hellaswag.jsonl": 1.0,   
    "race.jsonl": 1.0,
    "siqa.jsonl": 1.0,
    "squad.jsonl": 1.0,
    "story_cloze.jsonl": 1.0,  
}

# 默认阈值
DEFAULT_THRESHOLD = 1.0

# ==========================================
# 2. 数据预处理功能模块
# ==========================================

def process_line(data):
    """
    处理单行JSON数据：
    将 generate 字段按换行符分割，找到第一个能让前半部分包含实际单词或句子的 \n 进行分割。
    """
    if 'generate' in data and isinstance(data['generate'], str):
        content = data['generate']
        split_index = -1
        current_pos = 0
        
        while True:
            idx = content.find('\n', current_pos)
            if idx == -1:
                break
            
            part1_candidate = content[:idx]
            if re.search(r'\w', part1_candidate):
                split_index = idx
                break
            else:
                current_pos = idx + 1
        
        if split_index != -1:
            part1 = content[:split_index]
            part2 = content[split_index + 1:]
            
            data['generate'] = part1
            data['others'] = part2
            
    return data

def reorder_keys(data):
    """
    对字典的键进行重新排序：前4个(id, prompt, generate, ground_truth)，其余排后。
    """
    ordered_data = {}
    priority_keys = ["id", "prompt", "generate", "ground_truth"]
    
    for key in priority_keys:
        if key in data:
            ordered_data[key] = data[key]
            
    for key, value in data.items():
        if key not in priority_keys:
            ordered_data[key] = value
            
    return ordered_data

def preprocess_jsonl_file(file_path):
    """
    预处理单个JSONL文件：提取公共前缀，分割generate，重排键名，原地覆盖保存。
    """
    print(f"  -> [预处理] 开始清洗文件: {os.path.basename(file_path)}")
    all_lines_data = []
    prompts = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as source_file:
            for line_num, line in enumerate(source_file):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    processed_data = process_line(data)
                    all_lines_data.append(processed_data)
                    
                    if 'prompt' in processed_data and isinstance(processed_data['prompt'], str):
                        prompts.append(processed_data['prompt'])
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"    [错误] 读取文件 {file_path} 出错: {e}")
        return

    if not all_lines_data:
        return

    common_prefix = ""
    if prompts:
        common_prefix = os.path.commonprefix(prompts)
    
    if common_prefix:
        preview = common_prefix[:40].replace('\n', '\\n') + "..."
        print(f"    -> [预处理] 提取到 few-shot 前缀 (长度:{len(common_prefix)}) | 预览: {preview}")

    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(file_path), text=True)
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
            for data in all_lines_data:
                if common_prefix and 'prompt' in data and data['prompt'].startswith(common_prefix):
                    data['few_shot_prefix'] = common_prefix
                    data['prompt'] = data['prompt'][len(common_prefix):]
                
                final_data = reorder_keys(data)
                temp_file.write(json.dumps(final_data, ensure_ascii=False) + '\n')
        
        shutil.move(temp_path, file_path)
        print(f"    -> [预处理] 清洗完成，已覆写原文件")
    except Exception as e:
        print(f"    [错误] 写入临时文件出错: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ==========================================
# 3. 评测打分工具函数
# ==========================================

def normalize_text(text):
    if text is None: return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join(text.split())
    return text

def calculate_similarity(text1, text2):
    norm_t1 = normalize_text(text1)
    norm_t2 = normalize_text(text2)
    if not norm_t1 and not norm_t2: return 1.0
    matcher = difflib.SequenceMatcher(None, norm_t1, norm_t2)
    return matcher.ratio()

def get_first_word(text):
    norm_text = normalize_text(text)
    parts = norm_text.split()
    if parts: return parts[0]
    return ""

def extract_answer_segment(text):
    if text is None: return ""
    raw = str(text).strip()
    if not raw: return ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        m = re.search(r'(?i)\b(?:final\s+answer|answer)\b\s*(?:is|=|:|：)?\s*(.+)$', line)
        if m and m.group(1).strip(): return m.group(1).strip()
    return raw

def extract_choice_letter(text, max_choices=26):
    if text is None: return ""
    raw = str(text).strip()
    if not raw: return ""
    
    m = re.fullmatch(r'[\(\[]?\s*([A-Za-z])\s*[\)\]]?\.?', raw)
    if m:
        letter = m.group(1).upper()
        if 0 <= (ord(letter) - 65) < max_choices: return letter

    keyword_pattern = re.compile(r'(?i)\b(?:final\s+answer|answer|option|choice)\b\s*(?:is|=|:|：)?\s*[\(\[]?\s*([A-Za-z])\s*[\)\]]?(?=\s|$|[.,;:!?])')
    matches = keyword_pattern.findall(raw)
    if matches:
        letter = matches[-1].upper()
        if 0 <= (ord(letter) - 65) < max_choices: return letter

    if len(raw) <= 40:
        bracket_matches = re.findall(r'[\(\[]\s*([A-Za-z])\s*[\)\]]', raw)
        if bracket_matches:
            letter = bracket_matches[-1].upper()
            if 0 <= (ord(letter) - 65) < max_choices: return letter
    return ""

def match_choice_by_text(text, choices):
    if text is None or not choices: return ""
    norm_text = normalize_text(text)
    if not norm_text: return ""

    for i, choice in enumerate(choices):
        if normalize_text(choice) == norm_text: return chr(65 + i)

    cleaned = re.sub(r'(?i)^(the\s+)?(correct\s+)?(final\s+)?(answer|option|choice)\b\s*(is|=|:|：)?\s*', '', str(text)).strip()
    norm_cleaned = normalize_text(cleaned)
    if not norm_cleaned: return ""

    for i, choice in enumerate(choices):
        if normalize_text(choice) == norm_cleaned: return chr(65 + i)

    contained = [i for i, choice in enumerate(choices) if normalize_text(choice) and normalize_text(choice) in norm_cleaned]
    if len(contained) == 1: return chr(65 + contained[0])

    best_idx, best_score = -1, 0.0
    for i, choice in enumerate(choices):
        score = calculate_similarity(cleaned, choice)
        if score > best_score:
            best_idx = i
            best_score = score
    if best_idx >= 0 and best_score >= 0.9: return chr(65 + best_idx)
    return ""

def extract_mmlu_choice_letter(text, choices):
    max_choices = min(len(choices), 26)
    if max_choices == 0: return ""
    answer_segment = extract_answer_segment(text)
    for candidate in [answer_segment, text]:
        letter = extract_choice_letter(candidate, max_choices=max_choices)
        if letter: return letter
        letter = match_choice_by_text(candidate, choices)
        if letter: return letter
    return ""

def extract_gt_mmlu_choice_letter(gt_text, choices):
    max_choices = min(len(choices), 26)
    if max_choices == 0: return ""
    letter = extract_choice_letter(gt_text, max_choices=max_choices)
    if letter: return letter
    return match_choice_by_text(gt_text, choices)

# ==========================================
# 4. 核心文件处理逻辑 (预处理 + 评测)
# ==========================================

def process_single_file(file_path, threshold, write_detail_files=True):
    filename = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)
    file_stem = os.path.splitext(filename)[0]

    path_correct = os.path.join(dir_name, f"{file_stem}_correct.jsonl")
    path_wrong = os.path.join(dir_name, f"{file_stem}_wrong.jsonl")
    
    preprocess_jsonl_file(file_path)
    
    print(f"  -> [评测] 正在打分: {filename} | 阈值: {threshold}")

    stats = {"total": 0, "correct": 0, "wrong": 0, "accuracy": None}
    remainder_buckets = {}
    id_to_correct = {}
    id_to_remainder = {}
    f_cor = None
    f_err = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f_in:
            if write_detail_files:
                f_cor = open(path_correct, 'w', encoding='utf-8')
                f_err = open(path_wrong, 'w', encoding='utf-8')
            
            for line_idx, line in enumerate(f_in, 1):
                line = line.strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                    gen = data.get("generate", "")
                    gt = data.get("ground_truth", data.get("answer", ""))
                    
                    if filename == "lambada.jsonl":
                        word_gen = get_first_word(gen)
                        word_gt = get_first_word(gt)
                        score = 1.0 if word_gen == word_gt else 0.0
                        data['extracted_gen_word'] = word_gen
                        data['extracted_gt_word'] = word_gt
                        
                    elif filename in ["mmlu.jsonl", "obqa.jsonl", "race.jsonl", "siqa.jsonl"]:
                        choices = data.get("choices", [])
                        pred_choice = extract_mmlu_choice_letter(gen, choices)
                        gt_choice = extract_gt_mmlu_choice_letter(gt, choices)

                        if pred_choice and gt_choice and pred_choice == gt_choice:
                            score = 1.0
                        else:
                            sim_score = calculate_similarity(gen, gt)
                            score = 1.0 if sim_score >= threshold else sim_score
                            
                        data['extracted_gen_choice'] = pred_choice
                        data['extracted_gt_choice'] = gt_choice

                    else:
                        score = calculate_similarity(gen, gt)
                    
                    data['similarity_score'] = round(score, 4)
                    is_correct = score >= threshold
                    stats["total"] += 1
                    
                    if is_correct:
                        stats["correct"] += 1
                        if write_detail_files and f_cor is not None:
                            f_cor.write(json.dumps(data, ensure_ascii=False) + "\n")
                    else:
                        stats["wrong"] += 1
                        if write_detail_files and f_err is not None:
                            f_err.write(json.dumps(data, ensure_ascii=False) + "\n")

                    record_id = data.get("id")
                    if record_id is not None:
                        id_to_correct[record_id] = is_correct

                    remainder = data.get("prompt_len_mod_patch_size")
                    if remainder is not None:
                        r = int(remainder)
                        if record_id is not None:
                            id_to_remainder[record_id] = r
                        if r not in remainder_buckets:
                            remainder_buckets[r] = {"total": 0, "correct": 0, "wrong": 0, "records": []}
                        remainder_buckets[r]["total"] += 1
                        if is_correct:
                            remainder_buckets[r]["correct"] += 1
                        else:
                            remainder_buckets[r]["wrong"] += 1
                        remainder_buckets[r]["records"].append(data)

                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"  [错误] 第 {line_idx} 行出错: {e}")

        if stats["total"] > 0:
            acc = (stats["correct"] / stats["total"]) * 100
            stats["accuracy"] = acc
            print(f"  -> 结果: 准确率 {acc:.2f}% (对 {stats['correct']} / 错 {stats['wrong']})")
        else:
            print("  -> 无有效数据")

        remainder_stats = {}
        if remainder_buckets:
            for r in sorted(remainder_buckets.keys()):
                bucket = remainder_buckets[r]
                if bucket["total"] > 0:
                    r_acc = (bucket["correct"] / bucket["total"]) * 100
                    remainder_stats[r] = {
                        "total": bucket["total"],
                        "correct": bucket["correct"],
                        "wrong": bucket["wrong"],
                        "accuracy": r_acc,
                    }
                    print(f"    -> mod={r}: 准确率 {r_acc:.2f}% (对 {bucket['correct']} / 总 {bucket['total']})")

                if write_detail_files:
                    cor_path = os.path.join(dir_name, f"{file_stem}_mod{r}_correct.jsonl")
                    err_path = os.path.join(dir_name, f"{file_stem}_mod{r}_wrong.jsonl")
                    with open(cor_path, 'w', encoding='utf-8') as fc, \
                         open(err_path, 'w', encoding='utf-8') as fe:
                        for rec in bucket["records"]:
                            if rec.get("similarity_score", 0) >= threshold:
                                fc.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            else:
                                fe.write(json.dumps(rec, ensure_ascii=False) + "\n")

    except FileNotFoundError:
        print(f"  [错误] 找不到文件: {file_path}")
    finally:
        if f_cor is not None: f_cor.close()
        if f_err is not None: f_err.close()
    
    print("-" * 50)
    return stats, remainder_stats, id_to_correct, id_to_remainder

# ==========================================
# 5. 批处理与主入口
# ==========================================

def extract_run_alias(dir_path):
    dir_name = os.path.basename(os.path.normpath(dir_path))
    if dir_name.startswith(EVAL_DIR_PREFIX):
        alias = dir_name[len(EVAL_DIR_PREFIX):].strip()
        return alias if alias else dir_name
    return dir_name

def make_unique_alias(alias, used_aliases):
    if alias not in used_aliases: return alias
    idx = 2
    while f"{alias}_{idx}" in used_aliases: idx += 1
    return f"{alias}_{idx}"

def find_eval_dirs(root_dir):
    found = []
    seen = set()
    walker = os.walk(root_dir) if RECURSIVE_SEARCH else [(root_dir, *next(os.walk(root_dir))[1:])]
    
    for current_root, _, files in walker:
        dir_name = os.path.basename(os.path.normpath(current_root))
        if not dir_name.startswith(EVAL_DIR_PREFIX): continue
        if not [task for task in TASK_CONFIG if task in files]: continue

        norm_root = os.path.normpath(current_root)
        if norm_root in seen: continue
        seen.add(norm_root)
        found.append(norm_root)

    found.sort()
    return found

def evaluate_single_directory(eval_dir):
    print(f"\n评测目录: {eval_dir}")
    print("=" * 50)
    task_acc_map = {}
    task_remainder_map = {}
    task_id_correct_map = {}
    task_id_remainder_map = {}
    
    for task_file in TASK_CONFIG:
        file_path = os.path.join(eval_dir, task_file)
        if not os.path.isfile(file_path):
            print(f"跳过: {task_file} (文件不存在)\n" + "-" * 50)
            task_acc_map[task_file] = None
            continue

        threshold = TASK_CONFIG.get(task_file, DEFAULT_THRESHOLD)
        stats, remainder_stats, id_to_correct, id_to_remainder = process_single_file(
            file_path, threshold, write_detail_files=WRITE_DETAIL_FILES
        )
        task_acc_map[task_file] = stats.get("accuracy")
        if remainder_stats:
            task_remainder_map[task_file] = remainder_stats
        if id_to_correct:
            task_id_correct_map[task_file] = id_to_correct
        if id_to_remainder:
            task_id_remainder_map[task_file] = id_to_remainder

    return task_acc_map, task_remainder_map, task_id_correct_map, task_id_remainder_map

def write_summary_csv(summary_csv_path, results_by_alias, remainder_by_alias):
    aliases = list(results_by_alias.keys())
    tasks = list(TASK_CONFIG.keys())

    all_remainder_keys = set()
    for alias in aliases:
        for task, r_stats in remainder_by_alias.get(alias, {}).items():
            for r in r_stats:
                all_remainder_keys.add((task, r))

    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task"] + aliases)
        for task in tasks:
            task_stem = os.path.splitext(task)[0]
            row = [task_stem]
            for alias in aliases:
                acc = results_by_alias[alias].get(task)
                row.append("" if acc is None else f"{acc:.2f}")
            writer.writerow(row)

            task_remainders = sorted({r for t, r in all_remainder_keys if t == task})
            for r in task_remainders:
                row = [f"{task_stem}_mod{r}"]
                for alias in aliases:
                    r_stats = remainder_by_alias.get(alias, {}).get(task, {}).get(r)
                    if r_stats is None:
                        row.append("")
                    else:
                        row.append(f"{r_stats['accuracy']:.2f}")
                writer.writerow(row)

        avg_row = ["tasks_average"]
        avg_values = {}
        for alias in aliases:
            accs = [
                results_by_alias[alias].get(task)
                for task in tasks
                if results_by_alias[alias].get(task) is not None
            ]
            if accs:
                avg = sum(accs) / len(accs)
                avg_values[alias] = avg
                avg_row.append(f"{avg:.2f}")
            else:
                avg_values[alias] = None
                avg_row.append("")
        writer.writerow(avg_row)

    print("\n各 alias 的任务平均准确率:")
    for alias in aliases:
        avg = avg_values.get(alias)
        if avg is None:
            print(f"  {alias}: N/A")
        else:
            print(f"  {alias}: {avg:.2f}%")

def main():
    global ROOT_DIR, SUMMARY_CSV_PATH

    if len(sys.argv) >= 2 and sys.argv[1].strip(): ROOT_DIR = sys.argv[1].strip()
    if len(sys.argv) >= 3 and sys.argv[2].strip(): SUMMARY_CSV_PATH = sys.argv[2].strip()
    else: SUMMARY_CSV_PATH = os.path.join(ROOT_DIR, SUMMARY_CSV_NAME)

    if not os.path.exists(ROOT_DIR):
        print(f"错误: 目录 {ROOT_DIR} 不存在")
        return

    eval_dirs = find_eval_dirs(ROOT_DIR)
    if not eval_dirs:
        print(f"未在 {ROOT_DIR} 下找到符合标准的评测子目录。")
        return

    print("开始批量预处理与评测...\n" + "=" * 50)
    print(f"搜索根目录: {ROOT_DIR}")
    print(f"发现候选目录数: {len(eval_dirs)}")
    
    used_aliases = set()
    aliases_ordered = []
    results_by_alias = {}
    remainder_by_alias = {}
    id_correct_by_alias = {}
    id_remainder_by_alias = {}

    for eval_dir in eval_dirs:
        raw_alias = extract_run_alias(eval_dir)
        alias = make_unique_alias(raw_alias, used_aliases)
        used_aliases.add(alias)
        aliases_ordered.append(alias)
        
        task_acc_map, task_remainder_map, task_id_correct, task_id_remainder = evaluate_single_directory(eval_dir)
        results_by_alias[alias] = task_acc_map
        if task_remainder_map:
            remainder_by_alias[alias] = task_remainder_map
        if task_id_correct:
            id_correct_by_alias[alias] = task_id_correct
        if task_id_remainder:
            id_remainder_by_alias[alias] = task_id_remainder

    # Cross-reference: for aliases without remainder data (p=1),
    # compute per-remainder accuracy using the ID groupings from p>1 aliases.
    ref_id_remainder = {}
    for alias in aliases_ordered:
        for task, id_rem in id_remainder_by_alias.get(alias, {}).items():
            if task not in ref_id_remainder:
                ref_id_remainder[task] = {}
            ref_id_remainder[task].update(id_rem)

    for alias in aliases_ordered:
        if alias in remainder_by_alias:
            continue
        for task, ref_mapping in ref_id_remainder.items():
            id_correct = id_correct_by_alias.get(alias, {}).get(task, {})
            if not id_correct or not ref_mapping:
                continue
            per_r = {}
            for rec_id, rem in ref_mapping.items():
                if rec_id not in id_correct:
                    continue
                if rem not in per_r:
                    per_r[rem] = {"total": 0, "correct": 0, "wrong": 0}
                per_r[rem]["total"] += 1
                if id_correct[rec_id]:
                    per_r[rem]["correct"] += 1
                else:
                    per_r[rem]["wrong"] += 1
            r_stats = {}
            for rem, bucket in sorted(per_r.items()):
                if bucket["total"] > 0:
                    r_acc = (bucket["correct"] / bucket["total"]) * 100
                    r_stats[rem] = {
                        "total": bucket["total"],
                        "correct": bucket["correct"],
                        "wrong": bucket["wrong"],
                        "accuracy": r_acc,
                    }
                    print(f"  [交叉关联] {alias} | {task} | mod={rem}: "
                          f"准确率 {r_acc:.2f}% (对 {bucket['correct']} / 总 {bucket['total']})")
            if r_stats:
                if alias not in remainder_by_alias:
                    remainder_by_alias[alias] = {}
                remainder_by_alias[alias][task] = r_stats

    os.makedirs(os.path.dirname(SUMMARY_CSV_PATH) or ".", exist_ok=True)
    write_summary_csv(SUMMARY_CSV_PATH, results_by_alias, remainder_by_alias)
    print(f"\n汇总完成! CSV 已保存至: {SUMMARY_CSV_PATH}")

if __name__ == "__main__":
    main()
