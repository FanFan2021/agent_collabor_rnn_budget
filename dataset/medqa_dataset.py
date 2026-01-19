import json
import torch
from typing import List, Dict
from typing import Union
import re
import os
import pdb

def load_medqa(file_path: str) -> List[Dict]:
    """
    Loads MedQA data by first reading all lines into memory.
    """
    processed_data = []
    label_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # pdb.set_trace()
    data_path = current_dir + file_path

    # Step 1: Load all lines into a variable (list of strings)
    with open(data_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    # Step 2: Process each line from the variable
    for line in all_lines:
        item = json.loads(line)
        
        # 1. Formatting Prompt
        stem = item["question"].strip()
        options = item["options"]
        ordered_labels = sorted(options.keys())
        choice_lines = [f"{lab}: {options[lab]}" for lab in ordered_labels]
        full_prompt = stem + "\n" + "\n".join(choice_lines)

        # 2. Answer Normalization
        ans_idx_raw = item.get("answer_idx", "").strip().upper()
        gold_val = float(label_to_idx.get(ans_idx_raw, -1))

        if gold_val != -1:
            processed_data.append({
                'question': full_prompt,
                'solution': gold_val,
                'gold': ans_idx_raw,
            })

    print(f"Loaded {len(processed_data)} MedQA samples from memory.")
    return processed_data


# def postprocess_answer(answer: Union[str, List[str]]) -> str:
#     if isinstance(answer, list):
#         if len(answer) > 0:
#             answer = answer[0]
#         else:
#             answer = ""
#     if not isinstance(answer, str):
#         raise Exception("Expected string")
#     if len(answer) > 0:
#         ans_pos = answer.find("answer is")
#         if ans_pos != -1:
#             answer = answer[ans_pos+len("answer is"):].strip(":").strip().strip("Option").strip()
#         answer = answer[0] # Try to format the answer by taking the first letter
#     return answer

def postprocess_answer(answer: Union[str, List[str]]) -> str:
    # 1. Handle List/Type checks
    if isinstance(answer, list):
        answer = answer[0] if len(answer) > 0 else ""
    if not isinstance(answer, str):
        raise Exception("Expected string")
    
    answer = answer.strip()
    if not answer:
        return ""

    # 2. Priority 1: Check the First Line
    lines = answer.split('\n')
    first_line = lines[0].strip()
    
    # Matches "A", "A.", "(A)", or "A:" at the very start of the first line
    first_line_match = re.match(r"^[\(\[]?([A-E])[\)\]]?[\s\.\:]*$", first_line, re.IGNORECASE)
    if first_line_match:
        return first_line_match.group(1).upper()

    # 3. Priority 2: Fallback to "answer is" logic in full text
    # This catches the "Therefore, the correct answer is Mudville (C)" style
    ans_pos = answer.lower().find("answer is")
    if ans_pos != -1:
        # Look at the text immediately following "answer is"
        after_phrase = answer[ans_pos + len("answer is"):].strip()
        # Find the first [A-E] in the following 20 characters
        keyword_match = re.search(r"([A-E])", after_phrase, re.IGNORECASE)
        if keyword_match:
            return keyword_match.group(1).upper()

    # 4. Final Fallback: First standalone letter [A-E] anywhere in the text
    # Standard practice for messy LLM outputs
    fallback_match = re.search(r"\b([A-E])\b", answer, re.IGNORECASE)
    if fallback_match:
        return fallback_match.group(1).upper()

    # If all fails, take the first non-whitespace character as a last resort
    return answer[0].upper() if answer else ""