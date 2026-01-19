from typing import Iterable, Dict, Optional, Union, List
from datasets import load_dataset
import pdb
from AgentCoop.utils.utils import normalize_answer
import re


def load_arc_challenge(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split, cache_dir=cache_dir)
    processed_data = []
    
    # Expanded map to handle more choices if they appear
    label_map = {
        "1": "A", "2": "B", "3": "C", "4": "D", "5": "E", "6": "F",
        "A": "A", "B": "B", "C": "C", "D": "D", "E": "E"
    }

    def map_label(l: str) -> str:
        s = str(l).strip().upper()
        # Returns the mapped letter if it exists, otherwise returns the original string
        return label_map.get(s, s)

    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]

        formatted_choices = {}
        mapped_order = []
        
        # 1. Process choices and unify labels
        for label, text in zip(labels, texts):
            mlabel = map_label(label)
            formatted_choices[mlabel] = text.strip()
            mapped_order.append(mlabel)

        # 2. Build the formatted question string
        # Using sorted mapped_order or direct iteration to ensure A, B, C order
        ordered_labels = sorted(formatted_choices.keys())
        choice_lines = [f"{lab}: {formatted_choices[lab]}" for lab in ordered_labels]
        question_text = stem + "\n" + "\n".join(choice_lines)

        # 3. Handle the Answer Key
        raw_answer = item.get("answerKey", "").strip()
        mapped_answer = map_label(raw_answer)
        
        # Ensure 'gold' uses the unified A/B/C/D format
        gold = normalize_answer(mapped_answer)

        data_entry = {
            "question": question_text,
            "solution": mapped_answer,
            "gold": gold,
        }
        
        # Add individual choice fields for compatibility with GNN/RNN architectures
        for label in formatted_choices:
            data_entry[label] = f"{label}: {formatted_choices[label]}"
        
        processed_data.append(data_entry)
        
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
