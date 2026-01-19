import random
from typing import Optional, List, Dict, Union
import re
import pdb
from datasets import load_dataset

def load_openbookqa(split: str = "test", subset: str = "main", cache_dir: Optional[str] = None) -> List[Dict]:
    """
    Loads OpenBookQA maintaining original dataset order.
    Returns only the formatted question (with options) and the gold label.
    """
    # Loading the specific subset and split as shown in your screenshot
 
    ds = load_dataset("allenai/openbookqa", subset, split=split, cache_dir=cache_dir)
    processed_data = []

    for item in ds:
        # 1. Retrieve the question stem
        stem = item["question_stem"].strip()
        
        # 2. Extract choices and labels exactly as they appear
        # item['choices'] is a dict: {'text': ["choice1", ...], 'label': ["A", "B", ...]}
        choice_texts = item['choices']['text']
        choice_labels = item['choices']['label']
        
        # 3. Retrieve the dataset's answerKey as the 'gold' label
        gold = item['answerKey'].strip() # e.g., "D"

        # 4. Format the options into a single string for the question prompt
        formatted_options = []
        for i in range(len(choice_texts)):
            label = choice_labels[i]
            text = choice_texts[i].strip()
            formatted_options.append(f"{label}) {text}")

        # 5. Combine stem and options into one 'question' block
        full_question = f"{stem}\n" + "\n".join(formatted_options)

        processed_data.append({
            "question": full_question,
            "gold": gold
        })

    return processed_data

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