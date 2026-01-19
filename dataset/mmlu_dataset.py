import glob
import pandas as pd
from typing import Union, List, Literal, Any, Dict
import numpy as np
from abc import ABC
import os
import re

class MMLUDataset(ABC):
    def __init__(self,
        split: Union[Literal['dev'], Literal['val'], Literal['test']],
        ) -> None:

        self._split = split

        # data_path = f"datasets/MMLU/data/{self._split}/"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, f"MMLU/data/{self._split}/")
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'mmlu'

    @staticmethod
    def _load_data(
        data_path: str,
        ) -> pd.DataFrame:

        rng = np.random.default_rng(888)

        csv_paths = glob.glob(data_path + "*.csv")
        csv_paths = sorted(csv_paths)
        print("Number of topics: ", len(csv_paths))

        names = ['question', 'A', 'B', 'C', 'D', 'correct_answer']

        total_df = pd.DataFrame(columns=names)
        for path in csv_paths:
            single_df = pd.read_csv(path, header=None,
                            names=names,encoding='utf-8')
            total_df = pd.concat([total_df, single_df])

        total_df = total_df.reset_index(drop=True)

        # Pseudorandom shuffle
        total_df = total_df.reindex(rng.permutation(total_df.index))

        print("Total number of questions: ", len(total_df))

        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_input(record: pd.DataFrame) -> Dict[str, Any]:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
            )
        input_dict = {"task": demo_question}
        return input_dict

    # def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
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

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer

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