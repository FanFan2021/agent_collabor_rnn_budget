import os
import json
import math
import time
import asyncio
from typing import Union,Literal,Optional,Iterator,List,Any,Dict
from tqdm import tqdm
import copy
import pdb
from AgentCoop.graph import Graph_GSM8K as Graph
from experiments.accuracy import Accuracy
from AgentCoop.utils.globals import Cost, PromptTokens, CompletionTokens
from dataset.gsm8k_dataset import postprocess_answer
import itertools
import pdb
from AgentCoop.utils.utils import count_dataset

async def evaluate(
        graph:Graph,
        dataset,
        num_rounds:int = 1,
        limit_questions: Optional[int] = None,
        eval_batch_size: int = 4,
        attack_rounds:int = 2,
        attack_rate:float = 0.5,
        attack_mode:str = 'fixed',
        epsilon:float = 0.1,
        same_wrong_answer:bool = True,
        ) -> float:

    print(f"Evaluating gdesigner on GSM8K split test")

    accuracy = Accuracy()
    graph.rnn.eval()

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None:
                if i_record >= limit_questions:
                    break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if len(records) > 0:
            yield records
        return
    # This handles the "limit_questions" logic without needing len()
   
    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / eval_batch_size))

    # pdb.set_trace()
    

    if data_len == 0:
        pdb.set_trace()

    # pdb.set_trace()

    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
        print('eval-batch-'+'-'*80)

        start_ts = time.time()
        answers = []
        
        for record in record_batch:
            realized_graph = copy.deepcopy(graph)
            input_dict = {
                'question': record['question'],
                # 'solution': record['solution'],
                # 'gold': record['gold']
            }
            correct_answer = int(record['gold'])
            # print(input_dict)
            answers.append(asyncio.create_task(realized_graph.arun(
                                                                input=input_dict,
                                                                num_rounds=num_rounds,
                                                                attack_rounds=attack_rounds,
                                                                attack_rate=attack_rate,
                                                                attack_mode=attack_mode,
                                                                correct_answer=correct_answer,
                                                                same_wrong_answer=same_wrong_answer,
                                                                # threshold=threshold,
                                                                epsilon=epsilon,
                                                                training=False,
                                                                options=None,
                                                                )))
        raw_results = await asyncio.gather(*answers)
        raw_answers, _ = zip(*raw_results)
        # pdb.set_trace()
        print(f"Batch time {time.time() - start_ts:.3f}")
        for raw_answer, record in zip(raw_answers, record_batch):
            # print("Raw answer:", raw_answer)
            # print("Postprocessed answer:", answer)
            # correct_answer = dataset.record_to_target_answer(record)
            # answer = postprocess_answer(raw_answer)
            # pdb.set_trace()
            answer = int(raw_answer)
            correct_answer = int(record['gold'])
            print(f"answer: {answer}, correct_answer: {correct_answer}")
            # print("Correct answer:", correct_answer)
            accuracy.update(answer, correct_answer)
            accuracy.print()
        # print(f"Cost {Cost.instance().value}")
        # print(f"PromptTokens {PromptTokens.instance().value}")
        # print(f"CompletionTokens {CompletionTokens.instance().value}")
    accuracy.print()
    print("Done!")

    return accuracy.get()


def dump_eval_results(self, dct: Dict[str, Any]) -> None:
    if self._art_dir_name is not None:
        eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
        with open(eval_json_name, "w") as f:
            json.dump(dct, f)
