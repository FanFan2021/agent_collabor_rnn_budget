import os
import json
import math
import time
import asyncio
from typing import Union,Literal,Optional,Iterator,List,Any,Dict
from tqdm import tqdm
import copy

from AgentCoop.graph import Graph_MEDQA as Graph
from experiments.accuracy import Accuracy
from AgentCoop.utils.globals import Cost, PromptTokens, CompletionTokens
from dataset.medqa_dataset import postprocess_answer

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
        # threshold:float = 0.5,
        ) -> float:

    print(f"Evaluating AgentCoop on {dataset.__class__.__name__}")
    
    graph.rnn.eval()
    # graph.analysis_proj.eval()
    accuracy = Accuracy()
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
    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / eval_batch_size))

    if graph.domain == "mmlu":
        options = ["A", "B", "C", "D"]
    else:
        options = None
    

    predict_losses = []
    predict_accs = []

    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
        print(80*'-')

        start_ts = time.time()
        answer_log_probs = []
        
        for record in record_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.rnn = graph.rnn
            input_dict = {
                'task': record['question'],
            }
            correct_answer = record['gold']
            # print(input_dict)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(
                                                                        input_dict,
                                                                        num_rounds,
                                                                        attack_rounds=attack_rounds,
                                                                        attack_rate=attack_rate,
                                                                        attack_mode=attack_mode,
                                                                        correct_answer=correct_answer,
                                                                        same_wrong_answer=same_wrong_answer,
                                                                        # threshold=threshold,
                                                                        epsilon=epsilon,
                                                                        training=False,
                                                                        options=options,
                                                                        )
                                                                    ))
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, comm_log_probs, predict_loss, predict_acc = zip(*raw_results)
        # print(f"Batch time {time.time() - start_ts:.3f}")
        for raw_answer, record, loss, acc in zip(raw_answers, record_batch, predict_loss, predict_acc):
            # print("Raw answer:", raw_answer)d  vb
            answer = postprocess_answer(raw_answer)
            # print("Postprocessed answer:", answer)
            correct_answer = record['gold']
            print(f"evaluation answer:{answer}, correct answer:{correct_answer}")
            accuracy.update(answer, correct_answer)
            accuracy.print()

            predict_losses.append(loss)
            predict_accs.append(acc)
        # print(f"Cost {Cost.instance().value}")
        # print(f"PromptTokens {PromptTokens.instance().value}")
        # print(f"CompletionTokens {CompletionTokens.instance().value}")
    accuracy.print()
    print("Done!")

    avg_predict_loss = torch.mean(torch.stack(predict_losses))
    avg_predict_acc = torch.mean(torch.stack(predict_accs))

    return accuracy.get(), avg_predict_loss, avg_predict_acc


def dump_eval_results(self, dct: Dict[str, Any]) -> None:
    if self._art_dir_name is not None:
        eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
        with open(eval_json_name, "w") as f:
            json.dump(dct, f)
