import torch
from typing import Iterator
import pandas as pd
import numpy as np
import time
import asyncio
from typing import List
import copy

from AgentCoop.graph import Graph_MMLU as Graph
from experiments.accuracy import Accuracy
from AgentCoop.utils.globals import Cost, PromptTokens, CompletionTokens
from experiments.evaluate_mmlu import evaluate
from dataset.mmlu_dataset import postprocess_answer
from typing import Optional
import wandb
import torch
import math
import pdb


# Global counters
_global_step = 0

def get_global_step():
    global _global_step
    return _global_step

def increment_global_step():
    global _global_step
    _global_step += 1
    return _global_step


async def train(graph:Graph,
            dataset_train,
            dataset_val,
            num_epochs:int=100,
            num_rounds:int=1,
            lr:float=0.1,
            batch_size:int = 4,
            eval_interval:int = 10,
            limit_train_questions:int = 153,
            limit_eval_questions:int = 153,
            attack_rounds:int = 2,
            attack_rate:float = 0.5,
            attack_mode:str = 'fixed',
            epsilon:float = 0.1,
            max_tries:int = 3,
            max_time:int = 600,
            device:str = 'cuda',
            wandb_run:Optional[wandb.run] = None,
          ) -> None:

    
    
    
    
    def infinite_data_loader(size_to_use:int) -> Iterator:
        # Determine the actual size to use
        # dataset_size = len(dataset_train)
        print(f"Total number of questions in the training set: {size_to_use}")
        
        while True:
            # Create permutation for the actual size we're using
            perm = np.random.permutation(size_to_use)
            for idx in perm:
                record = dataset_train[int(idx)]  
                yield record
    
     
    size_to_use = min(limit_train_questions, len(dataset_train)) if limit_train_questions is not None else len(dataset_train)
    loader = infinite_data_loader(size_to_use=size_to_use)
    num_batches = int(math.ceil(size_to_use / batch_size))
    num_iters = int(num_epochs * num_batches)
    
    # optimizer = torch.optim.Adam(list(graph.rnn.parameters()) + list(graph.analysis_proj.parameters()), lr=lr)    
    optimizer = torch.optim.Adam(graph.rnn.parameters(), lr=lr)    
    graph.rnn.train()
    # graph.analysis_proj.train()


    global _global_step
    print(f"Total number of questions in the training set: {limit_train_questions}")
    print(f"Total number of questions in the evaluation set: {limit_eval_questions}")
    print(f"Total number of batches: {num_batches}")
    print(f"Total number of iterations: {num_iters}")
    # print(f"Total number of parameters: {sum(p.numel() for p in graph.rnn.parameters()) + sum(p.numel() for p in graph.analysis_proj.parameters())}")
    print(f"Total number of parameters: {sum(p.numel() for p in graph.rnn.parameters())}")

    if graph.domain == "mmlu":
        options = ["A", "B", "C", "D"]
    else:
        options = None


    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80*'-')
        start_ts = time.time()
        correct_answers = []
        answer_log_probs = []

        for i_record, record in zip(range(batch_size), loader):
            realized_graph = copy.deepcopy(graph)
            realized_graph.rnn = graph.rnn
            # realized_graph.analysis_proj = graph.analysis_proj
            input_dict = dataset_train.record_to_input(record)
            # input_dict['A'] = record['A']
            # input_dict['B'] = record['B']
            # input_dict['C'] = record['C']
            # input_dict['D'] = record['D']
            correct_answer = dataset_train.record_to_target_answer(record)
            # print(input_dict)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(
                                                                    input=input_dict,
                                                                    num_rounds=num_rounds,
                                                                    attack_rounds=attack_rounds,
                                                                    attack_rate=attack_rate,
                                                                    attack_mode=attack_mode,
                                                                    correct_answer=correct_answer,
                                                                    same_wrong_answer=True,
                                                                    # threshold=None,  # CRITICAL: Must be None for REINFORCE training (stochastic sampling)
                                                                    training=True,
                                                                    epsilon=epsilon,
                                                                    max_tries=max_tries,
                                                                    max_time=max_time,
                                                                    options=options,
                                                                    i_iter=i_iter,
                                                                )))
           
            correct_answers.append(correct_answer)
        
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, comm_log_probs, predict_loss, predict_acc = zip(*raw_results)
       
        answers: List[str] = []
        rewards: List[float] = []

        predict_losses: List[float] = []
        predict_accs: List[float] = []
        
        batch_acc = Accuracy()
        for raw_answer, correct_answer, loss, acc in zip(raw_answers, correct_answers, predict_loss, predict_acc):
            answer = postprocess_answer(raw_answer)
            answers.append(answer)
            assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
            accuracy = Accuracy()
            accuracy.update(answer, correct_answer)
            batch_acc.update(answer, correct_answer)

            utility = accuracy.get()
            print(f"Training, answer: {answer}, correct_answer: {correct_answer}, Utility: {utility}")
            
            # Use reward centered at 0: correct=+1, wrong=-1
            reward = 2 * utility - 1
            rewards.append(reward)

            predict_losses.append(loss)
            predict_accs.append(acc)
        
        # Compute baseline as mean reward in batch (REINFORCE with baseline)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        baseline = torch.mean(rewards)

        # compute the decay sum of comm log probs
        # pdb.set_trace()
        gamma = 0.9
        # pdb.set_trace()
        comm_log_probs = torch.stack(comm_log_probs)
        # pdb.set_trace()
        weights = gamma ** torch.arange(num_rounds-1, -1, -1, device=device)
        comm_log_probs = (comm_log_probs * weights).sum(dim=1)
        
        # pdb.set_trace()

        # decision_log_prob = torch.stack(decision_log_prob)

        log_probs = comm_log_probs

        advantage = rewards - baseline

        loss = - log_probs * advantage
        # REINFORCE with baseline: reduces variance significantly
        total_loss = torch.mean(loss)
        optimizer.zero_grad() 
        total_loss.backward()
        optimizer.step()

        avg_predict_acc = torch.mean(torch.stack(predict_accs))
        avg_predict_loss = torch.mean(torch.stack(predict_losses))

        train_prompt_tokens = PromptTokens.instance().value 
        train_completion_tokens = CompletionTokens.instance().value
        train_tokens = train_prompt_tokens + train_completion_tokens


        

        if wandb_run is not None:
            current_step = increment_global_step()
            wandb_run.log({
                'train/batch_accuracy': batch_acc.get(),
                'train/loss': total_loss.item(),
                'train/batch_baseline': baseline.item(),
                'train/cost': Cost.instance().value,
                'train/prompt_tokens': train_prompt_tokens,
                'train/completion_tokens': train_completion_tokens,
                'train/tokens': train_tokens,
                'train/predict_loss': avg_predict_loss.item(),
                'train/predict_acc': avg_predict_acc.item(),
                'train/batch_time': time.time() - start_ts,
            }, step=current_step)



                # Evaluation phase
        if i_iter % eval_interval == 0 or i_iter == num_iters - 1:
            
            # FIX: Switch to evaluation mode for the evaluation phase
            # This disables dropout, batch norm training, etc.
            # graph.gat.eval()
            graph.rnn.eval()
            # graph.node_encoder.eval()
            # graph.analysis_proj.eval()
            # graph.decision_encoder.eval()
            # graph.analysis_encoder.eval()
            with torch.no_grad():  # Disable gradient computation during evaluation
                eval_start = time.time()
                
                # Note: Can use threshold for deterministic evaluation (no gradients needed)
                score, predict_loss, predict_acc = await evaluate(graph=graph,
                                    dataset=dataset_val,
                                    num_rounds=num_rounds,
                                    limit_questions=limit_eval_questions,
                                    eval_batch_size=batch_size,
                                    attack_rounds=attack_rounds,
                                    attack_rate=attack_rate,
                                    attack_mode=attack_mode,
                                    # threshold=threshold,  # OK to use threshold during eval
                                    epsilon=0.1,
                                    # training=False,
                                    )
                eval_time = time.time() - eval_start
                print(f"Evaluation time: {eval_time:.3f} sec")
                print(f"Eval score (Iter {i_iter + 1}): {score}")
                print(f"Eval predict_loss (Iter {i_iter + 1}): {predict_loss}")
                print(f"Eval predict_acc (Iter {i_iter + 1}): {predict_acc}")

                eval_prompt_tokens = PromptTokens.instance().value
                eval_completion_tokens = CompletionTokens.instance().value
                eval_tokens = eval_prompt_tokens + eval_completion_tokens
         
                
                if wandb_run:
                    wandb_run.log({
                        "eval/score": score,
                        'eval/cost': Cost.instance().value,
                        'eval/prompt_tokens': eval_prompt_tokens,
                        'eval/completion_tokens': eval_completion_tokens,
                        'eval/tokens': eval_tokens,
                        "eval/diff_prompt_tokens": eval_prompt_tokens - train_prompt_tokens,
                        "eval/diff_completion_tokens": eval_completion_tokens - train_completion_tokens,
                        "eval/diff_tokens": eval_tokens - train_tokens,
                        "eval/diff_per_query_agent_round": (eval_tokens - train_tokens) / (num_rounds * limit_eval_questions),
                        'eval/eval_time': eval_time,
                        'eval/predict_loss': predict_loss.item(),
                        'eval/predict_acc': predict_acc.item(),
                    }, step=current_step)



def print_grad_graph(tensor, name="tensor"):
    fn = tensor.grad_fn
    print(f"Computational graph for {name}:")
    
    if fn is None:
        print("  (no grad_fn: tensor is a leaf or detached)")
        return
    
    i = 0
    visited = set()
    while fn is not None and fn not in visited:
        visited.add(fn)
        tname = type(fn).__name__
        print(f"{i}: {tname} ; fn = {fn}")

        # Check if this fn corresponds to a named variable
        if hasattr(fn, "variable") and hasattr(fn.variable, "name"):
            print("    ↳ variable:", fn.variable.name)
        
        # inspect next_functions (list of (Function, index))
        nexts = getattr(fn, "next_functions", None)
        if not nexts:
            break
        
        # go to first next function that is not None
        found = False
        for nf, idx in nexts:
            if nf is not None:
                fn = nf
                found = True
                break
        if not found:
            break
        
        i += 1

