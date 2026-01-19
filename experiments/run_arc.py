import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
from typing import Union, Literal, List
import argparse
import random
import numpy as np
from AgentCoop.graph import Graph_ARC as Graph
from dataset.arc_challenge_dataset import load_arc_challenge
from experiments.train_arc import train
from experiments.evaluate_arc import evaluate
from AgentCoop.utils.const import AgentCoop_ROOT
import wandb
import torch



def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered','Star', 'Mesh',
                                 'FakeFullConnected','FakeRandom','FakeChain','FakeStar','FakeMesh','FakeAGRandom','FakeAGFull'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--llm_name', type=str, default="gpt-4.1-nano",
                        help="Model name, None runs the default ChatGPT4")
    parser.add_argument('--domain', type=str, default="arc-c",
                        help="Domain (the same as dataset name), default 'ARC-C'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[6],
                        help='Specify the number of agents for each name in agent_names')
    
    
    
    
    parser.add_argument('--num_epochs', type=float, default=1,
                        help="Number of optimization epochs. Default 10.")
    parser.add_argument('--num_rounds',type=int,default=4,
                        help="Number of optimization/inference rounds for one query")  
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="batch size")
    parser.add_argument('--decision_method', type=str, default="FinalMajorVote",
                        help="the decision method of the final node")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help="The threshold for the communication connection")
    parser.add_argument('--limit_train_questions', type=int, default=1120,
                        help="The number of questions to use for training")
    parser.add_argument('--limit_eval_questions', type=int, default=299,
                        help="The number of questions to use for evaluation")
    parser.add_argument('--eval_interval', type=int, default=10,
                        help="The interval of evaluation")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="The device to use")


    # RNN
    # parser.add_argument('--input_dim', type=int, default=128,
    #                     help="The input dimension of the GAT")
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help="The output dimension of the GAT")
    parser.add_argument('--analysis_output_dim', type=int, default=128,
                        help="The output dimension of the analysis")
    

    # features args
    parser.add_argument('--solution_dim', type=int, default=5,
                    help="The dimension of the solution")
    parser.add_argument('--add_analysis', action='store_false',
                        help="Whether to add the analysis to features")
    parser.add_argument('--sentence_transformer', type=str, default='all-MiniLM-L6-v2',
                        help="The sentence transformer model name")
    
    parser.add_argument('--sender_budget', type=int, default=1,
                        help="The budget for the sender")
    parser.add_argument('--receiver_budget', type=int, default=1,
                        help="The budget for the receiver")
    

    # attack args
    parser.add_argument('--attack_rounds', type=int, default=2,
                        help="The round of adversarial attack")
    parser.add_argument('--attack_rate', type=float, default=0.4,
                        help="The rate of adversarial attack")
    parser.add_argument('--attack_mode', type=str, default="random",
                        choices=['random', 'fixed', 'half'],
                        help="The mode of adversarial attack")
    
    # llm times args
    parser.add_argument('--max_tries', type=int, default=3,
                        help="The maximum number of tries for the LLM")
    parser.add_argument('--max_time', type=int, default=600,
                        help="The maximum time for the LLM")



    # wandb args
    parser.add_argument('--wandb_project', type=str, default="AgentCoop-ARC-ADV-Budget",
                        help="The wandb project name")
    parser.add_argument('--wandb_entity', type=str, default="hiwenzhe",
                        help="The wandb entity name")
    parser.add_argument('--wandb_name', type=str, default=None,
                        help="The wandb name")
    parser.add_argument('--wandb_mode', type=str, default='offline',
                        choices=['online', 'offline', 'disabled'],
                        help='Set Wandb mode: online, offline, or disabled.')
    
    # Debug flag
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode: small dataset subsets, fewer iterations.')
    # Add seed argument
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility. Default: 42")
    parser.add_argument('--comments', type=str, default="",
                        help="Comments for the run")
                

    args = parser.parse_args()
    result_path = AgentCoop_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
        
    return args


def set_seed(seed:int):
    """Set random seeds for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)  # FIX: Uncommented for full reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # FIX: Set seed for all GPUs



async def main():
    args = parse_args()
    
    mode = args.mode
    decision_method = args.decision_method
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(mode,len(agent_names))
    limit_eval_questions = args.limit_eval_questions
    limit_train_questions = args.limit_train_questions
    eval_interval = args.eval_interval
    num_epochs = args.num_epochs



    # Proper CUDA device initialization
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            args.device = 'cpu'
        else:
            torch.cuda.set_device(args.device) # Attempt to set based on arg
            print(f"Successfully set device to {torch.cuda.current_device()} based on args: {args.device}")



    # Construct a meaningful run name for logging and wandb
    name_parts = [
        args.mode,
        f"num{args.agent_nums[0]}" if args.agent_nums else "numX",
        f"epoch{args.num_epochs}", # Iterations per epoch
        f"evalint{args.eval_interval}", # Eval interval within epoch
        f"id{args.hidden_dim}",
        f"aod{args.analysis_output_dim}",
        f"sd{args.solution_dim}",
        f"st{args.sentence_transformer}",
        f"sb{args.sender_budget}",
        f"rb{args.receiver_budget}",
        f"lr{args.lr}",
        f"bs{args.batch_size}",
        f"rnd{args.num_rounds}",
        f"ct{args.threshold}",
        f"seed{args.seed}",
    ]

    if args.attack_rounds is not None:
        name_parts.append(f"ar{args.attack_rounds}")
    if args.attack_rate is not None:
        name_parts.append(f"art{args.attack_rate}")
    if args.attack_mode is not None:
        name_parts.append(f"atm{args.attack_mode}")
    if args.limit_train_questions is not None:
        name_parts.append(f"trn{args.limit_train_questions}")
    if args.limit_eval_questions is not None:
        name_parts.append(f"evl{args.limit_eval_questions}")
    name_parts.append(args.llm_name)
    if args.comments:
        name_parts.append(args.comments)
    run_name = "_".join(name_parts)
        
    # --- Wandb Initialization ---
    if args.wandb_mode != "disabled":
        final_run_name = args.wandb_name if args.wandb_name else run_name
        wandb_run = wandb.init(
            project=args.wandb_project + "-" + args.mode, # Append mode to project for better organization
            entity=args.wandb_entity,
            name=final_run_name,
            config=vars(args),
            mode=args.wandb_mode
        )
        print(f"Wandb run initialized: {final_run_name} (Mode: {args.wandb_mode})")
        if args.wandb_mode == "online" and wandb_run:
            print(f"Run URL: {wandb_run.url}")
    else:
        wandb_run = None
        print("Wandb logging disabled.")

    
    graph = Graph(domain=args.domain,
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  analysis_output_dim=args.analysis_output_dim,
                  hidden_dim=args.hidden_dim,
                  solution_dim=args.solution_dim,
                  sentence_transformer=args.sentence_transformer,
                  device=args.device,
                  sender_budget=args.sender_budget,
                  receiver_budget=args.receiver_budget,
                  **kwargs)
    
    dataset_train = load_arc_challenge(split="train")
    dataset_val = load_arc_challenge(split="validation")
    
    await train(graph=graph,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            num_epochs=num_epochs,
            num_rounds=args.num_rounds,
            lr=args.lr,
            batch_size=args.batch_size,
            eval_interval=eval_interval,
            limit_train_questions=limit_train_questions,
            limit_eval_questions=limit_eval_questions,
            attack_rounds=args.attack_rounds,
            attack_rate=args.attack_rate,
            attack_mode=args.attack_mode,
            # threshold=args.threshold,
            max_tries=args.max_tries,
            max_time=args.max_time,
            device=args.device,
            wandb_run=wandb_run)



def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star'],Literal['Mesh'],
                          Literal['FakeFullConnected'],Literal['FakeRandom'],Literal['FakeChain'],Literal['FakeStar'],Literal['FakeMesh'],Literal['FakeAGRandom'],Literal['FakeAGFull']],
               N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0]*N for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_mesh_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(0, N):
            for j in range(i+1,N):
                adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(1,N):
            adj_matrix[0][i] = 1
        return adj_matrix
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Normal'}]
    elif mode=='FullConnected' or mode == 'FakeFullConnected' or mode=='FakeAGFull':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random' or mode == 'FakeRandom' or mode == 'FakeAGRandom':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain' or mode == 'FakeChain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Mesh' or mode=='FakeMesh':
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star' or mode=='FakeStar':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    if 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':'Normal'} for i in range(N)]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':None} for i in range(N)]
        
    # return {"initial_spatial_probability": initial_spatial_probability,
    #         "fixed_spatial_masks": fixed_spatial_masks,
    #         "initial_temporal_probability": initial_temporal_probability,
    #         "fixed_temporal_masks": fixed_temporal_masks,
    return {
            "node_kwargs":node_kwargs}    

if __name__ == "__main__":
    asyncio.run(main())
