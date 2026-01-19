import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio
import pdb
import math
import random
from AgentCoop.graph.node import Node
from AgentCoop.agents.agent_registry import AgentRegistry
from AgentCoop.prompt.prompt_set_registry import PromptSetRegistry
from AgentCoop.rnn.rnn import AgentRNN
from AgentCoop.rnn.rnn import MLP
from AgentCoop.graph.constants import ADV_ROLES
from sentence_transformers import SentenceTransformer
import torch

sentence_transformer_dim_map = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "all-distilroberta-v1": 768,
}


class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                add_analysis:bool = True,
                analysis_output_dim:int = 16,
                input_dim:int = 128,
                hidden_dim:int = 64,
                solution_dim:int = 4,
                sentence_transformer: str = 'all-MiniLM-L6-v2',
                device: str = 'cuda',
                node_kwargs:List[Dict] = None,
                ):
        
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.agent_names:List[str] = agent_names
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        self.potential_edges:List[List[str, str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
    
        
        
        self.init_nodes() # add nodes to the self.nodes
        self.init_potential_edges() # add potential edges to the self.potential_edges

        num_nodes = len(self.nodes)
        self.node_id_to_idx = {node_id: idx for idx, node_id in enumerate(self.nodes.keys())}
        self.idx_to_node_id = {idx: node_id for idx, node_id in enumerate(self.nodes.keys())}
        self.comm_masks = torch.ones(num_nodes, num_nodes)

        # Store dimensions as instance variables
        self.solution_dim = solution_dim
        self.analysis_output_dim = analysis_output_dim
        analysis_input_dim = sentence_transformer_dim_map[sentence_transformer]
        # self.analysis_output_dim = sentence_transformer_dim_map[sentence_transformer]

        self.analysis_proj = MLP(input_size=analysis_input_dim, hidden_size=analysis_output_dim, output_size=analysis_output_dim)

        # Feature includes: solution + analysis_embed + reliable_score + neighbor_solution + neighbor_analysis_embed
        feature_dim = solution_dim + self.analysis_output_dim + 1 + solution_dim + self.analysis_output_dim

        # pdb.set_trace()

        self.rnn = AgentRNN(input_dim=feature_dim, hidden_dim=hidden_dim, rnn_type='rnn')

        self.sentence_transformer = SentenceTransformer(sentence_transformer)
        
        self.prompt_set = PromptSetRegistry.get(domain)

        self.device = device

        # self.analysis_proj.to(device)
        self.rnn.to(self.device)
        
        # Track adversarial agents and their fixed wrong answers for current query
        self.adversarial_answers = {}  # {node_id: fixed_answer}
    


    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name,kwargs in zip(self.agent_names,self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                if node1_id != node2_id:
                    self.potential_edges.append([node1_id,node2_id])

    def clear_comm_connection(self):
        """
        Clear all the communication connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].predecessors = []
            self.nodes[node_id].successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_comm_connection(self, reliable_scores:torch.Tensor, threshold:float = None): 
        self.clear_comm_connection()
        # if threshold:
        #     log_probs = [torch.tensor(0.0, requires_grad=False)]
        # else:
        #     log_probs = [torch.tensor(0.0, requires_grad=True)]
        log_probs = []
        # 1. Define epsilon for numerical stability
        epsilon = 1e-6

        sorted_scores, sorted_idx = torch.sort(reliable_scores, descending=True)

        adj = torch.zeros(len(self.nodes), len(self.nodes))

        for score, sender_idx_tensor in zip(sorted_scores, sorted_idx):
            # FIX 3: Define integer index once
            sender_idx = sender_idx_tensor.item() 
            sender_id = self.idx_to_node_id[sender_idx]
            sender = self.find_node(sender_id)
            
            sampling_prob = score 

            if threshold:
                # Note: Still requires attention to gradient flow if training=True
                # mean = torch.mean(reliable_scores)
                sampling_prob = torch.tensor(1.0 if score > threshold else 0.0)

            # 2. Stochastic Sampling Decision
            if torch.rand(1, device=self.device) < sampling_prob:
                # Connection is established
                stable_score = torch.clamp(score, min=epsilon)
                log_probs.append(torch.log(stable_score))
                
                # Connection logic
                for receiver_id, receiver_node in self.nodes.items():
                    if receiver_id != sender.id:
                        receiver_idx = self.node_id_to_idx[receiver_id]
                        
                        mask = self.comm_masks[receiver_idx, sender_idx] 
                        
                        if mask == 0.0:
                            continue
                        
                        # FIX 1: Swap arguments to check for path receiver -> sender
                        if not self.check_cycle(receiver_node, {sender}): 
                            sender.add_successor(receiver_node)
                            adj[receiver_idx, sender_idx] = 1.0
                        
            else:
                # Connection is NOT established
                stable_one_minus_score = torch.clamp(1.0 - score, min=epsilon)
                log_probs.append(torch.log(stable_one_minus_score))
        
        # print(adj)
        # pdb.set_trace()

        return torch.sum(torch.stack(log_probs)), adj


    def construct_node_features(self):
        node_features = []
        rnn_hidden_states = []
        for node_id, node in self.nodes.items():
            solution = node.last_memory['solution_embed'].to(self.device)
            analysis_embed = node.last_memory['analysis_embed'].to(self.device)
            reliable_scores = node.last_memory['reliable_scores'].detach().to(self.device).unsqueeze(0)
            rnn_hidden_state.append(node.last_memory['rnn_hidden_state'].to(self.device))
            # get received messages from neighbors
            neighbors_node_ids = node.last_memory['neighbors']
            if len(neighbors_node_ids) > 0:
                neighbor_solution, neighbor_analysis_embed = self.embed_messages_from_neighbors(neighbors_node_ids)
            else:
                neighbor_solution = torch.zeros(self.solution_dim, device=self.device)
                neighbor_analysis_embed = torch.zeros(self.analysis_output_dim, device=self.device)
            
            # pdb.set_trace()
            node_features.append(torch.cat([solution, analysis_embed, reliable_scores, neighbor_solution, neighbor_analysis_embed], dim=0))
        return torch.stack(node_features, dim=0), torch.stack(rnn_hidden_states, dim=0)
    

    def embed_messages_from_neighbors(self, neighbors_node_ids:List[str]):
        neighbor_solutions = []
        neighbor_analysis_embeds = []
        neighbor_reliable_scores = []
        for neighbor_node_id in neighbors_node_ids:
            neighbor_node = self.find_node(neighbor_node_id)
            neighbor_solutions.append(neighbor_node.last_memory['solution_embed'].to(self.device))
            neighbor_analysis_embeds.append(neighbor_node.last_memory['analysis_embed'].to(self.device))
            # neighbor_reliable_scores.append(neighbor_node.last_memory['reliable_scores'])
        
        # pdb.set_trace()
        
        # Stack to get shape (num_neighbors, embedding_dim)
        neighbor_solutions_tensor = torch.stack(neighbor_solutions, dim=0)  # (N, solution_dim)
        neighbor_analysis_embeds_tensor = torch.stack(neighbor_analysis_embeds, dim=0)  # (N, analysis_dim)
        # neighbor_reliable_scores_tensor = torch.stack(neighbor_reliable_scores, dim=0).unsqueeze(1).to(self.device)  # (N, 1)
        
        # Weighted sum: multiply by scores and sum across neighbors
        # neighbor_solution = (neighbor_solutions_tensor * neighbor_reliable_scores_tensor).sum(dim=0)  # (solution_dim,)
        # neighbor_analysis_embed = (neighbor_analysis_embeds_tensor * neighbor_reliable_scores_tensor).sum(dim=0)  # (analysis_dim,)
        neighbor_solution = neighbor_solutions_tensor.mean(dim=0)  # (solution_dim,)
        neighbor_analysis_embed = neighbor_analysis_embeds_tensor.mean(dim=0)  # (analysis_dim,)
        
        return neighbor_solution, neighbor_analysis_embed
    
    def get_weighted_majority_vote(self, reliable_scores:torch.Tensor):
        # Ensure scores sum to 1.0 (assuming reliable_scores are logits or raw values)
        agent_weights = torch.softmax(reliable_scores, dim=0) 

        # Initialize the aggregated vote vector (shape: solution_dim)
        weighted_answers = torch.zeros(self.solution_dim)

        for node_id, node in self.nodes.items():
            output = node.outputs[-1] 
            solution = self.prompt_set.postprocess_answer(output)
            solution_embed = self.prompt_set.change_solution_to_one_hot_vector(solution)
            node_idx = self.node_id_to_idx[node_id]
            agent_weight = agent_weights[node_idx].item()

            weighted_answers += solution_embed * agent_weight
        
        idx = torch.argmax(weighted_answers)

        # pdb.set_trace()
        
        # Convert the index back to a character answer (e.g., 0->A, 1->B)
        answer = chr(idx.item() + ord('A')) # Use .item() to convert tensor to standard Python int
        return answer
    
    def _sample_wrong(self, correct_answer: str, options=None) -> str:
        """Sample a wrong answer given the correct answer."""
        options = options or ["A", "B", "C", "D"]
        wrongs = [o for o in options if o != correct_answer]
        return random.choice(wrongs) if wrongs else correct_answer
    
    def adversarial_attack(self, attack_rate: float = 0.5, attack_mode: str = "random", 
                          correct_answer: str = None, same_wrong_answer: bool = True,
                          options: List[str] = None) -> List[str]:
        """
        Change some non-adversarial agents into adversarial agents.

        Args:
            attack_rate: fraction of available (non-adversarial) agents to convert (0.0-1.0).
            attack_mode: "random" (sample randomly) or "fixed" (take the first k non-adversarial agents in insertion order).
            correct_answer: the correct answer to generate wrong answers from.
            same_wrong_answer: if True, all adversarial agents output the same wrong answer;
                             if False, each adversarial agent outputs a different random wrong answer.
            options: list of possible answer options (default: ["A", "B", "C", "D"]).

        Returns:
            List of agent_ids that were changed to adversarial.
        """
        if not (0.0 <= attack_rate <= 1.0):
            raise ValueError("attack_rate must be in [0, 1]")

        # filter out already-adversarial
        potential_change_agents = [
            nid for nid, node in self.nodes.items()
            if getattr(node, "role", None) not in ADV_ROLES
            and getattr(node, "role", None) != "decision maker"
        ]

        m = len(potential_change_agents)
        num_to_change = int(math.floor(m * attack_rate))
        if num_to_change <= 0:
            return []

        if attack_mode == "fixed":
            selected_agents = potential_change_agents[-num_to_change:]
        elif attack_mode == "random":
            selected_agents = random.sample(potential_change_agents, num_to_change)
        else:
            raise ValueError("attack_mode must be 'fixed' or 'random'")

        # Generate wrong answer(s)
        if same_wrong_answer:
            # All adversarial agents share the same wrong answer
            shared_wrong_answer = self._sample_wrong(correct_answer, options)
        
        # Replace with adversarial agents and assign fixed wrong answer
        for agent_id in selected_agents:
            self.nodes[agent_id].role = "Fake"
            if same_wrong_answer:
                self.adversarial_answers[agent_id] = shared_wrong_answer
            else:
                # Each adversarial agent gets a different random wrong answer
                self.adversarial_answers[agent_id] = self._sample_wrong(correct_answer, options)
        
        return selected_agents

    def run(self, input: Any, 
                  num_rounds:int = 3, 
                  attack_rounds:int = 2,
                  attack_rate:float = 0.5,
                  attack_mode:str = 'fixed',
                  threshold:float = None,
                  max_tries: int = 3,
                  correct_answer: str = None,
                  same_wrong_answer: bool = True,
                  options: List[str] = None,
                  ) -> List[Any]:
        log_probs = 0
        
        # Clear adversarial answers for new query
        self.adversarial_answers = {}

        for round in range(num_rounds):
            # Perform adversarial attack at the specified round
            if round == attack_rounds and attack_rate > 0 and correct_answer is not None:
                self.adversarial_attack(
                    attack_rate=attack_rate,
                    attack_mode=attack_mode,
                    correct_answer=correct_answer,
                    same_wrong_answer=same_wrong_answer,
                    options=options
                )
            
            if round == 0:
                reliable_scores = torch.full(size=(len(self.nodes),), fill_value=0.5)
            else:
            
                node_features = self.construct_node_features()
             
                reliable_scores, self.rnn_hidden_state = self.rnn(node_features=node_features, hidden_state=self.rnn_hidden_state)
            
            round_log_probs, adj = self.construct_comm_connection(reliable_scores=reliable_scores, threshold=threshold)
            log_probs += round_log_probs
            
            in_degree = {node_id: len(node.predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(input, adversarial_answers=self.adversarial_answers) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory(input=input, reliable_scores=reliable_scores, adj=adj)
        
     
        # self.connect_decision_node()
        # await self.decision_node.async_execute(input)
        # final_answers = self.decision_node.outputs
        final_answers = self.get_weighted_majority_vote(reliable_scores)
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                  num_rounds:int = 3, 
                  attack_rounds:int = 2,
                  attack_rate:float = 0.5,
                  attack_mode:str = 'fixed',
                  threshold:float = None,
                  max_tries: int = 3, 
                  max_time: int = 600,
                  correct_answer: str = None,
                  same_wrong_answer: bool = True,
                  options: List[str] = None,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = []
        
        # Clear adversarial answers for new query
        self.adversarial_answers = {}

        for round in range(num_rounds):

            
            # Perform adversarial attack at the specified round
            if round == attack_rounds and attack_rate > 0 and correct_answer is not None:
                self.adversarial_attack(
                    attack_rate=attack_rate,
                    attack_mode=attack_mode,
                    correct_answer=correct_answer,
                    same_wrong_answer=same_wrong_answer,
                    options=options
                )
            
            

            
            if round == 0:
                reliable_scores = torch.full(size=(len(self.nodes),), fill_value=0.5, device=self.device)
                rnn_hidden_states = torch.zeros(len(self.nodes), self.rnn.hidden_dim, device=self.device)
            else:
                node_features, rnn_hidden_states = self.construct_node_features()
                reliable_scores, rnn_hidden_states = self.rnn(node_features=node_features, hidden_state=rnn_hidden_states)
            
            # print(f"reliable_scores: {reliable_scores}")
            
            log_prob, adj = self.construct_comm_connection(reliable_scores=reliable_scores, threshold=threshold)

            # print(f"adj: {adj}")
            log_probs.append(log_prob)
            
            in_degree = {node_id: len(node.predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(input, adversarial_answers=self.adversarial_answers),timeout=max_time) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            print(f'-------------------------------- round {round} --------------------------------')
            solution = []
            for node_id, node in self.nodes.items():
                solution.append(node.outputs[-1].strip()[0])
            roles = [node.role for node in self.nodes.values()]
            print(f"roles: {roles}")
            print(f"solution: {solution}, correct_answer: {correct_answer}")
            print(f"reliable_scores: {reliable_scores}")
            # pdb.set_trace()
            
            # pdb.set_trace()
            self.update_memory(input=input, reliable_scores=reliable_scores, adj=adj, rnn_hidden_states=rnn_hidden_states)
        
        # solution = []
        # for node_id, node in self.nodes.items():
            
        #     solution.append(node.outputs[-1].strip()[0])
        # pdb.set_trace()
            
        # self.connect_decision_node()
        # await self.decision_node.async_execute(input)
        # final_answers = self.decision_node.outputs
        final_answers = self.get_weighted_majority_vote(reliable_scores)
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        
        # pdb.set_trace()
        return final_answers, torch.sum(torch.stack(log_probs))
    
    def update_memory(self, input:Dict[str,str], reliable_scores:torch.Tensor, adj:torch.Tensor, rnn_hidden_states:torch.Tensor):
        for id, node in self.nodes.items():
            idx = self.node_id_to_idx[id]
            node.last_memory['reliable_scores'] = reliable_scores[idx]
            node.last_memory['inputs'] = input['task']
            # pdb.set_trace()
            if len(node.outputs) == 0:
                print(f"Warning: Node {id} has no outputs, skipping memory update")
                pdb.set_trace()
                continue
            output = node.outputs[-1]
            node.last_memory['outputs'] = output
            solution = self.prompt_set.postprocess_answer(output)
            analysis = output[1:].strip()
            node.last_memory['solution'] = solution
            solution_embed = self.prompt_set.change_solution_to_one_hot_vector(solution)
            node.last_memory['solution_embed'] = solution_embed
            analysis_embedding = torch.tensor(self.sentence_transformer.encode(analysis), device=self.device)
            node.last_memory['analysis_embed'] = self.analysis_proj(analysis_embedding).detach()
            # node.last_memory['analysis_embed'] = analysis_embedding
            # node.last_memory['analysis_embed'] = analysis_embedding
            # Find all nodes that send messages to this node (senders)
            # adj[idx, j] = 1 means node j sends to node idx
            neighbors_idx = torch.nonzero(adj[idx] == 1).flatten().tolist()
            neighbors_node_ids = [self.idx_to_node_id[neighbor_idx] for neighbor_idx in neighbors_idx]
            node.last_memory['neighbors'] = neighbors_node_ids
            node.last_memory['rnn_hidden_states'] = rnn_hidden_states[idx].detach()
    

    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

def min_max_norm(tensor:torch.Tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_0_to_1 = (tensor - min_val) / (max_val - min_val)
    normalized_minus1_to_1 = normalized_0_to_1 * 2 - 1
    return normalized_minus1_to_1
    