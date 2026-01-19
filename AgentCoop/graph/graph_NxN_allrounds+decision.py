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
                sentence_transformer: str = 'all-mpnet-base-v2',
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
        self.comm_masks = torch.ones(num_nodes, num_nodes, device=device)
        self.comm_masks.fill_diagonal_(0.0)

        # Store dimensions as instance variables
        self.solution_dim = solution_dim
        self.analysis_output_dim = analysis_output_dim
        analysis_input_dim = sentence_transformer_dim_map[sentence_transformer]
        # self.analysis_output_dim = sentence_transformer_dim_map[sentence_transformer]

        self.analysis_proj = MLP(input_size=analysis_input_dim, hidden_size=analysis_output_dim, output_size=analysis_output_dim)

        # Feature includes: solution + analysis_embed + reliable_score + neighbor_solution + neighbor_analysis_embed
        feature_dim = solution_dim + self.analysis_output_dim + 1 + solution_dim + self.analysis_output_dim + 4 + 1

        # pdb.set_trace()

        self.rnn = AgentRNN(input_dim=feature_dim, hidden_dim=hidden_dim, rnn_type='rnn')

        self.hidden_dim = hidden_dim

        self.sentence_transformer = SentenceTransformer(sentence_transformer)
        
        self.prompt_set = PromptSetRegistry.get(domain)

        self.device = device

        self.analysis_proj.to(self.device)
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
    

    def clear_decision_edges(self):
        """
        Clear all the decision edges of the nodes in the graph.
        """
        self.decision_node.successors = []
        self.decision_node.predecessors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_comm_connection(self, reliable_scores: torch.Tensor, threshold: float = None, training: bool = True): 
        self.clear_comm_connection()
        
        # Initialize with 0.0 to prevent errors if no edges are added
        log_probs = [torch.tensor(0.0, device=self.device, requires_grad=training)] 
        
        # 1. Define epsilon for numerical stability
        epsilon = 1e-6

        if threshold is not None:
            adaptive_threshold = torch.round(reliable_scores.mean(), decimals=4)
            edge_prob = torch.tensor(1.0 if reliable_scores > adaptive_threshold else 0.0, device=self.device)
        else:
            


        sorted_scores, sorted_idx = torch.sort(reliable_scores, descending=True)

        # Broadcasting to get N x N probability matrix
        if reliable_scores.dim() == 1:
            reliable_scores = reliable_scores.view(-1, 1)
        edge_probs = torch.min(reliable_scores, reliable_scores.t())

        # pdb.set_trace()

        adj = torch.zeros(len(self.nodes), len(self.nodes), device=self.device)

        for sender_idx_tensor in sorted_idx:
            # FIX 1: Convert tensor to integer for dict lookup
            sender_idx = sender_idx_tensor.item() 
            sender_id = self.idx_to_node_id[sender_idx]
            sender = self.find_node(sender_id)

            for receiver_idx_tensor in sorted_idx:
                # FIX 1: Convert tensor to integer
                receiver_idx = receiver_idx_tensor.item()
                receiver_id = self.idx_to_node_id[receiver_idx]
                receiver = self.find_node(receiver_id)

                if sender.id == receiver.id:
                    continue
                
                # Check mask using integer indices
                if self.comm_masks[receiver_idx, sender_idx] == 0.0:
                    continue
                
                # Check for cycle (Receiver -> ... -> Sender exists?)
                # This relies on 'sender' and 'receiver' objects being up-to-date
                if self.check_cycle(receiver, {sender}):
                    continue
                
                # pdb.set_trace()

                # Get probability
                edge_prob = edge_probs[receiver_idx, sender_idx]
                
                # Determine sampling probability (separate from gradient path if needed)
                sampling_prob = edge_prob
                if threshold:
                    # pdb.set_trace()
                    sampling_prob = torch.tensor(1.0 if edge_prob > adaptive_threshold else 0.0, device=self.device)

                # Stochastic Sampling
                if torch.rand(1, device=self.device) < sampling_prob:
                    # 1. Update Adjacency Matrix
                    adj[receiver_idx, sender_idx] = 1.0
                    
                    # FIX 2: Update the Node Object so subsequent check_cycle calls work
                    sender.add_successor(receiver) 

                    # 2. Calculate Log Prob (Use STABLE edge_prob for gradients)
                    # We clamp to avoid log(0)
                    stable_prob = torch.clamp(edge_prob, min=epsilon, max=1.0-epsilon)
                    log_probs.append(torch.log(stable_prob))
                else:
                    # Edge NOT added
                    stable_prob_neg = torch.clamp(1.0 - edge_prob, min=epsilon, max=1.0-epsilon)
                    log_probs.append(torch.log(stable_prob_neg))
        
        return torch.sum(torch.stack(log_probs)), adj


    def construct_node_features(self):
        node_features = []
        rnn_hidden_states = []

        # FIX 1: Iterate by Index to ensure alignment with Adjacency Matrix
        # (Assuming self.idx_to_node_id is correctly built)
        for idx in range(len(self.nodes)):
            node_id = self.idx_to_node_id[idx]
            node = self.nodes[node_id]

            # 1. Retrieve Self Features
            # Ensure 1D shapes for concatenation
            solution = node.last_memory['solution_embed'].to(self.device).view(-1) 
            analysis_embed = node.last_memory['analysis_embed'].to(self.device).view(-1)
            
            # reliable_scores usually scalar. view(-1) ensures it is shape [1]
            reliable_scores = node.last_memory['reliable_scores'].detach().to(self.device).view(-1)

            # 2. Retrieve Hidden State (Handle initialization case)
            h_state = node.last_memory['rnn_hidden_states']
            if h_state is None:
                # Initialize zeros if not found (First Round)
                # Assuming hidden_dim is defined in self.hidden_dim
                # h_state = torch.zeros(self.hidden_dim, device=self.device)
                h_state = None
            else:
                h_state = h_state.to(self.device)
            
            # FIX 2: Correct variable name (plural)
            rnn_hidden_states.append(h_state)

            # 3. Retrieve Neighbor Messages
            neighbors_node_ids = node.last_memory['neighbors'] # Use .get for safety

            if len(neighbors_node_ids) > 0:
                # Assumption: This function returns AGGREGATED (1D) vectors (e.g., mean of neighbors)
                neighbor_solution, neighbor_analysis_embed = self.embed_messages_from_neighbors(neighbors_node_ids)

                # pdb.set_trace()

                sol_agree = (solution * neighbor_solution).sum().view(-1)
                ana_agree = (analysis_embed * neighbor_analysis_embed).sum().view(-1)

                sol_dev_norm = torch.norm(
                    solution - neighbor_solution,
                    p=1
                ).view(-1)   # scalar

                ana_dev_norm = torch.norm(
                    analysis_embed - neighbor_analysis_embed,
                    p=2
                ).view(-1)   # scalar

                has_neighbor = torch.tensor(1.0, device=self.device).view(-1)
            else:
                # Handle 0 neighbors (Zero Padding)
                neighbor_solution = torch.zeros(self.solution_dim, device=self.device)
                neighbor_analysis_embed = torch.zeros(self.analysis_output_dim, device=self.device)

                sol_agree = torch.tensor(0.0, device=self.device).view(-1)
                ana_agree = torch.tensor(0.0, device=self.device).view(-1)
                sol_dev_norm = torch.tensor(0.0, device=self.device).view(-1)
                ana_dev_norm = torch.tensor(0.0, device=self.device).view(-1)    

                has_neighbor = torch.tensor(0.0, device=self.device).view(-1)
            
            # pdb.set_trace()
            
            # 4. Concatenate all features into one vector per node
            # Shapes: [Sol] + [Ana] + [1] + [Sol] + [Ana]
            feature_vec = torch.cat([
                solution, 
                analysis_embed, 
                reliable_scores, 
                neighbor_solution, 
                neighbor_analysis_embed,
                sol_agree,
                ana_agree,
                sol_dev_norm,
                ana_dev_norm,
                has_neighbor
            ], dim=0)
            
            node_features.append(feature_vec)
        
        
        node_features = torch.stack(node_features, dim=0)
        rnn_hidden_states = self.stack_or_none(rnn_hidden_states)

        # pdb.set_trace() 

        # Stack to create [N, Total_Dim] and [N, Hidden_Dim]
        return node_features, rnn_hidden_states
    
    def stack_or_none(self, rnn_hidden_states):
        # If ANY element is None → return None
        if rnn_hidden_states is None:
            return None
        if any(h is None for h in rnn_hidden_states):
            return None
        
        # Otherwise, stack normally
        return torch.stack(rnn_hidden_states, dim=0)

    

    def embed_messages_from_neighbors(self, neighbors_node_ids: List[str]):

        neighbor_solutions = []
        neighbor_analysis_embeds = []
        neighbor_scores = []

        for neighbor_node_id in neighbors_node_ids:
            neighbor_node = self.find_node(neighbor_node_id)
            
            # 1. Retrieve and Flatten (Ensure 1D)
            sol = neighbor_node.last_memory['solution_embed'].view(-1)
            ana = neighbor_node.last_memory['analysis_embed'].view(-1)
            
            # Retrieve Score (Scalar or 1D tensor)
            score = neighbor_node.last_memory['reliable_scores'].view(-1)
            
            neighbor_solutions.append(sol)
            neighbor_analysis_embeds.append(ana)
            neighbor_scores.append(score)
        
        # 2. Stack Tensors -> Move to Device
        # Features: [N_neighbors, Dim]
        neighbor_solutions_tensor = torch.stack(neighbor_solutions, dim=0).to(self.device)
        neighbor_analysis_embeds_tensor = torch.stack(neighbor_analysis_embeds, dim=0).to(self.device)
        
        # Scores: [N_neighbors, 1] (Unsqueeze for broadcasting)
        raw_scores = torch.stack(neighbor_scores, dim=0).to(self.device)
        
        # 3. Normalize Weights (Important!)
        # We add epsilon to avoid division by zero if all scores are 0
        # epsilon = 1e-8
        # total_score = raw_scores.sum(dim=0) + epsilon
        
        # # weights shape: [N_neighbors, 1]
        # weights = raw_scores / total_score
        weights = torch.softmax(raw_scores, dim=0)
        
        # 4. Weighted Sum
        # Broadcast multiply: [N, D] * [N, 1] -> [N, D]
        # Then sum over neighbors (dim 0) -> [D]
        neighbor_solution = (neighbor_solutions_tensor * weights).sum(dim=0)
        neighbor_analysis_embed = (neighbor_analysis_embeds_tensor * weights).sum(dim=0)
        
        return neighbor_solution, neighbor_analysis_embed
    
    def construct_decision_edges(self, reliable_scores: torch.Tensor, threshold: float = None, training: bool = True):
        """
        Construct the decision edges of the nodes in the graph.
        """
        self.clear_decision_edges()

        weighted_answers = torch.zeros(self.solution_dim, device=self.device)

        log_probs = [torch.tensor(0.0, device=self.device, requires_grad=training)]  # Don't initialize with tensor(0.0)

        if threshold is not None:
            adaptive_threshold = torch.round(reliable_scores.mean(), decimals=4)

        for node_id, node in self.nodes.items():
            rs = reliable_scores[self.node_id_to_idx[node_id]]

            if threshold:
                edge_prob = torch.tensor(1.0 if rs > adaptive_threshold else 0.0, device=self.device)
            else:
                edge_prob = rs
            
            if torch.rand(1, device=self.device) < edge_prob:
                self.decision_node.add_predecessor(node)
                log_probs.append(torch.log(edge_prob))
                solution = self.prompt_set.postprocess_answer(node.outputs[-1].strip())
                solution_embed = self.prompt_set.change_solution_to_one_hot_vector(solution)
                solution_embed = solution_embed.to(self.device)
                # pdb.set_trace()
                weighted_answers += solution_embed * edge_prob
            else:
                log_probs.append(torch.log(1.0 - edge_prob))
        
        
        
        if weighted_answers.sum() == 0:
            log_probs, weighted_answers = self.get_weighted_majority_vote(reliable_scores, training=training)
        else:
            log_probs = torch.sum(torch.stack(log_probs))
        
        idx = torch.argmax(weighted_answers)
        answer = chr(idx.item() + ord('A'))

        # if threshold is not None:
        #     pdb.set_trace()
        
        return log_probs, answer
                
      
    
    def get_weighted_majority_vote(self, reliable_scores:torch.Tensor, training: bool = True):
        # Ensure scores sum to 1.0 (assuming reliable_scores are logits or raw values)
        agent_weights = torch.softmax(reliable_scores, dim=0) 

        # Initialize the aggregated vote vector (shape: solution_dim)
        weighted_answers = torch.zeros(self.solution_dim, device=self.device)

        log_probs = [torch.tensor(0.0, device=self.device, requires_grad=training)]

        for node_id, node in self.nodes.items():
            output = node.outputs[-1].strip()
            solution = self.prompt_set.postprocess_answer(output)
            solution_embed = self.prompt_set.change_solution_to_one_hot_vector(solution)
            solution_embed = solution_embed.to(self.device)
            node_idx = self.node_id_to_idx[node_id]
            agent_weight = agent_weights[node_idx]

            weighted_answers += solution_embed * agent_weight

            log_probs.append(torch.log(agent_weight))
        
        # idx = torch.argmax(weighted_answers)

        # # pdb.set_trace()
        
        # # Convert the index back to a character answer (e.g., 0->A, 1->B)
        # answer = chr(idx.item() + ord('A')) # Use .item() to convert tensor to standard Python int
        return torch.sum(torch.stack(log_probs)), weighted_answers
    
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
        comm_log_probs = []
        
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

            # if round > 0:
            comm_log_probs.append(round_log_probs)
            
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
        decision_log_probs, final_answers = self.construct_decision_edges(reliable_scores, threshold=threshold)
        
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers, torch.stack(comm_log_probs), decision_log_probs

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
                  training: bool = True,
                  options: List[str] = None,) -> List[Any]:
        # inputs:{'task':"xxx"}
        comm_log_probs = []
        
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
            
            
            print(f'-------------------------------- round {round} --------------------------------')
            
            if round == 0:
                reliable_scores = torch.full(size=(len(self.nodes),), fill_value=0.5, device=self.device)
                rnn_hidden_states = None
                # rnn_hidden_states = torch.zeros(len(self.nodes), self.rnn.hidden_dim, device=self.device)
            else:

                
                node_features, last_hidden_states = self.construct_node_features()

                # if round >= 1:
                #     pdb.set_trace()
                # pdb.set_trace()
                reliable_scores, rnn_hidden_states = self.rnn(node_features=node_features, hidden_state=last_hidden_states)
            
            # print(f"reliable_scores: {reliable_scores}")
            
            log_prob, adj = self.construct_comm_connection(reliable_scores=reliable_scores, threshold=threshold, training=training)

            # print(f"adj: {adj}")
            if round == num_rounds - 1:
                comm_log_probs.append(log_prob)
            
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

        print(f"reliable score for decision node:{reliable_scores}")
            
        # self.connect_decision_node()
        # await self.decision_node.async_execute(input)
        # final_answers = self.decision_node.outputs
        decision_log_probs, final_answers = self.construct_decision_edges(reliable_scores, threshold=threshold, training=training)
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        
        # pdb.set_trace()
        return final_answers, torch.stack(comm_log_probs), decision_log_probs
    
    def update_memory(self, input:Dict[str,str], reliable_scores:torch.Tensor, adj:torch.Tensor, rnn_hidden_states:torch.Tensor):
        for id, node in self.nodes.items():
            idx = self.node_id_to_idx[id]
            node.last_memory['reliable_scores'] = reliable_scores[idx].detach()
            node.last_memory['inputs'] = input['task']
            # pdb.set_trace()
            if len(node.outputs) == 0:
                print(f"Warning: Node {id} has no outputs, skipping memory update")
                pdb.set_trace()
                continue
            output = node.outputs[-1].strip()
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
            if rnn_hidden_states is not None:
                node.last_memory['rnn_hidden_states'] = rnn_hidden_states[idx]
            else:
                node.last_memory['rnn_hidden_states'] = None
    

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
    