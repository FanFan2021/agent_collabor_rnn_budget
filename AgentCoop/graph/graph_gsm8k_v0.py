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
from collections import defaultdict
import torch.nn.functional as F

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
        # self.original_roles = {node_id: node.role for node_id, node in self.nodes.items()}

        num_nodes = len(self.nodes)
        self.node_id_to_idx = {node_id: idx for idx, node_id in enumerate(self.nodes.keys())}
        self.idx_to_node_id = {idx: node_id for idx, node_id in enumerate(self.nodes.keys())}
        self.comm_masks = torch.ones(num_nodes, num_nodes, device=device)
        self.comm_masks.fill_diagonal_(0.0)

        # Store dimensions as instance variables
        self.solution_dim = solution_dim
        # self.analysis_output_dim = analysis_output_dim
        # analysis_input_dim = sentence_transformer_dim_map[sentence_transformer]
        self.analysis_output_dim = sentence_transformer_dim_map[sentence_transformer]

        # self.analysis_proj = MLP(input_size=analysis_input_dim, hidden_size=analysis_output_dim, output_size=analysis_output_dim)

        # Feature includes: solution + analysis_embed + reliable_score + neighbor_solution + neighbor_analysis_embed + sol_agree + ana_agree + sol_dev_norm + ana_dev_norm + has_neighbor
        feature_dim = solution_dim + self.analysis_output_dim + 1 + solution_dim + self.analysis_output_dim + 4 + 1

        # pdb.set_trace()

        self.rnn = AgentRNN(input_dim=feature_dim, hidden_dim=hidden_dim, rnn_type='rnn')

        self.hidden_dim = hidden_dim

        self.sentence_transformer = SentenceTransformer(sentence_transformer)
        
        self.prompt_set = PromptSetRegistry.get(domain)

        self.device = device

        # self.analysis_proj.to(self.device)
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

    def construct_comm_connection(self, reliable_scores: torch.Tensor, epsilon: float = 0.1, training: bool = True): 
        """
        Constructs the communication graph based on agent reliability scores.
        
        Args:
            reliable_scores: [N] or [N, 1] Tensor of scores (0 to 1).
            epsilon: Probability of random exploration flip during training.
            training: Whether to use training mode.
            epsilon: Probability of random exploration flip during training.
        """
        self.clear_comm_connection()
        
        # Ensure 1D shape for distribution handling
        reliable_scores = reliable_scores.view(-1)
        sum_log_probs = torch.tensor(0.0, device=self.device)
        
        # --- LOGIC BRANCHING --
        # MODE 1: TRAINING (Stochastic + Exploration)
        if training:
            # 1. Stability Clamp
            # Prevents log(0) which causes NaNs during backprop if probability is 0.0 or 1.0
            probs = torch.clamp(reliable_scores, min=1e-6, max=1.0 - 1e-6)
            
            # 2. Policy Distribution
            dist = torch.distributions.Bernoulli(probs=probs)
            
            # 3. Initial Sample (What the model wants to do)
            policy_samples = dist.sample()
            
            # 4. Apply Epsilon-Greedy Exploration
            # We create a noise mask where 1 = "flip the decision"
            noise_dist = torch.distributions.Bernoulli(
                probs=torch.full_like(policy_samples, epsilon)
            )
            noise_mask = noise_dist.sample()
            
            # XOR Logic: Flip policy if noise is 1 (0->1, 1->0)
            # abs(a - b) acts as XOR for 0/1 floats
            exec_samples = torch.abs(policy_samples - noise_mask)

            # 5. Compute Log Probs on EXECUTED Samples
            # CRITICAL: We calculate prob of the *executed* action under the *original* policy.
            # This allows REINFORCE to learn "I should have done what the noise forced me to do"
            # if it yields a high reward.
            action_log_probs = dist.log_prob(exec_samples)
            sum_log_probs = action_log_probs.sum()
        else:
            cutoff = torch.round(reliable_scores.mean(), decimals=4)
            
            # Create binary mask (1.0 or 0.0)
            exec_samples = (reliable_scores > cutoff).float().to(self.device)

        # --- GRAPH CONSTRUCTION ---
        # Optimization: Move tensor to CPU lists once to avoid 
        # hundreds of .item() calls inside the nested loop (slows down GPU sync)
        exec_samples_cpu = exec_samples.detach().cpu().numpy()
        
        # We sort to prioritize high-reliability nodes adding edges first.
        # This is a heuristic to make the DAG consistent with reliability.
        sorted_scores, sorted_idx = torch.sort(reliable_scores, descending=True)
        sorted_idx_cpu = sorted_idx.cpu().numpy()
        
        adj = torch.zeros(len(self.nodes), len(self.nodes), device=self.device)

        # Loop over indices using standard Python integers (Fast)
        for i in range(len(sorted_idx_cpu)):
            sender_idx = sorted_idx_cpu[i]
            
            # If this node is "Off", it cannot send messages
            if exec_samples_cpu[sender_idx] == 0:
                continue
            
            sender_id = self.idx_to_node_id[sender_idx]
            sender = self.find_node(sender_id)

            for j in range(len(sorted_idx_cpu)):
                receiver_idx = sorted_idx_cpu[j]
                
                # If receiver node is "Off", it cannot receive messages
                if exec_samples_cpu[receiver_idx] == 0:
                    continue
                
                receiver_id = self.idx_to_node_id[receiver_idx]
                receiver = self.find_node(receiver_id)

                # No self-loops 
                if sender_idx == receiver_idx:
                    continue
                
                # Check spatial/predefined constraints
                if self.comm_masks[receiver_idx, sender_idx] == 0.0:
                    continue
                
                # Check for cycle (DAG enforcement)
                # Since we iterate sorted by score, high-score nodes tend to be parents.
                if self.check_cycle(receiver, {sender}):
                    continue
                
                # Update Node State and Adjacency Matrix
                sender.add_successor(receiver) 
                adj[receiver_idx, sender_idx] = 1.0

        return sum_log_probs, adj, exec_samples


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

                diff = abs(solution - neighbor_solution)
                normalization = max(abs(solution), abs(neighbor_solution), 1e-8)
                sol_agree = 1 - (diff / normalization)

                # ana_agree = (analysis_embed * neighbor_analysis_embed).sum().view(-1)
                cos_sim = F.cosine_similarity(analysis_embed, neighbor_analysis_embed, dim=0).view(-1)
                ana_agree = (cos_sim + 1) / 2

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
        # pdb.set_trace()
        
        # 4. Weighted Sum
        # Broadcast multiply: [N, D] * [N, 1] -> [N, D]
        # Then sum over neighbors (dim 0) -> [D]
        neighbor_solution = (neighbor_solutions_tensor * weights).sum(dim=0)
        neighbor_analysis_embed = (neighbor_analysis_embeds_tensor * weights).sum(dim=0)
        
        return neighbor_solution, neighbor_analysis_embed


    def construct_decision_edges(self, reliable_scores: torch.Tensor, samples: torch.Tensor, training: bool = True, temperature: float = 0.5):
        """
        Args:
            temperature: Float. Controls exploration. Pass in a decaying value (e.g., 1.0 -> 0.1) during training.
        """
        # 1. Clean Inputs
        reliable_scores = reliable_scores.view(-1).to(self.device)
        # Ensure samples is strictly binary for the logic check
        active_mask = (samples.view(-1) > 0.5).to(self.device) 

        weighted_answers = torch.zeros(self.solution_dim, device=self.device)
        valid_options_mask = torch.zeros(self.solution_dim, device=self.device)

        # 2. Aggregation Loop
        # NOTE: If self.nodes is large, consider vectorizing the text-processing outside this method 
        # to speed up the loop. For <20 agents, this loop is fine.
        for node_id, node in self.nodes.items():
            idx = self.node_id_to_idx[node_id]

            # --- FILTER: CONSISTENCY CHECK ---
            if not active_mask[idx]:
                continue

            # Get Agent's Vote
            output = node.outputs[-1].strip()
            solution = self.prompt_set.postprocess_answer(output) 
            
            # Convert to One-Hot
            # detach() is implied here since it's created from string, but good to be explicit 
            # that gradients stop here for the agent.
            solution_embed = self.prompt_set.change_solution_to_one_hot_vector(solution).to(self.device)
            
            # Weight the vote by Reliability Score
            # Gradient flows through 'reliable_scores[idx]'
            rs = reliable_scores[idx]
            weighted_answers = weighted_answers + (solution_embed * rs)
            
            # Track valid options (Logical OR)
            valid_options_mask = torch.max(valid_options_mask, solution_embed)

        # 3. Handle Edge Case: No valid votes
        if valid_options_mask.sum().item() == 0:
            # --- ATTEMPT 2: Soft Fallback (Recover Gradients) ---
            # If we are here, it means the 'samples' (Comm Graph) silenced everyone useful.
            # Let's ignore 'samples' and see if we can get a vote from the full population.
            
            # Recalculate weighted_answers using ALL nodes (ignoring samples mask)
            weighted_answers = torch.zeros(self.solution_dim, device=self.device)
            valid_options_mask = torch.zeros(self.solution_dim, device=self.device)
            
            for node_id, node in self.nodes.items():
                idx = self.node_id_to_idx[node_id]
                # Note: We removed the "if samples[idx] == 0" check here
                
                output = node.outputs[-1].strip()
                solution = self.prompt_set.postprocess_answer(output) 
                solution_embed = self.prompt_set.change_solution_to_one_hot_vector(solution).to(self.device)
                
                rs = reliable_scores[idx]
                weighted_answers = weighted_answers + (solution_embed * rs)
                valid_options_mask = torch.max(valid_options_mask, solution_embed)
            
            # --- ATTEMPT 3: Hard Fallback (Safety Net) ---
            # If it is STILL zero, it means every single agent outputted garbage (invalid text).
            # Now we truly have no choice but to kill the gradient.
            if valid_options_mask.sum().item() == 0:
                 # Pick the agent with the highest reliability score just to output SOMETHING
                best_idx = torch.argmax(reliable_scores).item()
                max_node_id = self.idx_to_node_id[best_idx]
                max_node = self.find_node(max_node_id)
                fallback_ans = self.prompt_set.postprocess_answer(max_node.outputs[-1].strip())
                
                return fallback_ans, torch.tensor(0.0, device=self.device)

        # 4. Make the Decision
        if training:
            # Scale by temperature
            logits = weighted_answers / temperature
            
            # Mask invalid options with -inf
            # We use a clone to avoid modifying the original logits tensor in place if needed elsewhere
            masked_logits = logits.clone()
            masked_logits[valid_options_mask == 0] = float('-inf')
            
            # Create Distribution
            dist = torch.distributions.Categorical(logits=masked_logits)
            
            # Sample
            answer_idx = dist.sample()
            
            # Compute Log Prob for RL (Gradients flow back to reliable_scores)
            decision_log_prob = dist.log_prob(answer_idx)
            
        else:
            # Inference: Deterministic Greedy
            # We mask here too, just in case negative weights exist (unlikely but safe)
            weighted_answers[valid_options_mask == 0] = float('-inf')
            answer_idx = torch.argmax(weighted_answers)
            decision_log_prob = torch.tensor(0.0, device=self.device)

        # 5. Convert Index back to Character
        final_answer = chr(answer_idx.item() + ord('A'))
        
        return final_answer, decision_log_prob      
    
    # def get_weighted_majority_vote(self, reliable_scores:torch.Tensor, samples: torch.Tensor, solutions: List[int]):
    #     """
    #     Args:
    #         reliable_scores: torch.Tensor [num_agents] - Raw reliability logits/scores.
    #         samples: torch.Tensor [num_agents] - Binary mask (1 if edge exists, 0 otherwise).
    #     """
    #     # pdb.set_trace()
    
    #     # 2. Apply Softmax
    #     # Agents with -inf will have exactly 0.0 probability
    #     agent_weights = F.softmax(reliable_scores, dim=-1)
    #     agent_weights = agent_weights.masked_fill(samples == 0, 0.0)

    #         # unique_sols: the actual values; inverse_indices: mapping of original to unique
    #     unique_sols, inverse_indices = torch.unique(torch.tensor(solutions), return_inverse=True)
    #     inverse_indices = inverse_indices.to(agent_weights.device)
        
    #     # 2. Aggregate weights for each unique solution
    #     # Create a buffer for the summed weights
    #     summed_weights = torch.zeros(len(unique_sols), device=agent_weights.device)
        
    #     # Use scatter_add to sum weights based on the solution index
    #     summed_weights.scatter_add_(0, inverse_indices, agent_weights)
        
    #     # 3. Find the solution with the maximum weight
    #     best_idx = torch.argmax(summed_weights)
    #     best_solution = unique_sols[best_idx].item()

    #     print(f"summed_weights: {summed_weights}, unique_sols: {unique_sols}")

    #     return best_solution

    def get_weighted_majority_vote(self, reliable_scores:torch.Tensor, samples: torch.Tensor):
        """
        Args:
            reliable_scores: torch.Tensor [num_agents] - Raw reliability logits/scores.
            samples: torch.Tensor [num_agents] - Binary mask (1 if edge exists, 0 otherwise).
        """
        # 1. Normalize scores into weights that sum to 1.0
        agent_weights = torch.softmax(reliable_scores, dim=0) 

        

        # 2. Dictionary to aggregate weights for unique numeric answers
        # Key: Integer Answer, Value: Cumulative Weight (Tensor for grad tracking)
        weighted_votes = defaultdict(lambda: torch.tensor(0.0, device=self.device))

        # We track if at least one agent is active to avoid a total failure
        active_agents_count = 0

        solutions = []

        for node_id, node in self.nodes.items():
            idx = self.node_id_to_idx[node_id]
            
            # GATE: Only agents with an active edge (samples[idx] == 1) participate
            if samples[idx].item() == 0:
                continue
            output = node.outputs[-1].strip()
            analysis, solution_line = self.extract_analysis_and_output(output)
            solution = self.prompt_set.postprocess_answer(solution_line)
            # solution = int(float(solution))
            try:
                solution = int(float(solution))
            except:
                # pdb.set_trace()
            #     print(f"node_id: {node_id}, output: {output}, solution: {solution}")
            #     solution = 0
            # solution = self.safe_embed_solution(solution)
            # solutions.append(solution.item())
            # if solution is not None:
            #     node_idx = self.node_id_to_idx[node_id]
            #     agent_weight = agent_weights[node_idx]
            #     weighted_votes[solution.item()] =  weighted_votes[solution.item()] + agent_weight
            # else:
                pdb.set_trace()
                print(f"node_id: {node_id}, solution: {solution}")

        # 3. Decision Logic
        if not weighted_votes:
            # Fallback: If no edges were formed or no numbers found, return a dummy
            return 0

        # pdb.set_trace()
        # Find the answer with the highest cumulative weight
        # We use a loop here to keep the winner as a dictionary key but track its weight
        print(f"solutions for deicsion: {solutions}")
        print(f"reliable scores for deicsion: {reliable_scores}")
        print(f"samples for deicsion: {samples}")
        final_answer = max(weighted_votes, key=lambda k: weighted_votes[k].item())

        # pdb.set_trace()

        return final_answer
      
    
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
        

        if attack_mode == "fixed":
            num_to_change = int(math.floor(m * attack_rate))
            if num_to_change <= 0:
                return []
            selected_agents = potential_change_agents[-num_to_change:]
            
        elif attack_mode == "random" or attack_mode == "half":
            # CHANGE: Randomly pick 1, 2, or 3
            target_num = random.choice([1, 2, 3, 4])
            
            # Safety: Ensure we don't try to pick more agents than exist
            num_to_change = min(target_num, m)
            if num_to_change <= 0:
                return []
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
    
    # def recover_agents_roles(self):
    #     # pdb.set_trace()
    #     # Restore the original nodes from the copy made during initialization
    #     for node_id, node in self.nodes.items():
    #         node.role = self.original_roles[node_id]
    def apply_consensus_attack(self, agent_indices: List[int], wrong_answer: str):
        """Sets the chosen agents to output the consensus wrong answer."""
        for idx in agent_indices:
            node_id = self.idx_to_node_id[idx]
            node = self.nodes[node_id]
            node.role = "Fake"
            self.adversarial_answers[node_id] = wrong_answer
            print(f"[LOG] Round Event: Agent {node.id} turned adversarial (Target: {wrong_answer})")

    def run(self, input: Any, 
                  num_rounds:int = 3, 
                  attack_rounds:int = 2,
                  attack_rate:float = 0.5,
                  attack_mode:str = 'fixed',
                #   threshold:float = None,
                  training: bool = True,
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
            if round == attack_rounds and correct_answer is not None:
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
            
            round_log_probs, adj = self.construct_comm_connection(reliable_scores=reliable_scores, training=training)

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
                #   threshold:float = None,
                  epsilon: float = 0.1,
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

        num_nodes = len(self.nodes)

        # 1. Pre-calculate the Consensus Lie (the same wrong answer for all)
        if self.domain == "mmlu":
            wrong_options = [opt for opt in options if opt != correct_answer]
            global_wrong_answer = random.choice(wrong_options)
        elif self.domain == "gsm8k":
            
            correct_int = int(float(correct_answer))
          
            
            # Strategy 1: Off-by-one or small constant error
            # Strategy 2: Off-by-scale (common if a 'doubling' or 'halving' step was missed)
            # Strategy 3: Digit flip or random nearby integer
            
            choices = [
                correct_int + random.choice([-1, 1, 2, 5]), # Simple arithmetic slip
                correct_int * 2,                            # Missed a division step
                correct_int // 2 if correct_int > 1 else 3, # Missed a multiplication step
                correct_int + 10,                           # Decimal alignment error
                abs(correct_int - random.randint(1, 10))    # General calculation error
            ]
            
            # Filter out the correct answer if it was accidentally generated
            wrong_options = [str(val) for val in choices if val != correct_int and val >= 0]
            
            # Final safety check: if list is empty, just offset the correct answer
            global_wrong_answer = random.choice(wrong_options) if wrong_options else str(correct_int + 1)
     
        # wrong_options = [opt for opt in options if opt != correct_answer]
        # global_wrong_answer = random.choice(wrong_options)
        
        # 2. Setup the Flip Schedule based on Mode
        flip_schedule = {} # agent_index -> round_to_flip
        
        if attack_mode == "fixed":
            # FIXED: Fixed number of agents flip at a Fixed round
            num_adv = int(num_nodes * attack_rate)
            adv_indices = list(range(num_nodes))[-num_adv:]
            for idx in adv_indices:
                flip_schedule[idx] = attack_rounds
                
        elif attack_mode == "random":
            # RANDOM: Random number of agents flip at different Random rounds
            # RANDOM: A random number of agents (between 2 and CAP) flip at random rounds
            # Here, attack_rate sets the upper bound
            max_adv = max(2, int(num_nodes * attack_rate)) 
            num_adv = random.randint(2, max_adv)
            adv_indices = random.sample(range(num_nodes), num_adv)
            for idx in adv_indices:
                flip_schedule[idx] = random.randint(2, max(2, num_rounds - 1))
        else:
            raise ValueError("attack_mode must be 'fixed' or 'random'")
        
        # pdb.set_trace()

                
        # 3. Main Loop
        flipped_agents = set()
        for round in range(num_rounds+1):
            if training:
                print(f'-------------------------------- train round {round} --------------------------------')
            else:
                print(f'-------------------------------- eval round {round} --------------------------------')
            
            # Identify who flips THIS round
            to_flip = [idx for idx, r in flip_schedule.items() 
                    if r == round and idx not in flipped_agents]
            
            if to_flip:
                self.apply_consensus_attack(to_flip, global_wrong_answer)
                flipped_agents.update(to_flip)
            
            # pdb.set_trace()
         
            
            if round == 0:
                reliable_scores = torch.full(size=(num_nodes,), fill_value=0.5, device=self.device)
                rnn_hidden_states = None
                # rnn_hidden_states = torch.zeros(len(self.nodes), self.rnn.hidden_dim, device=self.device)
            else:
                raw_features, last_hidden_states = self.construct_node_features()
                reliable_scores, rnn_hidden_states = self.rnn(raw_features=raw_features, hidden_state=last_hidden_states)
            
            # print(f"reliable_scores: {reliable_scores}")
            
            log_prob, adj, samples = self.construct_comm_connection(reliable_scores=reliable_scores, epsilon=epsilon, training=training)
            # training=training)

            # print(f"adj: {adj}")
            if round > 0:
                comm_log_probs.append(log_prob)
            
            if round < num_rounds:
            
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
            
            solutions = []
            for node_id, node in self.nodes.items():
                try:
                    analysis, solution_line = self.extract_analysis_and_output(node.outputs[-1].strip())
                    solution = self.prompt_set.postprocess_answer(solution_line.strip())
                    solution = self.safe_embed_solution(solution)
                    solutions.append(solution.item())
                    # print(f"node_id: {node_id}, output: {node.outputs[-1].strip()}")
                except:
                    pdb.set_trace()
                    print(f"node_id: {node_id}, output: {node.outputs}")
            
            # pdb.set_trace()
            # if round == num_rounds:
            #     pdb.set_trace()
            roles = [node.role for node in self.nodes.values()]

            print(f"attack_rounds: {attack_rounds}")
            print(f"attack_rate: {attack_rate}")
            if attack_mode == "random":
                print(f"max num of adv agents: {max_adv}")
            else:
                print(f"num of adv agents: {num_adv}")
            print(f"flip_schedule: {flip_schedule}")
           
            print(f"solutions: {solutions}, correct_answer: {correct_answer}")
            print(f"roles: {roles}")
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
        # final_answers, decision_log_prob = self.construct_decision_edges(reliable_scores, samples=samples, training=training)
        final_answers = self.get_weighted_majority_vote(reliable_scores, samples=samples)
        print(f"final_answers: {final_answers}, correct_answer: {correct_answer}")
        if final_answers is None:
            final_answers=int(0)
        
        # self.recover_agents_roles()
        
        # pdb.set_trace()
        return final_answers, torch.stack(comm_log_probs)
    
    def update_memory(self, input:Dict[str,str], reliable_scores:torch.Tensor, adj:torch.Tensor, rnn_hidden_states:torch.Tensor):
        for id, node in self.nodes.items():
            idx = self.node_id_to_idx[id]
            node.last_memory['reliable_scores'] = reliable_scores[idx].detach()
            if self.domain == "mmlu":
                node.last_memory['inputs'] = input['task']
            else: # gsm8k
                node.last_memory['inputs'] = input['question']
            # pdb.set_trace()
            if len(node.outputs) == 0:
                print(f"Warning: Node {id} has no outputs, skipping memory update")
                pdb.set_trace()
                continue
            output = node.outputs[-1].strip()
            node.last_memory['outputs'] = output
            # solution = self.prompt_set.postprocess_answer(output)
            analysis, solution_line = self.extract_analysis_and_output(output)
            # pdb.set_trace()
            solution = self.prompt_set.postprocess_answer(solution_line)
            node.last_memory['solution'] = solution
            try:
                solution = int(float(solution))
            except:
                # pdb.set_trace()
                print(f"solution: {solution}")
                solution = 0
            # pdb.set_trace()
            solution_embed = self.safe_embed_solution(solution)
            node.last_memory['solution_embed'] = solution_embed
            analysis_embedding = torch.tensor(self.sentence_transformer.encode(analysis), device=self.device)
            # node.last_memory['analysis_embed'] = self.analysis_proj(analysis_embedding).detach()
            node.last_memory['analysis_embed'] = analysis_embedding
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
    
    def safe_embed_solution(self, solution):
        # Limits for 32-bit integer (torch.int32)
        MAX_INT32 = 2147483647
        MIN_INT32 = -2147483648
        
        try:
            # 1. Ensure solution is a valid number
            if solution is None:
                return torch.tensor([0], dtype=torch.int32, device=self.device)
            
            # 2. Convert to int and check bounds
            val = int(solution)
            if MIN_INT32 <= val <= MAX_INT32:
                return torch.tensor([val], dtype=torch.int32, device=self.device)
            else:
                # If it's too big, it's likely a hallucination; cap it or default to 0
                print(f"Warning: Solution {val} overflows int32. Defaulting to 0.")
                return torch.tensor([0], dtype=torch.int32, device=self.device)
                
        except (OverflowError, RuntimeError, ValueError):
            # Catch non-integer strings or massive Python-only integers
            return torch.tensor([0], dtype=torch.int32, device=self.device)

    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False
    
    def extract_analysis_and_output(self, text: str):
        # 1. Standardize line breaks
        text = text.strip()
        
        # 2. Define the delimiter
        delimiter = "The answer is"
        
        if delimiter in text:
            # Split into [analysis, final_line]
            parts = text.split(delimiter)
            analysis = parts[0].strip()
            solution_line = parts[1].strip()
            
            
        else:
            # Fallback if the delimiter is missing
            analysis = text
            solution_line = text
            
        return analysis, solution_line

def min_max_norm(tensor:torch.Tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_0_to_1 = (tensor - min_val) / (max_val - min_val)
    normalized_minus1_to_1 = normalized_0_to_1 * 2 - 1
    return normalized_minus1_to_1
    