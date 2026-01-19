import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class AgentRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_type='gru', temperature=0.5):
        """
        RNN to compute reliability scores for agents.
        Treats 'num_agents' as the batch dimension for parallel processing.
        
        Args:
            input_dim: Node feature dimension at each round.
            hidden_dim: RNN hidden state dimension.
            rnn_type: 'gru' or 'rnn' (LSTM requires tuple state handling).
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()
        self.temperature = temperature
        # --- The Feature Projector ---
        # self.feature_projector = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim * 2),
        #     nn.LayerNorm(hidden_dim * 2), 
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     nn.ReLU() 
        # )
        # pdb.set_trace()
        self.feature_projector = nn.Linear(input_dim, hidden_dim)

        
        # RNN for each agent
        if self.rnn_type == 'gru':
            self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNNCell(hidden_dim, hidden_dim)
        else:
            # If you need LSTM, init_hidden must return (h, c)
            raise ValueError(f"Supported RNN types are 'gru' or 'rnn'. Got: {rnn_type}")
            
        # Linear layer for reliability score
        # Project hidden state to a score [0, 1]
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # pdb.set_trace()
        
    # def init_hidden(self, num_agents, device):
    #     """Initialize hidden states as zeros."""
    #     # Returns shape: [num_agents, hidden_dim]
    #     return torch.zeros(num_agents, self.hidden_dim, device=device)
    
    def forward(self, raw_features, hidden_state=None):
        """
        Process one round.
        
        Args:
            raw_features: Tensor of shape [num_agents, input_dim]
            hidden_state: Tensor of shape [num_agents, hidden_dim] (or None for first step)
            
        Returns:
            scores: [num_agents]
            hidden_state: [num_agents, hidden_dim] (Updated)
        """
        # 1. Sanity Check for Dimensions
        if raw_features.dim() != 2 or raw_features.size(1) != self.input_dim:
            raise ValueError(f"Expected input shape [num_agents, {self.input_dim}], "
                             f"got {raw_features.shape}")

        num_agents = raw_features.size(0)
        device = raw_features.device

        node_features = self.feature_projector(raw_features)
        
        # 2. Initialize hidden if first round
        # if hidden_state is None:
        #     hidden_state = self.init_hidden(num_agents, device)
        
        # 3. RNN forward step
        # Note: RNNCell/GRUCell inputs are (input, hidden)
        hidden_state = self.rnn(node_features, hidden_state)
        
        # 4. Compute Scores
        # hidden_state shape: [num_agents, hidden_dim]
        # score_head output: [num_agents, 1] -> squeeze -> [num_agents]
        logits = self.score_head(hidden_state).squeeze(-1)

        # pdb.set_trace()

        # normalized_logits = self.zscore(logits)

        mean = logits.mean()
        std = logits.std()
        normalized_logits = (logits - mean) / (std + 1e-6)
        scores = torch.sigmoid(normalized_logits / self.temperature)
        # scores = torch.sigmoid(normalized_logits / self.temperature)
        
        return scores, hidden_state

    # def zscore(self, x, eps=1e-6):
    #     pdb.set_trace()
    #     mean = x.mean(dim=-1, keepdim=True)
    #     std  = x.std(dim=-1, keepdim=True, unbiased=False)
    #     return (x - mean) / (std + eps)



class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x