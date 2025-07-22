"""
PIATSG Framework - Neural Network Components
Physics-Informed Adaptive Transformers with Safety Guarantees

Core neural network components: Actor, AdaptivePINN, NeuralOperator, SafetyConstraint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import numpy as np

class Actor(nn.Module):
    """Hierarchical actor with Decision Transformer for precision control"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=1536, max_history=50):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_history = max_history
        
        # Decision transformer components
        self.embedding_dim = 256
        self.dt_embed = nn.Linear(state_dim + action_dim + 1, self.embedding_dim)
        self.dt_pos_embed = nn.Parameter(torch.zeros(1, max_history, self.embedding_dim))
        
        # Multi-head attention for temporal dependencies
        self.dt_attention = nn.MultiheadAttention(self.embedding_dim, 8, batch_first=True)
        self.dt_norm = nn.LayerNorm(self.embedding_dim)
        
        # Hierarchical control networks
        self.high_level_policy = nn.Sequential(
            nn.Linear(state_dim + self.embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        self.precision_refiner = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        # Log std for stochastic policy - Start with lower variance
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)
        
        # History storage
        self.history_states = deque(maxlen=max_history)
        self.history_actions = deque(maxlen=max_history)
        self.history_rewards = deque(maxlen=max_history)
        
        # Initialize network weights for stable initial policy
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize actor network weights for stable initial policy"""
        for module in [self.high_level_policy, self.precision_refiner]:
            if hasattr(module, '__iter__'):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        if layer == module[-1]:  # Final layer
                            nn.init.uniform_(layer.weight, -0.003, 0.003)
                            nn.init.zeros_(layer.bias)
                        else:
                            nn.init.xavier_uniform_(layer.weight)
                            nn.init.zeros_(layer.bias)
    
    def reset_dt_history(self):
        """Reset decision transformer history"""
        self.history_states.clear()
        self.history_actions.clear()
        self.history_rewards.clear()
    
    def update_dt_history(self, state, action, reward=0.0):
        """Update decision transformer history"""
        self.history_states.append(state.copy())
        self.history_actions.append(action.copy())
        self.history_rewards.append(reward)
    
    def forward(self, state, deterministic=False):
        batch_size = state.shape[0] if len(state.shape) > 1 else 1
        
        # Decision transformer processing
        dt_context = torch.zeros(batch_size, self.embedding_dim).to(state.device)
        
        if len(self.history_states) > 0:
            # Create sequence from history
            seq_len = min(len(self.history_states), self.max_history)
            
            # Build input sequence
            seq_states = torch.zeros(batch_size, seq_len, self.state_dim).to(state.device)
            seq_actions = torch.zeros(batch_size, seq_len, self.action_dim).to(state.device)
            seq_rewards = torch.zeros(batch_size, seq_len, 1).to(state.device)
            
            # Fill sequence with recent history
            for i, (h_state, h_action, h_reward) in enumerate(zip(
                list(self.history_states)[-seq_len:],
                list(self.history_actions)[-seq_len:],
                list(self.history_rewards)[-seq_len:]
            )):
                seq_states[:, i] = torch.FloatTensor(h_state).to(state.device)
                seq_actions[:, i] = torch.FloatTensor(h_action).to(state.device)
                seq_rewards[:, i, 0] = h_reward
            
            # Transformer embedding
            seq_input = torch.cat([seq_states, seq_actions, seq_rewards], dim=-1)
            embedded = self.dt_embed(seq_input)
            embedded += self.dt_pos_embed[:, :seq_len]
            
            # Self-attention
            attended, _ = self.dt_attention(embedded, embedded, embedded)
            attended = self.dt_norm(attended)
            
            # Extract context from last timestep
            dt_context = attended[:, -1]
        
        # Ensure state has correct batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Hierarchical policy
        combined_input = torch.cat([state, dt_context], dim=-1)
        high_level_action = self.high_level_policy(combined_input)
        
        # Precision refinement
        refiner_input = torch.cat([state, high_level_action], dim=-1)
        action_refinement = self.precision_refiner(refiner_input)
        
        # Final action with numerical stability
        mean = high_level_action + 0.1 * action_refinement
        mean = torch.clamp(mean, -10, 10)
        
        log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, -10, 2)
        std = torch.exp(log_std)
        std = torch.clamp(std, 1e-4, 10)
        
        # Handle NaN/Inf
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.zeros_like(mean)
        if torch.isnan(std).any() or torch.isinf(std).any():
            std = torch.ones_like(std) * 0.1
        
        if deterministic:
            return torch.clamp(mean.squeeze(0) if batch_size == 1 else mean, -1, 1)
        else:
            normal = Normal(mean, std)
            action = normal.rsample()
            action = torch.clamp(action, -1, 1)
            return action.squeeze(0) if batch_size == 1 else action


class AdaptivePINN(nn.Module):
    """Self-Adaptive Physics-Informed Neural Network for dynamics learning"""
    
    def __init__(self, state_dim, hidden_dim=1024):
        super().__init__()
        
        # Physics network
        self.physics_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, state_dim)
        )
        
        # Adaptive weights
        self.physics_weight = nn.Parameter(torch.ones(1))
        self.data_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, state):
        return self.physics_net(state)
    
    def physics_loss(self, state, next_state, action):
        """Physics-informed loss with conservation constraints"""
        predicted_next = self.forward(state)
        physics_loss = F.mse_loss(predicted_next, next_state)
        
        # Simple physics constraints
        pos = state[:, :3]
        pred_pos = predicted_next[:, :3]
        
        # Gravity constraint (z-acceleration should be negative without thrust)
        gravity_loss = F.relu(pred_pos[:, 2] - pos[:, 2] - 0.1).mean()
        
        return self.physics_weight * physics_loss + 0.1 * gravity_loss


class NeuralOperator(nn.Module):
    """Physics-Informed Neural Operator for multi-scale dynamics prediction"""
    
    def __init__(self, state_dim, hidden_dim=1024):
        super().__init__()
        
        self.operator_net = nn.Sequential(
            nn.Linear(state_dim + 4, hidden_dim),  # state + action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, state_dim)
        )
        
    def forward(self, state, action):
        combined = torch.cat([state, action], dim=-1)
        return self.operator_net(combined)


class SafetyConstraint(nn.Module):
    """Safety constraint network with Control Barrier Function principles"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=1024):
        super().__init__()
        
        self.safety_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Return logits
        )
        
    def forward(self, state, action):
        combined = torch.cat([state, action], dim=-1)
        return self.safety_net(combined)