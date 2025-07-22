"""
PIATSG Framework - Main Agent
Physics-Informed Adaptive Transformers with Safety Guarantees

Main agent integrating all components for UAV control.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .components import Actor, AdaptivePINN, NeuralOperator, SafetyConstraint
from .buffer import ReplayBuffer

class PIATSGAgent:
    """PIATSG Agent with integrated physics-informed components"""
    
    def __init__(self, config):
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.device = config.device
        self.config = config
        
        # Initialize mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda')
        
        # PIATSG components
        self.actor = Actor(
            self.state_dim, 
            self.action_dim, 
            hidden_dim=config.hidden_dim
        ).to(self.device)
        
        # Optimized critic networks
        self.critic1 = self._create_critic().to(self.device)
        self.critic2 = self._create_critic().to(self.device)
        
        # Target networks
        self.target_critic1 = self._create_critic().to(self.device)
        self.target_critic2 = self._create_critic().to(self.device)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # PIATSG-specific components
        self.adaptive_pinn = AdaptivePINN(self.state_dim, hidden_dim=1024).to(self.device)
        self.neural_operator = NeuralOperator(self.state_dim, hidden_dim=1024).to(self.device)
        self.safety_constraint = SafetyConstraint(
            self.state_dim, 
            self.action_dim, 
            hidden_dim=1024
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.critic_lr)
        self.pinn_optimizer = optim.Adam(self.adaptive_pinn.parameters(), lr=config.pinn_lr)
        self.operator_optimizer = optim.Adam(self.neural_operator.parameters(), lr=config.operator_lr)
        self.safety_optimizer = optim.Adam(self.safety_constraint.parameters(), lr=config.safety_lr)
        
        # SAC parameters
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = config.target_entropy
        
        # Replay buffer
        self.memory = ReplayBuffer(config.buffer_size, config.batch_size, self.device)
        
        # Training parameters
        self.tau = config.tau
        self.gamma = config.gamma
        
        # Print initialization info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"PIATSG Agent initialized:")
        print(f"  Total parameters: {total_params:,}")
        if torch.cuda.is_available():
            print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    def _create_critic(self):
        """Create critic network architecture"""
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def parameters(self):
        """Get all parameters for counting"""
        params = []
        params.extend(self.actor.parameters())
        params.extend(self.critic1.parameters())
        params.extend(self.critic2.parameters())
        params.extend(self.adaptive_pinn.parameters())
        params.extend(self.neural_operator.parameters())
        params.extend(self.safety_constraint.parameters())
        return params
    
    def select_action(self, state, deterministic=False):
        """Select action using the actor with training-phase handling"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor, deterministic=deterministic)
            
            # Conservative action clipping for stability
            action = torch.clamp(action, -0.5, 0.5)
            
            # Training phase action scaling
            if len(self.memory) < 5000:  # Early training
                action = action * 0.3
            elif len(self.memory) < 15000:  # Medium training  
                action = action * 0.6
            
        return action.cpu().numpy()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self, batch_size=None):
        """Update all components with separate forward passes"""
        if batch_size is None:
            batch_size = self.config.batch_size
            
        if len(self.memory) < batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Check for NaN in input data
        if (torch.isnan(states).any() or torch.isnan(actions).any() or 
            torch.isnan(rewards).any() or torch.isnan(next_states).any()):
            print("Warning: NaN detected in training data, skipping update")
            return None
        
        losses = {}
        
        # Update Critics
        losses.update(self._update_critics(states, actions, rewards, next_states, dones))
        
        # Update Actor
        actor_loss = self._update_actor(states)
        losses['actor_loss'] = actor_loss
        
        # Update Alpha
        alpha_loss = self._update_alpha(states)
        losses['alpha_loss'] = alpha_loss
        
        # Update PIATSG Components
        piatsg_losses = self._update_piatsg_components(states, actions, next_states)
        losses.update(piatsg_losses)
        
        # Update target networks
        self._update_target_networks()
        
        return losses
    
    def _update_critics(self, states, actions, rewards, next_states, dones):
        """Update critic networks"""
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            # Target computation
            with torch.no_grad():
                next_actions = self.actor(next_states, deterministic=True)
                next_actions = torch.clamp(next_actions, -1, 1)
                
                target_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=1))
                target_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=1))
                target_q = torch.min(target_q1, target_q2)
                
                alpha = torch.clamp(self.log_alpha.exp(), 0.001, 10.0)
                next_log_probs = -0.5 * torch.sum((next_actions ** 2), dim=1, keepdim=True)
                target_q = target_q - alpha * next_log_probs
                target_q = rewards + (1 - dones) * self.gamma * target_q
                target_q = torch.clamp(target_q, -1000, 1000)
            
            # Current Q values
            current_q1 = self.critic1(torch.cat([states, actions], dim=1))
            current_q2 = self.critic2(torch.cat([states, actions], dim=1))
            
            if torch.isnan(current_q1).any() or torch.isnan(current_q2).any():
                print("Warning: NaN in Q values, skipping update")
                return {'critic1_loss': 0.0, 'critic2_loss': 0.0}
            
            # Critic losses
            critic1_loss = F.smooth_l1_loss(current_q1, target_q)
            critic2_loss = F.smooth_l1_loss(current_q2, target_q)
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        self.scaler.scale(critic1_loss).backward()
        self.scaler.unscale_(self.critic1_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.scaler.step(self.critic1_optimizer)
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        self.scaler.scale(critic2_loss).backward()
        self.scaler.unscale_(self.critic2_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.scaler.step(self.critic2_optimizer)
        
        self.scaler.update()
        
        return {
            'critic1_loss': critic1_loss.item() if isinstance(critic1_loss, torch.Tensor) else 0.0,
            'critic2_loss': critic2_loss.item() if isinstance(critic2_loss, torch.Tensor) else 0.0
        }
    
    def _update_actor(self, states):
        """Update actor network"""
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            new_actions = self.actor(states, deterministic=False)
            new_actions = torch.clamp(new_actions, -1, 1)
            
            q1_new = self.critic1(torch.cat([states, new_actions], dim=1))
            q2_new = self.critic2(torch.cat([states, new_actions], dim=1))
            q_new = torch.min(q1_new, q2_new)
            
            alpha = torch.clamp(self.log_alpha.exp(), 0.001, 10.0)
            log_probs = -0.5 * torch.sum((new_actions ** 2), dim=1, keepdim=True)
            actor_loss = (alpha.detach() * log_probs - q_new).mean()
            
            if torch.isnan(actor_loss):
                print("Warning: NaN in actor loss, skipping update")
                return 0.0
        
        # Update actor
        self.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()
        
        return actor_loss.item() if isinstance(actor_loss, torch.Tensor) else 0.0
    
    def _update_alpha(self, states):
        """Update alpha parameter"""
        with torch.no_grad():
            actions_alpha = self.actor(states, deterministic=False)
            actions_alpha = torch.clamp(actions_alpha, -1, 1)
            log_probs_alpha = -0.5 * torch.sum((actions_alpha ** 2), dim=1, keepdim=True)
        
        alpha_loss = -(self.log_alpha * (log_probs_alpha.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], 1.0)
        self.alpha_optimizer.step()
        
        return alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else 0.0
    
    def _update_piatsg_components(self, states, actions, next_states):
        """Update PIATSG-specific components"""
        losses = {}
        
        try:
            # PINN update
            self.pinn_optimizer.zero_grad()
            pinn_loss = self.adaptive_pinn.physics_loss(
                states.detach(), next_states.detach(), actions.detach()
            )
            if not torch.isnan(pinn_loss):
                pinn_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.adaptive_pinn.parameters(), 1.0)
                self.pinn_optimizer.step()
                losses['pinn_loss'] = pinn_loss.item()
            else:
                losses['pinn_loss'] = 0.0
            
            # Neural operator update
            self.operator_optimizer.zero_grad()
            operator_pred = self.neural_operator(states.detach(), actions.detach())
            operator_loss = F.smooth_l1_loss(operator_pred, next_states.detach())
            if not torch.isnan(operator_loss):
                operator_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.neural_operator.parameters(), 1.0)
                self.operator_optimizer.step()
                losses['operator_loss'] = operator_loss.item()
            else:
                losses['operator_loss'] = 0.0
            
            # Safety update
            self.safety_optimizer.zero_grad()
            safety_logits = self.safety_constraint(states.detach(), actions.detach())
            safety_loss = F.binary_cross_entropy_with_logits(
                safety_logits, torch.ones_like(safety_logits)
            )
            if not torch.isnan(safety_loss):
                safety_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.safety_constraint.parameters(), 1.0)
                self.safety_optimizer.step()
                losses['safety_loss'] = safety_loss.item()
            else:
                losses['safety_loss'] = 0.0
                
        except Exception as e:
            print(f"Warning: Error in PIATSG components: {e}")
            losses = {'pinn_loss': 0.0, 'operator_loss': 0.0, 'safety_loss': 0.0}
        
        return losses
    
    def _update_target_networks(self):
        """Update target networks using soft updates"""
        with torch.no_grad():
            for tp1, p1, tp2, p2 in zip(
                self.target_critic1.parameters(), self.critic1.parameters(),
                self.target_critic2.parameters(), self.critic2.parameters()
            ):
                tp1.data.mul_(1 - self.tau).add_(p1.data, alpha=self.tau)
                tp2.data.mul_(1 - self.tau).add_(p2.data, alpha=self.tau)
    
    def save_model(self, filepath):
        """Save all model components"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'pinn': self.adaptive_pinn.state_dict(),
            'operator': self.neural_operator.state_dict(),
            'safety': self.safety_constraint.state_dict(),
            'log_alpha': self.log_alpha,
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath):
        """Load all model components"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.adaptive_pinn.load_state_dict(checkpoint['pinn'])
        self.neural_operator.load_state_dict(checkpoint['operator'])
        self.safety_constraint.load_state_dict(checkpoint['safety'])
        self.log_alpha = checkpoint['log_alpha']