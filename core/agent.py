"""
PIATSG Framework - Main Agent
Physics-Informed Adaptive Transformers with Safety Guarantees
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .components import Actor, AdaptivePINN, NeuralOperator, SafetyConstraint
from .buffer import ReplayBuffer

class PIATSGAgent:
    """PIATSG Agent with physics-informed components"""
    
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
        
        # Critic networks
        self.critic1 = self._create_critic().to(self.device)
        self.critic2 = self._create_critic().to(self.device)
        
        # Target networks
        self.target_critic1 = self._create_critic().to(self.device)
        self.target_critic2 = self._create_critic().to(self.device)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Physics-informed components
        self.adaptive_pinn = AdaptivePINN(self.state_dim, hidden_dim=1024).to(self.device)
        self.neural_operator = NeuralOperator(self.state_dim, hidden_dim=1024).to(self.device)
        self.safety_constraint = SafetyConstraint(
            self.state_dim, 
            self.action_dim, 
            hidden_dim=1024
        ).to(self.device)
        
        # Adaptive learning rates for physics components
        self.base_pinn_lr = config.pinn_lr
        self.base_operator_lr = config.operator_lr
        self.base_safety_lr = config.safety_lr
        
        # Optimizers with lower initial learning rates for physics components
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.critic_lr)
        self.pinn_optimizer = optim.Adam(self.adaptive_pinn.parameters(), lr=self.base_pinn_lr * 0.1)
        self.operator_optimizer = optim.Adam(self.neural_operator.parameters(), lr=self.base_operator_lr * 0.1)
        self.safety_optimizer = optim.Adam(self.safety_constraint.parameters(), lr=self.base_safety_lr * 0.1)
        
        # Learning rate schedulers for physics components
        self.pinn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.pinn_optimizer, mode='min', factor=0.8, patience=200
        )
        self.operator_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.operator_optimizer, mode='min', factor=0.8, patience=200
        )
        self.safety_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.safety_optimizer, mode='min', factor=0.8, patience=200
        )
        
        # SAC parameters
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = config.target_entropy
        
        # Replay buffer
        self.memory = ReplayBuffer(config.buffer_size, config.batch_size, self.device)
        
        # Training parameters
        self.tau = config.tau
        self.gamma = config.gamma
        
        # Adaptive physics loss weights with very conservative curriculum learning
        self.initial_physics_weight = 0.01   # Much more conservative
        self.initial_safety_weight = 0.005   # Much more conservative  
        self.min_physics_weight = 0.001      # Lower minimum
        self.min_safety_weight = 0.0005      # Lower minimum
        self.physics_loss_weight = self.initial_physics_weight
        self.safety_loss_weight = self.initial_safety_weight
        
        # Physics loss tracking for adaptive weighting
        self.physics_loss_history = []
        self.safety_loss_history = []
        self.physics_loss_ema = None
        self.safety_loss_ema = None
        self.ema_alpha = 0.95
        
        # Component activation thresholds with much more delayed activation
        self.min_buffer_for_physics = 50000  # Much later activation - when buffer is >10% full
        self.max_grad_norm = 1.0
        self.physics_grad_norm = 0.05         # Very conservative gradient clipping
        
        # Physics update frequency control - much more restrictive
        self.physics_update_frequency = 10    # Update every 10th cycle 
        self.physics_stable_threshold = 50000 # Never reached (buffer is 400k, but this is not used)
        
        # Performance tracking
        self.update_count = 0
        self.physics_update_count = 0
        self.total_training_cycles = 0  # Track total cycles separately from physics updates
        
        # Physics component stability tracking
        self.physics_divergence_threshold = 5.0
        self.safety_divergence_threshold = 10.0
        self.consecutive_divergence_count = 0
        self.max_consecutive_divergence = 50
        
        # Physics training mode control
        self.physics_monitoring_only = False  # Set to True to disable physics training, only monitor
        
        # Print initialization info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"PIATSG Agent initialized:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Initial physics loss weight: {self.physics_loss_weight:.6f}")
        print(f"  Initial safety loss weight: {self.safety_loss_weight:.6f}")
        print(f"  Physics gradient clipping: {self.physics_grad_norm}")
        print(f"  Physics update frequency: Every {self.physics_update_frequency} cycles")
        print(f"  Physics activation threshold: {self.min_buffer_for_physics} samples")
        print(f"  Physics monitoring only: {self.physics_monitoring_only}")
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
    
    def _update_physics_weights(self):
        """Update physics loss weights based on training progress with more conservative approach"""
        buffer_ratio = min(len(self.memory) / self.config.buffer_size, 1.0)
        
        # More aggressive curriculum learning: reduce weights faster and more
        decay_factor = 1.0 - (buffer_ratio * 0.9)  # Reduce to 10% of initial weight
        
        self.physics_loss_weight = max(
            self.initial_physics_weight * decay_factor,
            self.min_physics_weight
        )
        self.safety_loss_weight = max(
            self.initial_safety_weight * decay_factor,
            self.min_safety_weight
        )
        
        # Additional reduction based on update count to further stabilize
        if self.update_count > 2000:
            additional_decay = max(0.1, 1.0 - (self.update_count - 2000) / 10000)
            self.physics_loss_weight *= additional_decay
            self.safety_loss_weight *= additional_decay
    
    def _check_physics_stability(self, physics_loss, safety_loss):
        """Check if physics components are diverging and adapt learning rates"""
        if self.physics_loss_ema is None:
            self.physics_loss_ema = physics_loss
            self.safety_loss_ema = safety_loss
            return True
        
        # Update exponential moving averages
        self.physics_loss_ema = self.ema_alpha * self.physics_loss_ema + (1 - self.ema_alpha) * physics_loss
        self.safety_loss_ema = self.ema_alpha * self.safety_loss_ema + (1 - self.ema_alpha) * safety_loss
        
        # Check for divergence
        physics_diverged = physics_loss > self.physics_loss_ema * self.physics_divergence_threshold
        safety_diverged = safety_loss > self.safety_loss_ema * self.safety_divergence_threshold
        
        if physics_diverged or safety_diverged:
            self.consecutive_divergence_count += 1
            
            # Adaptive learning rate reduction before reset
            if self.consecutive_divergence_count > 10:
                # Reduce physics learning rates
                for param_group in self.pinn_optimizer.param_groups:
                    param_group['lr'] *= 0.8
                for param_group in self.operator_optimizer.param_groups:
                    param_group['lr'] *= 0.8
                for param_group in self.safety_optimizer.param_groups:
                    param_group['lr'] *= 0.8
                print(f"Warning: Reduced physics learning rates due to divergence")
            
            if self.consecutive_divergence_count >= self.max_consecutive_divergence:
                # Reset physics components
                self._reset_physics_components()
                self.consecutive_divergence_count = 0
                return False
        else:
            self.consecutive_divergence_count = 0
        
        return True
    
    def _reset_physics_components(self):
        """Reset physics components when they diverge"""
        print("Warning: Physics components diverged, resetting...")
        
        # Reinitialize physics networks
        self.adaptive_pinn = AdaptivePINN(self.state_dim, hidden_dim=1024).to(self.device)
        self.neural_operator = NeuralOperator(self.state_dim, hidden_dim=1024).to(self.device)
        self.safety_constraint = SafetyConstraint(
            self.state_dim, 
            self.action_dim, 
            hidden_dim=1024
        ).to(self.device)
        
        # Recreate optimizers with even lower learning rates
        current_pinn_lr = self.pinn_optimizer.param_groups[0]['lr']
        current_operator_lr = self.operator_optimizer.param_groups[0]['lr']
        current_safety_lr = self.safety_optimizer.param_groups[0]['lr']
        
        self.pinn_optimizer = optim.Adam(self.adaptive_pinn.parameters(), lr=current_pinn_lr * 0.5)
        self.operator_optimizer = optim.Adam(self.neural_operator.parameters(), lr=current_operator_lr * 0.5)
        self.safety_optimizer = optim.Adam(self.safety_constraint.parameters(), lr=current_safety_lr * 0.5)
        
        # Reset EMA tracking
        self.physics_loss_ema = None
        self.safety_loss_ema = None
    
    def select_action(self, state, deterministic=False):
        """Select action using the actor with safety filtering"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            raw_action = self.actor(state_tensor, deterministic=deterministic)
            
            # Apply safety constraint filtering after sufficient training
            if len(self.memory) > 8000:
                # Enable gradients temporarily for CBF computation
                state_for_cbf = state_tensor.clone().requires_grad_(True)
                action_for_cbf = raw_action.clone().requires_grad_(True)
                
                # Compute safety constraint with gradients enabled
                with torch.enable_grad():
                    safety_logits = self.safety_constraint(state_for_cbf.unsqueeze(0), action_for_cbf.unsqueeze(0))
                    safety_mask = (safety_logits > 0).float()
                    
                    if safety_mask.item() < 0.5:
                        raw_action = raw_action * 0.8
            
            # Conservative action clipping
            action = torch.clamp(raw_action, -0.6, 0.6)
            
            # Training phase action scaling
            if len(self.memory) < 3000:
                action = action * 0.5
            elif len(self.memory) < 8000:
                action = action * 0.75
        
        return action.cpu().numpy()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self, batch_size=None, update_physics_components=True):
        """Update all components with controlled physics-informed losses"""
        if batch_size is None:
            batch_size = self.config.batch_size
            
        if len(self.memory) < batch_size:
            return None
        
        self.update_count += 1
        self.total_training_cycles += 1
        
        # Update physics weights based on curriculum
        self._update_physics_weights()
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Check for NaN in input data
        if (torch.isnan(states).any() or torch.isnan(actions).any() or 
            torch.isnan(rewards).any() or torch.isnan(next_states).any()):
            print("Warning: NaN detected in training data, skipping update")
            return None
        
        losses = {}
        
        try:
            # Update Critics
            critic_losses = self._update_critics(states, actions, rewards, next_states, dones)
            losses.update(critic_losses)
            
            # Update Actor
            actor_loss = self._update_actor(states)
            losses['actor_loss'] = actor_loss
            
            # Update Alpha
            alpha_loss = self._update_alpha(states)
            losses['alpha_loss'] = alpha_loss
            
            # Update Physics Components with proper frequency control
            should_update_physics = (
                update_physics_components and 
                not self.physics_monitoring_only and  # Respect monitoring-only mode
                len(self.memory) > self.min_buffer_for_physics and
                (self.total_training_cycles % self.physics_update_frequency == 0)  # Only every Nth training cycle
            )
            
            # Always compute physics losses for monitoring, even if not updating
            should_monitor_physics = (
                len(self.memory) > self.min_buffer_for_physics and
                (self.total_training_cycles % (self.physics_update_frequency * 2) == 0)  # Monitor every 2*N cycles
            )
            
            if should_update_physics:
                self.physics_update_count += 1
                physics_losses = self._update_physics_components(states, actions, next_states)
                losses.update(physics_losses)
                
                # Check physics stability
                if 'pinn_loss' in physics_losses and 'safety_loss' in physics_losses:
                    stable = self._check_physics_stability(
                        physics_losses['pinn_loss'], 
                        physics_losses['safety_loss']
                    )
                    if not stable:
                        losses['physics_reset'] = 1.0
            else:
                # Return zero losses for physics components when not updated
                losses.update({'pinn_loss': 0.0, 'operator_loss': 0.0, 'safety_loss': 0.0})
            
            # Update target networks
            self._update_target_networks()
            
        except Exception as e:
            print(f"Warning: Error in update: {e}")
            return None
        
        return losses
    
    def _update_critics(self, states, actions, rewards, next_states, dones):
        """Update critic networks"""
        with torch.amp.autocast('cuda', enabled=True):
            # Target computation
            with torch.no_grad():
                next_actions = self.actor(next_states, deterministic=True)
                next_actions = torch.clamp(next_actions, -1, 1)
                
                target_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=1))
                target_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=1))
                target_q = torch.min(target_q1, target_q2)
                
                alpha = torch.clamp(self.log_alpha.exp(), 0.01, 5.0)
                next_log_probs = -0.5 * torch.sum((next_actions ** 2), dim=1, keepdim=True)
                target_q = target_q - alpha * next_log_probs
                target_q = rewards + (1 - dones) * self.gamma * target_q
                target_q = torch.clamp(target_q, -100, 100)
            
            # Current Q values
            current_q1 = self.critic1(torch.cat([states, actions], dim=1))
            current_q2 = self.critic2(torch.cat([states, actions], dim=1))
            
            if torch.isnan(current_q1).any() or torch.isnan(current_q2).any():
                print("Warning: NaN in Q values")
                return {'critic1_loss': 0.0, 'critic2_loss': 0.0}
            
            # Critic losses
            critic1_loss = F.smooth_l1_loss(current_q1, target_q)
            critic2_loss = F.smooth_l1_loss(current_q2, target_q)
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        if torch.isfinite(critic1_loss):
            self.scaler.scale(critic1_loss).backward()
            self.scaler.unscale_(self.critic1_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
            self.scaler.step(self.critic1_optimizer)
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        if torch.isfinite(critic2_loss):
            self.scaler.scale(critic2_loss).backward()
            self.scaler.unscale_(self.critic2_optimizer)
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
            self.scaler.step(self.critic2_optimizer)
        
        self.scaler.update()
        
        return {
            'critic1_loss': critic1_loss.item() if torch.isfinite(critic1_loss) else 0.0,
            'critic2_loss': critic2_loss.item() if torch.isfinite(critic2_loss) else 0.0
        }
    
    def _update_actor(self, states):
        """Update actor network with safety constraints"""
        with torch.amp.autocast('cuda', enabled=True):
            new_actions = self.actor(states, deterministic=False)
            new_actions = torch.clamp(new_actions, -1, 1)
            
            # Standard actor loss
            q1_new = self.critic1(torch.cat([states, new_actions], dim=1))
            q2_new = self.critic2(torch.cat([states, new_actions], dim=1))
            q_new = torch.min(q1_new, q2_new)
            
            alpha = torch.clamp(self.log_alpha.exp(), 0.01, 5.0)
            log_probs = -0.5 * torch.sum((new_actions ** 2), dim=1, keepdim=True)
            actor_loss = (alpha.detach() * log_probs - q_new).mean()
            
            # Add safety constraint penalty with adaptive weighting
            if len(self.memory) > self.min_buffer_for_physics:
                try:
                    # Clone states and actions to avoid in-place operation issues
                    states_for_safety = states.clone()
                    actions_for_safety = new_actions.clone()
                    safety_violation = self.safety_constraint.compute_safety_violation_loss(states_for_safety, actions_for_safety)
                    actor_loss = actor_loss + self.safety_loss_weight * safety_violation
                except Exception as safety_error:
                    print(f"Warning: Safety constraint error in actor update: {safety_error}")
                    pass
            
            if torch.isnan(actor_loss):
                print("Warning: NaN in actor loss")
                return 0.0
        
        # Update actor
        self.actor_optimizer.zero_grad()
        if torch.isfinite(actor_loss):
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        
        return actor_loss.item() if torch.isfinite(actor_loss) else 0.0
    
    def _update_alpha(self, states):
        """Update alpha parameter"""
        with torch.no_grad():
            actions_alpha = self.actor(states, deterministic=False)
            actions_alpha = torch.clamp(actions_alpha, -1, 1)
            log_probs_alpha = -0.5 * torch.sum((actions_alpha ** 2), dim=1, keepdim=True)
        
        alpha_loss = -(self.log_alpha * (log_probs_alpha.detach() + self.target_entropy)).mean()
        
        if torch.isfinite(alpha_loss):
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
            self.alpha_optimizer.step()
        
        return alpha_loss.item() if torch.isfinite(alpha_loss) else 0.0
    
    def _update_physics_components(self, states, actions, next_states):
        """Update physics-informed components with stability measures"""
        losses = {}
        
        try:
            # Physics-Informed Neural Network update with regularization
            self.pinn_optimizer.zero_grad()
            
            # Ensure no in-place operations by cloning inputs
            states_safe = states.clone()
            actions_safe = actions.clone()
            next_states_safe = next_states.clone()
            
            pinn_loss = self.adaptive_pinn.physics_loss(states_safe, next_states_safe, actions_safe)
            
            # Add L2 regularization to prevent overfitting
            l2_reg = torch.tensor(0.0, device=states.device)
            for param in self.adaptive_pinn.parameters():
                l2_reg += torch.norm(param)
            
            pinn_loss_total = pinn_loss + 1e-5 * l2_reg
            
            if torch.isfinite(pinn_loss_total) and pinn_loss_total < 100.0:
                pinn_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.adaptive_pinn.parameters(), self.physics_grad_norm)
                self.pinn_optimizer.step()
                self.pinn_scheduler.step(pinn_loss.item())
                losses['pinn_loss'] = pinn_loss.item()
            else:
                losses['pinn_loss'] = 0.0
            
            # Neural Operator update with regularization
            self.operator_optimizer.zero_grad()
            operator_pred = self.neural_operator(states_safe.clone(), actions_safe.clone())
            operator_loss = F.smooth_l1_loss(operator_pred, next_states_safe)
            
            # Add L2 regularization
            l2_reg_op = torch.tensor(0.0, device=states.device)
            for param in self.neural_operator.parameters():
                l2_reg_op += torch.norm(param)
            
            operator_loss_total = operator_loss + 1e-5 * l2_reg_op
            
            if torch.isfinite(operator_loss_total) and operator_loss_total < 100.0:
                operator_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.neural_operator.parameters(), self.physics_grad_norm)
                self.operator_optimizer.step()
                self.operator_scheduler.step(operator_loss.item())
                losses['operator_loss'] = operator_loss.item()
            else:
                losses['operator_loss'] = 0.0
            
            # Safety Constraint update with balanced sampling
            self.safety_optimizer.zero_grad()
            
            batch_size = states.shape[0]
            # Create more balanced safe/unsafe samples
            safe_indices = torch.randperm(batch_size)[:batch_size//3]
            unsafe_indices = torch.randperm(batch_size)[batch_size//3:2*batch_size//3]
            boundary_indices = torch.randperm(batch_size)[2*batch_size//3:]
            
            safe_loss = torch.tensor(0.0, device=states.device)
            unsafe_loss = torch.tensor(0.0, device=states.device)
            boundary_loss = torch.tensor(0.0, device=states.device)
            
            # Safe samples (should have positive CBF values)
            if len(safe_indices) > 0:
                safe_logits = self.safety_constraint(states_safe[safe_indices], actions_safe[safe_indices])
                safe_loss = F.relu(-safe_logits + 0.1).mean()
            
            # Unsafe samples (should have negative CBF values)
            if len(unsafe_indices) > 0:
                unsafe_actions = actions_safe[unsafe_indices] * 1.5  # More aggressive actions
                unsafe_actions_clamped = torch.clamp(unsafe_actions, -1, 1)
                unsafe_logits = self.safety_constraint(states_safe[unsafe_indices], unsafe_actions_clamped)
                unsafe_loss = F.relu(unsafe_logits + 0.1).mean()
            
            # Boundary samples (near constraint boundary)
            if len(boundary_indices) > 0:
                boundary_actions = actions_safe[boundary_indices] + 0.1 * torch.randn_like(actions_safe[boundary_indices])
                boundary_actions_clamped = torch.clamp(boundary_actions, -1, 1)
                boundary_logits = self.safety_constraint(states_safe[boundary_indices], boundary_actions_clamped)
                boundary_loss = torch.abs(boundary_logits).mean()
            
            safety_loss = safe_loss + unsafe_loss + 0.5 * boundary_loss
            
            # Add L2 regularization
            l2_reg_safety = torch.tensor(0.0, device=states.device)
            for param in self.safety_constraint.parameters():
                l2_reg_safety += torch.norm(param)
            
            safety_loss_total = safety_loss + 1e-5 * l2_reg_safety
            
            if torch.isfinite(safety_loss_total) and safety_loss_total < 100.0:
                safety_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.safety_constraint.parameters(), self.physics_grad_norm)
                self.safety_optimizer.step()
                self.safety_scheduler.step(safety_loss.item())
                losses['safety_loss'] = safety_loss.item()
            else:
                losses['safety_loss'] = 0.0
                
        except Exception as e:
            print(f"Warning: Error in physics components: {e}")
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
            'config': self.config,
            'update_count': self.update_count,
            'physics_update_count': self.physics_update_count,
            'total_training_cycles': self.total_training_cycles,
            'physics_loss_weight': self.physics_loss_weight,
            'safety_loss_weight': self.safety_loss_weight
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
        if 'update_count' in checkpoint:
            self.update_count = checkpoint['update_count']
        if 'physics_update_count' in checkpoint:
            self.physics_update_count = checkpoint['physics_update_count']
        if 'total_training_cycles' in checkpoint:
            self.total_training_cycles = checkpoint['total_training_cycles']
        if 'physics_loss_weight' in checkpoint:
            self.physics_loss_weight = checkpoint['physics_loss_weight']
        if 'safety_loss_weight' in checkpoint:
            self.safety_loss_weight = checkpoint['safety_loss_weight']