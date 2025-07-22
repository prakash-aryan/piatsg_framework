"""
PIATSG Framework - Main Training Loop
Physics-Informed Adaptive Transformers with Safety Guarantees

Optimized training loop with comprehensive logging and evaluation.
"""

import time
import torch
import numpy as np
from collections import deque
from tensorboardX import SummaryWriter

from core.agent import PIATSGAgent
from simulation.environment import (
    reset_simulation, get_observation, apply_action, 
    step_simulation, compute_reward, check_done
)
from training.evaluation import evaluate_and_save_best, create_training_summary

class PIATSGTrainer:
    """Main trainer for PIATSG agent"""
    
    def __init__(self, config):
        self.config = config
        self.agent = PIATSGAgent(config)
        
        # Training tracking
        self.best_scores = {
            'precision_10cm': 0.0,
            'precision_5cm': 0.0,
            'precision_2cm': 0.0,
            'physics': 0.0,
            'safety': 0.0
        }
        
        # Logging
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f'logs/piatsg_training_{timestamp}')
        self.reward_history = deque(maxlen=200)
        
        print("PIATSG Trainer initialized!")
        print("Training Optimizations:")
        print(f"- Large batch size: {config.batch_size}")
        print(f"- Training frequency: Every {config.training_frequency} steps")
        print(f"- Updates per session: {config.updates_per_training}")
        print(f"- Memory pinning: ENABLED")
        print(f"- Mixed precision: ENABLED")
        print(f"- Buffer size: {config.buffer_size:,}")
    
    def train(self):
        """Main training loop"""
        episode = 0
        training_start = time.time()
        
        print("PIATSG Training Started!")
        
        while episode < self.config.num_episodes:
            episode_reward, episode_steps = self._run_episode(episode)
            
            # Update tracking
            self.reward_history.append(episode_reward)
            avg_reward = np.mean(self.reward_history)
            
            # Logging
            self._log_episode(episode, episode_reward, episode_steps, avg_reward)
            
            # Progress reporting
            if episode % self.config.log_frequency == 0:
                self._report_progress(episode, episode_reward, episode_steps, 
                                    avg_reward, training_start)
            
            # Evaluation
            if episode > 0 and episode % self.config.evaluation_frequency == 0:
                evaluate_and_save_best(self.agent, episode, self.best_scores, 'models')
            
            # Periodic cleanup
            if episode % 1000 == 0:
                torch.cuda.empty_cache()
            
            episode += 1
        
        # Training complete
        training_end = time.time()
        total_time = training_end - training_start
        
        create_training_summary(episode, total_time, self.best_scores)
        self.writer.close()
        
        return self.agent, self.best_scores
    
    def _run_episode(self, episode):
        """Run a single training episode"""
        reset_simulation()
        obs = get_observation()
        episode_reward = 0.0
        episode_steps = 0
        
        # Reset actor history for new episode
        self.agent.actor.reset_dt_history()
        
        for step in range(self.config.max_steps_per_episode):
            # Select and apply action
            action = self.agent.select_action(obs, deterministic=False)
            apply_action(action, obs)
            step_simulation()
            
            # Get next observation and reward
            next_obs = get_observation()
            reward = compute_reward(next_obs)
            done = check_done(next_obs)
            
            # Store transition
            self.agent.store_transition(obs, action, reward, next_obs, done)
            self.agent.actor.update_dt_history(obs, action, reward)
            
            # Training updates
            if (len(self.agent.memory) > self.config.batch_size and 
                step % self.config.training_frequency == 0):
                
                # Multiple updates per training session for efficiency
                for update_idx in range(self.config.updates_per_training):
                    losses = self.agent.update(batch_size=self.config.batch_size)
                    
                    # Log losses (less frequently to avoid overhead)
                    if (losses and episode % 200 == 0 and 
                        step % (self.config.training_frequency * 2) == 0):
                        self._log_losses(losses, episode * self.config.max_steps_per_episode + step)
            
            # Update episode tracking
            episode_reward += reward
            episode_steps += 1
            obs = next_obs
            
            if done:
                break
        
        return episode_reward, episode_steps
    
    def _log_episode(self, episode, episode_reward, episode_steps, avg_reward):
        """Log episode metrics to tensorboard"""
        if episode % self.config.log_frequency == 0:
            self.writer.add_scalar('Training/Episode_Reward', episode_reward, episode)
            self.writer.add_scalar('Training/Average_Reward', avg_reward, episode)
            self.writer.add_scalar('Training/Episode_Length', episode_steps, episode)
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                self.writer.add_scalar('Training/GPU_Memory', gpu_memory, episode)
    
    def _log_losses(self, losses, step):
        """Log training losses to tensorboard"""
        for key, value in losses.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                self.writer.add_scalar(f'Loss/{key}', value, step)
    
    def _report_progress(self, episode, episode_reward, episode_steps, 
                        avg_reward, training_start):
        """Report training progress"""
        elapsed = time.time() - training_start
        eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
        eta_hours = ((self.config.num_episodes - episode) / 
                    (eps_per_sec * 3600) if eps_per_sec > 0 else 0)
        
        # Memory and performance stats
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            gpu_percent = (gpu_memory / 16.6) * 100  # Assuming 16.6GB total
        else:
            gpu_memory = 0.0
            gpu_percent = 0.0
        
        # Buffer and training status
        buffer_size = len(self.agent.memory)
        buffer_percent = (buffer_size / self.config.buffer_size) * 100
        
        # Training phase indicators
        if buffer_size < 10000:
            training_phase = "Exploration"
        elif buffer_size < 30000:
            training_phase = "Early Learning"
        elif buffer_size < 100000:
            training_phase = "Active Learning"
        else:
            training_phase = "Advanced Learning"
        
        # Recent reward trend
        recent_rewards = list(self.reward_history)[-10:] if len(self.reward_history) >= 10 else list(self.reward_history)
        recent_avg = np.mean(recent_rewards) if recent_rewards else avg_reward
        
        print(f"Episode {episode}, Reward: {episode_reward:.0f}, Steps: {episode_steps}, "
              f"Avg: {avg_reward:.0f}, Recent: {recent_avg:.0f}, "
              f"Speed: {eps_per_sec:.1f} eps/sec, ETA: {eta_hours:.1f}h, "
              f"GPU: {gpu_memory:.1f}GB ({gpu_percent:.1f}%), "
              f"Buffer: {buffer_size}/{self.config.buffer_size} ({buffer_percent:.1f}%), "
              f"Phase: {training_phase}")
        
        # Reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

def create_trainer(config):
    """Factory function to create PIATSG trainer"""
    return PIATSGTrainer(config)