"""
PIATSG Framework Configuration and Utilities
Physics-Informed Adaptive Transformers with Safety Guarantees

Handles system configuration, device setup, and reproducible seeding.
"""

import os
import time
import random
import numpy as np
import torch

def set_reproducible_seed(seed=42):
    """Set seeds for consistent reproducible results across all libraries"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For maximum performance (slightly less deterministic but much faster)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"Reproducible seed set to {seed}")

def configure_device():
    """Configure and optimize device for training"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_vram:.1f} GB")
        
        # GPU optimizations for modern hardware
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')  # Use Tensor Cores aggressively
        
        # High VRAM usage (90% instead of 70%)
        available_vram = total_vram * 0.90
        
        # Large batch sizes for maximum GPU utilization
        optimal_batch_size = min(8192, max(4096, int(available_vram * 512)))
        optimal_buffer_size = min(1000000, max(200000, int(available_vram * 50000)))
        
        # Set memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        print(f"Performance optimizations enabled:")
        print(f"   - Mixed precision training: ENABLED")
        print(f"   - Large batch size: {optimal_batch_size}")
        print(f"   - Large buffer size: {optimal_buffer_size:,}")
        print(f"   - TensorFloat-32: ENABLED")
        print(f"   - CuDNN benchmark: ENABLED")
        print(f"   - Memory fraction: 95%")
        print(f"   - Tensor Core precision: HIGH")
        
        return device, optimal_batch_size, optimal_buffer_size, total_vram
    else:
        optimal_batch_size = 1024
        optimal_buffer_size = 100000
        return device, optimal_batch_size, optimal_buffer_size, 0.0

class TrainingConfig:
    """Training configuration parameters"""
    def __init__(self, device, batch_size, buffer_size):
        # Device configuration
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # Training parameters
        self.num_episodes = 5000
        self.max_steps_per_episode = 1000
        self.training_frequency = 500  # Train every N steps
        self.updates_per_training = 3   # Multiple updates per training session
        
        # Learning rates (conservative for mixed precision)
        self.actor_lr = 0.0001
        self.critic_lr = 0.0005
        self.pinn_lr = 0.0003
        self.operator_lr = 0.0003
        self.safety_lr = 0.0001
        self.alpha_lr = 0.0003
        
        # SAC parameters
        self.tau = 0.005
        self.gamma = 0.99
        self.target_entropy = -4  # -action_dim
        
        # Network dimensions
        self.state_dim = 18
        self.action_dim = 4
        self.hidden_dim = 1536
        
        # Evaluation parameters
        self.evaluation_episodes = 5
        self.evaluation_frequency = 100
        
        # Logging parameters
        self.log_frequency = 25
        self.checkpoint_frequency = 1000

def create_directories():
    """Create necessary directories for the framework"""
    directories = [
        'models',
        'logs', 
        'runs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Framework directories created")

def get_timestamp():
    """Get current timestamp for logging"""
    return time.strftime("%Y%m%d_%H%M%S")