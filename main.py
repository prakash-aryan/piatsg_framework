#!/usr/bin/env python3
"""
PIATSG Framework - Main Execution Script
Physics-Informed Adaptive Transformers with Safety Guarantees

Main entry point for training and evaluation of PIATSG agents.
"""

import argparse
import threading
import time

from utils.config import (
    set_reproducible_seed, configure_device, TrainingConfig, 
    create_directories, get_timestamp
)
from simulation.environment import initialize_simulation, launch_viewer, cleanup_simulation
from training.trainer import create_trainer

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='PIATSG Framework Training')
    parser.add_argument('--episodes', type=int, default=5000, 
                       help='Number of training episodes (default: 5000)')
    parser.add_argument('--no-viewer', action='store_true',
                       help='Run without MuJoCo viewer for faster training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override default batch size')
    parser.add_argument('--buffer-size', type=int, default=None,
                       help='Override default buffer size')
    
    args = parser.parse_args()
    
    try:
        print("=" * 80)
        print("PIATSG Framework - Physics-Informed Adaptive Transformers with Safety Guarantees")
        print("=" * 80)
        
        # Initialize framework
        set_reproducible_seed(args.seed)
        device, batch_size, buffer_size, total_vram = configure_device()
        create_directories()
        
        # Override sizes if specified
        if args.batch_size is not None:
            batch_size = args.batch_size
        if args.buffer_size is not None:
            buffer_size = args.buffer_size
            
        # Create training configuration
        config = TrainingConfig(device, batch_size, buffer_size)
        config.num_episodes = args.episodes
        
        print(f"\nTraining Configuration:")
        print(f"  Episodes: {config.num_episodes}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Buffer size: {config.buffer_size:,}")
        print(f"  Device: {config.device}")
        print(f"  Seed: {args.seed}")
        
        # Initialize simulation
        print(f"\nInitializing MuJoCo simulation...")
        initialize_simulation()
        print("MuJoCo simulation initialized successfully!")
        
        # Create trainer
        trainer = create_trainer(config)
        
        # Start training
        if args.no_viewer:
            print(f"\nStarting training without viewer...")
            agent, best_scores = trainer.train()
        else:
            print(f"\nStarting training with MuJoCo viewer...")
            # Launch training in separate thread
            training_thread = threading.Thread(target=lambda: trainer.train(), daemon=True)
            training_thread.start()
            
            # Launch viewer (blocks until viewer is closed)
            try:
                launch_viewer()
            except KeyboardInterrupt:
                print("\nViewer interrupted by user")
            
            # Wait for training to complete
            training_thread.join(timeout=10.0)
        
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        cleanup_simulation()
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        cleanup_simulation()
    finally:
        cleanup_simulation()
        print("Framework shutdown complete.")

if __name__ == "__main__":
    main()