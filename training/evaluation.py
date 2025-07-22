"""
PIATSG Framework - Agent Evaluation
Physics-Informed Adaptive Transformers with Safety Guarantees

Comprehensive evaluation metrics for PIATSG agent performance.
"""

import numpy as np
from simulation.environment import reset_simulation, get_observation, apply_action, step_simulation, check_done

def evaluate_agent(agent, episodes=5):
    """Evaluate agent performance with comprehensive metrics"""
    results = {
        'precisions_10cm': [],
        'precisions_5cm': [],
        'precisions_2cm': [],
        'physics_scores': [],
        'safety_scores': []
    }
    
    for ep in range(episodes):
        # Add slight randomization to avoid identical episodes
        reset_simulation(randomize=True)
        obs = get_observation()
        episode_length = 0
        max_episode_length = 1000
        
        positions = []
        velocities = []
        actions_taken = []
        
        for step in range(max_episode_length):
            action = agent.select_action(obs, deterministic=True)
            actions_taken.append(action.copy())
            apply_action(action, obs)
            step_simulation()
            obs = get_observation()
            
            positions.append(obs[0:3].copy())
            velocities.append(obs[7:10].copy())
            episode_length += 1
            
            if check_done(obs):
                break
        
        # Calculate comprehensive metrics
        positions = np.array(positions)
        velocities = np.array(velocities)
        actions_taken = np.array(actions_taken)
        target_pos = np.array([0.0, 0.0, 1.0])
        
        distances = np.linalg.norm(positions - target_pos, axis=1)
        
        # Precision metrics (percentage of time within distance thresholds)
        precision_10cm = np.mean(distances < 0.1) * 100
        precision_5cm = np.mean(distances < 0.05) * 100
        precision_2cm = np.mean(distances < 0.02) * 100
        
        # Physics score (smooth motion quality)
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        physics_score = np.mean(vel_magnitudes < 0.4) * 100
        
        # Safety score (no constraint violations)
        altitude_safe = np.all((positions[:, 2] > 0.3) & (positions[:, 2] < 1.7))
        position_safe = np.all((np.abs(positions[:, 0]) < 1.2) & (np.abs(positions[:, 1]) < 1.2))
        safety_score = 100.0 if altitude_safe and position_safe else 0.0
        
        results['precisions_10cm'].append(precision_10cm)
        results['precisions_5cm'].append(precision_5cm)
        results['precisions_2cm'].append(precision_2cm)
        results['physics_scores'].append(physics_score)
        results['safety_scores'].append(safety_score)
        
        # Detailed verification output
        final_pos = positions[-1]
        final_distance = np.linalg.norm(final_pos - target_pos)
        action_std = np.std(actions_taken, axis=0)
        action_mean = np.mean(actions_taken, axis=0)
        
        print(f"    Episode {ep} verification:")
        print(f"      Final position: {final_pos}")
        print(f"      Final distance: {final_distance:.3f}m")
        print(f"      Action stats: mean={action_mean}, std={action_std}")
        print(f"      Precision rates: 10cm={precision_10cm:.1f}%, "
              f"5cm={precision_5cm:.1f}%, 2cm={precision_2cm:.1f}%")
        print(f"      Physics: {physics_score:.1f}%, Safety: {safety_score:.1f}%")
    
    # Compute mean results
    results['mean_precision_10cm'] = np.mean(results['precisions_10cm'])
    results['mean_precision_5cm'] = np.mean(results['precisions_5cm']) 
    results['mean_precision_2cm'] = np.mean(results['precisions_2cm'])
    results['mean_physics_score'] = np.mean(results['physics_scores'])
    results['mean_safety_score'] = np.mean(results['safety_scores'])
    
    return results

def print_evaluation_summary(results):
    """Print a summary of evaluation results"""
    print(f"Evaluation Summary:")
    print(f"  Precision (10cm): {results['mean_precision_10cm']:.1f}%")
    print(f"  Super-precision (5cm): {results['mean_precision_5cm']:.1f}%")
    print(f"  Ultra-precision (2cm): {results['mean_precision_2cm']:.1f}%")
    print(f"  Physics score: {results['mean_physics_score']:.1f}%")
    print(f"  Safety score: {results['mean_safety_score']:.1f}%")

def evaluate_and_save_best(agent, episode, best_scores, checkpoint_dir):
    """Evaluate agent and save best models based on different metrics"""
    print(f"Evaluating at episode {episode}...")
    eval_results = evaluate_agent(agent, episodes=5)
    
    print_evaluation_summary(eval_results)
    
    # Track and save best performance
    model_saved = False
    
    if eval_results['mean_precision_10cm'] > best_scores['precision_10cm']:
        best_scores['precision_10cm'] = eval_results['mean_precision_10cm']
        agent.save_model(f'{checkpoint_dir}/best_precision_10cm.pth')
        print(f"New best precision (10cm): {best_scores['precision_10cm']:.1f}%")
        model_saved = True
    
    if eval_results['mean_precision_5cm'] > best_scores['precision_5cm']:
        best_scores['precision_5cm'] = eval_results['mean_precision_5cm']
        agent.save_model(f'{checkpoint_dir}/best_precision_5cm.pth')
        print(f"New best super-precision (5cm): {best_scores['precision_5cm']:.1f}%")
    
    if eval_results['mean_precision_2cm'] > best_scores['precision_2cm']:
        best_scores['precision_2cm'] = eval_results['mean_precision_2cm']
        agent.save_model(f'{checkpoint_dir}/best_precision_2cm.pth')
        print(f"New best ultra-precision (2cm): {best_scores['precision_2cm']:.1f}%")
    
    if eval_results['mean_physics_score'] > best_scores['physics']:
        best_scores['physics'] = eval_results['mean_physics_score']
        agent.save_model(f'{checkpoint_dir}/best_physics.pth')
        print(f"New best physics: {best_scores['physics']:.1f}%")
    
    if eval_results['mean_safety_score'] > best_scores['safety']:
        best_scores['safety'] = eval_results['mean_safety_score']
        agent.save_model(f'{checkpoint_dir}/best_safety.pth')
        print(f"New best safety: {best_scores['safety']:.1f}%")
    
    return eval_results, model_saved

def create_training_summary(episode, total_time, best_scores):
    """Create a comprehensive training summary"""
    print(f"\nTraining Summary:")
    print(f"=" * 60)
    print(f"Episodes completed: {episode}")
    print(f"Total training time: {total_time/3600:.1f} hours")
    print(f"Training speed: {episode/total_time:.2f} episodes/second")
    print(f"\nBest Performance Achieved:")
    print(f"  Precision (10cm): {best_scores['precision_10cm']:.1f}%")
    print(f"  Super-precision (5cm): {best_scores['precision_5cm']:.1f}%") 
    print(f"  Ultra-precision (2cm): {best_scores['precision_2cm']:.1f}%")
    print(f"  Physics score: {best_scores['physics']:.1f}%")
    print(f"  Safety score: {best_scores['safety']:.1f}%")
    print(f"=" * 60)