#!/usr/bin/env python3
"""
DQN Training Script for Diabetes Treatment Environment

This script trains a Deep Q-Network (DQN) agent to play the diabetes treatment
environment using the custom_env.py and rendering.py files.
"""

import sys
import os
import numpy as np
import time

# Add environment directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'environment'))

# Import required libraries
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    print("Stable Baselines3 imported successfully")
except ImportError as e:
    print(f"Error importing Stable Baselines3: {e}")
    print("Install with: pip install stable-baselines3[extra]")
    sys.exit(1)

# Import custom environment
try:
    from environment.custom_env import DiabetesTreatmentEnv
    print("Custom environment imported successfully")
except ImportError as e:
    print(f"Error importing custom environment: {e}")
    print("Make sure custom_env.py is in the environment/ directory")
    sys.exit(1)

def create_training_environment():
    """Create and validate the training environment."""
    print("Creating training environment...")
    
    # Create environment
    env = DiabetesTreatmentEnv()
    
    print(f"Environment created!")
    print(f"Action space: {env.action_space} (0=Up, 1=Down, 2=Left, 3=Right)")
    print(f"Observation space: {env.observation_space}")
    
    # Check environment compatibility with Stable Baselines3
    print("Checking environment compatibility...")
    try:
        check_env(env)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        return None
    
    return env

def setup_directories():
    """Create necessary directories for training."""
    directories = [
        "models",
        "models/dqn", 
        "logs",
        "logs/dqn"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created: {directory}")

def train_dqn_model():
    """Train DQN model on the diabetes treatment environment."""
    print("=" * 60)
    print("DQN TRAINING FOR DIABETES TREATMENT ENVIRONMENT")
    print("=" * 60)
    
    # Setup directories
    setup_directories()
    
    # Create training environment
    env = create_training_environment()
    if env is None:
        return None, None
    
    # Wrap environment with Monitor for logging
    env = Monitor(env, "logs/dqn/training")
    
    # Create evaluation environment
    eval_env = DiabetesTreatmentEnv()
    eval_env = Monitor(eval_env, "logs/dqn/evaluation")
    
    print("\nConfiguring DQN model...")
    
    # DQN configuration optimized for this environment
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,      # Standard learning rate
        buffer_size=100000,      # Large replay buffer
        batch_size=32,           # Standard batch size
        gamma=0.99,              # High discount factor for long-term planning
        train_freq=4,            # Train every 4 steps
        target_update_interval=10000,  # Update target network every 10k steps
        exploration_fraction=0.2,      # 20% of training for exploration
        exploration_initial_eps=1.0,   # Start with full exploration
        exploration_final_eps=0.02,    # End with 2% exploration
        policy_kwargs=dict(net_arch=[64, 64]),  # Simple network architecture
        verbose=1,               # Show training progress
        device="auto",           # Use GPU if available
        seed=42                  # For reproducible results
    )
    
    print("DQN model configured!")
    print(f" - Network architecture: [64, 64]")
    print(f" - Learning rate: {model.learning_rate}")
    print(f" - Buffer size: {model.buffer_size}")
    print(f" - Exploration: {model.exploration_initial_eps} → {model.exploration_final_eps}")
    print(f" - Training frequency: {model.train_freq} steps")
    # Create callbacks for evaluation and checkpointing
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/dqn/",
        log_path="logs/dqn/",
        eval_freq=5000,          # Evaluate every 5000 steps
        deterministic=True,
        render=False,
        n_eval_episodes=5,       # Run 5 episodes for evaluation
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/dqn/checkpoints/",
        name_prefix="dqn_diabetes"
    )
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print("Training parameters:")
    print(f"  - Total timesteps: 100,000")
    print(f"  - Evaluation frequency: Every 5,000 steps")
    print(f"  - Checkpoint frequency: Every 10,000 steps")
    print(f"  - Actions: 0=Up, 1=Down, 2=Left, 3=Right")
    print("=" * 60)
    
    # Start training
    start_time = time.time()
    
    model.learn(
        total_timesteps=100000,  # 100k timesteps for good learning
        callback=[eval_callback, checkpoint_callback],
        log_interval=100,        # Log every 100 timesteps
        progress_bar=True        # Show progress bar
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds!")
    
    # Save final model
    final_model_path = "models/dqn/dqn_diabetes_final"
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}.zip")
    
    return model, env

def evaluate_trained_model(model_path="models/dqn/dqn_diabetes_final"):
    """Evaluate the trained model."""
    print("\n" + "=" * 60)
    print("EVALUATING TRAINED MODEL")
    print("=" * 60)
    
    # Load the trained model
    try:
        model = DQN.load(model_path)
        print(f"Model loaded from: {model_path}.zip")
    except Exception as e:
        print(f"Could not load model: {e}")
        return
    
    # Create test environment
    env = DiabetesTreatmentEnv()
    
    print("\nRunning evaluation episodes...")
    
    # Run multiple test episodes
    total_rewards = []
    episode_lengths = []
    
    for episode in range(5):
        print(f"\nEpisode {episode + 1}/5:")
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(1000):  # Max steps to prevent infinite episodes
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Print progress every 50 steps
            if step % 50 == 0:
                print(f"  Step {step}: Sugar level = {env.sugar_level:.1f} mg/dL")
            
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Final reward: {episode_reward:.2f}")
        print(f"  Episode length: {episode_length} steps")
        print(f"  Final sugar level: {env.sugar_level:.1f} mg/dL")
    
    # Calculate statistics
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_length = np.mean(episode_lengths)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average episode length: {avg_length:.1f} steps")
    print(f"Reward range: {min(total_rewards):.2f} to {max(total_rewards):.2f}")
    
    env.close()

def demo_visual_test(model_path="models/dqn/dqn_diabetes_final"):
    """Run a visual demonstration of the trained model."""
    print("\n" + "=" * 60)
    print("VISUAL DEMONSTRATION")
    print("=" * 60)
    print("This will show the trained agent playing the environment visually.")
    print("Close the pygame window to stop the demonstration.")
    
    # Load the trained model
    try:
        model = DQN.load(model_path)
        print(f"Model loaded from: {model_path}.zip")
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Make sure to train the model first!")
        return
    
    # Create environment with rendering
    env = DiabetesTreatmentEnv()
    
    try:
        obs, _ = env.reset()
        print("\nStarting visual demonstration...")
        print("Watch the agent navigate and make treatment decisions!")
        
        for step in range(200):  # Run for 200 steps max
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action
            obs, reward, done, truncated, info = env.step(action)
            
            # Render the environment
            env.render()
            
            # Print occasional updates
            if step % 25 == 0:
                action_names = ["Up", "Down", "Left", "Right"]
                print(f"Step {step}: Action={action_names[action]}, "
                      f"Sugar={env.sugar_level:.1f} mg/dL, "
                      f"Time={env.simulation_time:.1f}h")
            
            # Small delay for smooth animation
            time.sleep(0.1)
            
            if done or truncated:
                print("Episode completed!")
                break
        
        print("Visual demonstration completed!")
        
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        env.close()

def main():
    """Main training and evaluation pipeline."""
    print("DQN Diabetes Treatment Training Pipeline")
    print("=" * 60)
    
    # Check if we need to train or just evaluate
    model_exists = os.path.exists("models/dqn/dqn_diabetes_final.zip")
    
    if model_exists:
        print("Trained model found!")
        choice = input("Choose: (t)rain new model, (e)valuate existing, or (d)emo visual? [t/e/d]: ").lower()
    else:
        print("No trained model found. Will train a new model.")
        choice = 't'
    
    if choice == 't' or not model_exists:
        # Train the model
        print("\nStarting training process...")
        model, env = train_dqn_model()
        
        if model is not None:
            print("\nTraining successful! Running evaluation...")
            evaluate_trained_model()
            
            # Ask if user wants to see visual demo
            demo_choice = input("\nShow visual demonstration? [y/n]: ").lower()
            if demo_choice == 'y':
                demo_visual_test()
        else:
            print("✗ Training failed!")
            
    elif choice == 'e':
        # Just evaluate existing model
        evaluate_trained_model()
        
    elif choice == 'd':
        # Just run visual demo
        demo_visual_test()
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print("Files created:")
    print("  - models/dqn/dqn_diabetes_final.zip (trained model)")
    print("  - models/dqn/best_model.zip (best model during training)")
    print("  - logs/dqn/ (training and evaluation logs)")
    print("\nTo run visual tests later: python dqn_training.py")

if __name__ == "__main__":
    main()
