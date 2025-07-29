#!/usr/bin/env python3
"""
PPO Training Script for Diabetes Treatment Environment

This script trains a Proximal Policy Optimization (PPO) model on the
diabetes treatment environment and saves it for later use.
"""

import os
import sys
import time
import torch

# Add environment directory to path (similar to main.py approach)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'environment'))

from stable_baselines3 import PPO
from custom_env import DiabetesTreatmentEnv

def main():
    """Train PPO model on diabetes treatment environment."""
    print("PPO Training for Diabetes Treatment Environment")
    print("=" * 50)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")
    
    # Create environment
    env = DiabetesTreatmentEnv()

    
    # Create PPO model with optimized hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=2.5e-4,      # Slightly lower for more stable learning
        n_steps=1024,              # Reduced for better sample efficiency
        batch_size=32,             # Smaller batch size for better gradient updates
        n_epochs=8,                # Reduced from 10 to prevent overfitting
        gamma=0.995,               # Higher discount factor for long-term planning
        gae_lambda=0.98,           # Higher for better advantage estimation
        clip_range=0.15,           # Slightly more conservative clipping
        clip_range_vf=None,        # Let it adapt automatically
        normalize_advantage=True,   # Keep normalization for stability
        ent_coef=0.01,             # Small entropy bonus to encourage exploration
        vf_coef=0.25,              # Reduced value function coefficient
        max_grad_norm=0.5,         # Keep gradient clipping for stability
        target_kl=0.01,            # Add KL divergence constraint
        verbose=1,
        device=device,
        policy_kwargs=dict(
            net_arch=[64, 64],     # Smaller network for this simple environment
            activation_fn=torch.nn.Tanh  # Better for continuous control-like problems
        )
    )
    
    # Training parameters
    total_timesteps = 80000
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print("=" * 50)
    
    # Train the model
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time/60:.1f} minutes")
    
    # Save the model
    models_dir = "models/ppo"
    os.makedirs(models_dir, exist_ok=True)
    model_path = f"{models_dir}/ppo_model"
    model.save(model_path)
    print(f"Model saved as: {model_path}.zip")
    
    # Quick test
    print("Testing model...")
    obs, _ = env.reset()
    for i in range(3):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        print(f"Test {i+1}: Action={action}, Sugar={env.sugar_level:.1f}")
        if done or truncated:
            obs, _ = env.reset()
    
    env.close()
    print("PPO training complete!")

if __name__ == "__main__":
    main()
