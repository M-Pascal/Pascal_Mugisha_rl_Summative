#!/usr/bin/env python3
"""
A2C Training Script for Diabetes Treatment Environment

This script trains an Advantage Actor-Critic (A2C) model on the
diabetes treatment environment and saves it for later use.
"""

import os
import sys
import time
import torch

# Add environment directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'environment'))

from stable_baselines3 import A2C
from custom_env import DiabetesTreatmentEnv

def main():
    """Train A2C model on diabetes treatment environment."""
    print("A2C Training for Diabetes Treatment Environment")
    print("=" * 50)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")
    
    # Create environment
    env = DiabetesTreatmentEnv()

    # Create A2C model with optimized hyperparameters
    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=7e-4,        # Good for A2C
        gamma=0.99,                # Standard discount factor
        gae_lambda=1.0,            # No GAE for A2C (use TD estimates)
        n_steps=5,                 # Steps per update (smaller for A2C)
        vf_coef=0.25,              # Value function coefficient
        ent_coef=0.01,             # Entropy coefficient for exploration
        max_grad_norm=0.5,         # Gradient clipping
        normalize_advantage=False,  # A2C doesn't typically normalize
        verbose=1,
        device=device,
        policy_kwargs=dict(
            net_arch=[64, 64],     # Network architecture
            activation_fn=torch.nn.Tanh
        )
    )
    
    # Training parameters
    total_timesteps = 60000
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print("=" * 50)
    
    # Train the model
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time/60:.1f} minutes")
    
    # Save the model
    models_dir = "models/a2c"
    os.makedirs(models_dir, exist_ok=True)
    model_path = f"{models_dir}/a2c_model"
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
    print("A2C training complete!")

if __name__ == "__main__":
    main()
