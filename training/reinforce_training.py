import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add environment directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'environment'))
from custom_env import DiabetesTreatmentEnv

# Simple Policy Network for Diabetes Treatment
class PolicyNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 64),  # Smaller network for faster training
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.fc(x)

# Fast REINFORCE Trainer for Diabetes Treatment Environment
def reinforce_diabetes_training(episodes=50, gamma=0.95, lr=0.01):
    """Train REINFORCE on diabetes treatment environment with progress tracking."""
    print("REINFORCE Training for Diabetes Treatment Environment")
    print("=" * 60)
    
    # Create our custom diabetes environment
    env = DiabetesTreatmentEnv()
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    print(f"Environment: Diabetes Treatment (7x7 grid)")
    print(f"Network: {obs_size} -> 64 -> 32 -> {n_actions}")
    print(f"Episodes: {episodes} (fast training)")
    print(f"Learning rate: {lr}")
    print("=" * 60)
    print("Training Progress:")
    print("-" * 60)
    
    policy = PolicyNet(obs_size, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Training metrics
    episode_rewards = []
    best_reward = float('-inf')
    start_time = time.time()
    
    for episode in range(episodes):
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        episode_reward = 0
        step_count = 0
        
        # Suppress environment output during training
        import io
        import contextlib
        
        while not done:
            # Capture and suppress environment prints
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                probs = policy(obs_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                
                log_probs.append(dist.log_prob(action))
                obs, reward, terminated, truncated, _ = env.step(action.item())
                rewards.append(reward)
                episode_reward += reward
                step_count += 1
                
                done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Compute discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards)
        
        # Normalize rewards for stability
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Compute policy gradient loss
        loss = []
        for log_prob, R in zip(log_probs, discounted_rewards):
            loss.append(-log_prob * R)
        loss = torch.stack(loss).sum()
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Progress reporting - show every 5 episodes
        if episode % 5 == 0 or episode == episodes - 1:
            avg_reward = np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else np.mean(episode_rewards)
            elapsed_time = time.time() - start_time
            
            print(f"Ep {episode:2d}/{episodes} | "
                  f"Reward: {episode_reward:5.1f} | "
                  f"Avg5: {avg_reward:5.1f} | "
                  f"Best: {best_reward:5.1f} | "
                  f"Steps: {step_count:2d} | "
                  f"{elapsed_time:4.1f}s")
            
            # Force flush to ensure immediate display
            sys.stdout.flush()
    
    # Training summary
    total_time = time.time() - start_time
    final_avg = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
    
    print("-" * 60)
    print(f"TRAINING COMPLETED!")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Best reward: {best_reward:.1f}")
    print(f"Final avg reward: {final_avg:.1f}")
    print(f"Total episodes: {len(episode_rewards)}")
    
    # Save the model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = f"{models_dir}/reinforce_model.pth"
    
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'obs_size': obs_size,
        'n_actions': n_actions,
        'episode_rewards': episode_rewards,
        'training_time': total_time,
        'best_reward': best_reward,
        'final_avg': final_avg
    }, model_path)
    print(f"Model saved: {model_path}")
    
    # Quick silent test run
    print("\nTesting trained model (silent)...")
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        obs, _ = env.reset()
        test_reward = 0
        
        for step in range(3):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                probs = policy(obs_tensor)
                action = torch.argmax(probs).item()
            
            obs, reward, done, truncated, _ = env.step(action)
            test_reward += reward
            
            if done or truncated:
                break
    
    print(f"Test completed - Total reward: {test_reward:.1f}")
    env.close()
    print("REINFORCE training complete!")
    print("=" * 60)
    
    return policy, episode_rewards

if __name__ == "__main__":
    # Run fast REINFORCE training with clean output
    policy, rewards = reinforce_diabetes_training(episodes=50, gamma=0.95, lr=0.01)
