#!/usr/bin/env python3
"""
Random Action Demo for Diabetes Treatment Environment

This file demonstrates the agent taking random actions in the custom environment
without any trained model, as required by the assignment.
Creates a GIF showing random exploration of the environment.
"""

import sys
import os
import time
import random
import numpy as np

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from PIL import Image
    import pygame
    PIL_AVAILABLE = True
    PYGAME_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    PYGAME_AVAILABLE = False
    print("Warning: PIL or Pygame not available for GIF recording")

# Import environment - try different paths
try:
    from custom_env import DiabetesTreatmentEnv
except ImportError:
    try:
        from environment.custom_env import DiabetesTreatmentEnv
    except ImportError:
        # Add parent directory to path and try again
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)
        from environment.custom_env import DiabetesTreatmentEnv

class RandomActionGIFRecorder:
    def __init__(self, filename="random_agent_demo.gif", duration=15, fps=5):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.filename = os.path.join(self.root_dir, filename)
        self.duration = duration
        self.fps = fps
        self.frames = []
        self.recording = False
        self.start_time = None
        self.frame_interval = 1.0 / fps
        self.last_capture_time = 0
        
    def start_recording(self):
        if not PIL_AVAILABLE or not PYGAME_AVAILABLE:
            print("Warning: Cannot record GIF - PIL or Pygame not available")
            return False
            
        self.frames = []
        self.recording = True
        self.start_time = time.time()
        self.last_capture_time = 0
        print(f"Started recording random demo GIF: {self.filename}")
        return True
        
    def capture_frame(self, surface):
        if not self.recording or not PIL_AVAILABLE or not PYGAME_AVAILABLE:
            return
        
        if surface is None:
            return
            
        current_time = time.time() - self.start_time
        
        if current_time >= self.duration:
            self.stop_recording()
            return
            
        if current_time - self.last_capture_time >= self.frame_interval:
            try:
                w, h = surface.get_size()
                raw = pygame.image.tostring(surface, 'RGB')
                pil_image = Image.frombytes('RGB', (w, h), raw)
                
                # Resize if too large
                if w > 600 or h > 450:
                    pil_image = pil_image.resize((600, 450), Image.Resampling.LANCZOS)
                
                self.frames.append(pil_image)
                self.last_capture_time = current_time
                
                if len(self.frames) % 10 == 0:
                    progress = (current_time / self.duration) * 100
                    print(f"Recording random demo... {progress:.1f}%")
                    
            except Exception as e:
                print(f"Error capturing frame: {e}")
                
    def stop_recording(self):
        if not self.recording:
            return
            
        self.recording = False
        
        if not self.frames:
            print("No frames captured for random demo GIF")
            return
            
        try:
            print(f"Saving random demo GIF with {len(self.frames)} frames...")
            frame_duration = int(1000 / self.fps)
            
            self.frames[0].save(
                self.filename,
                save_all=True,
                append_images=self.frames[1:],
                duration=frame_duration,
                loop=0,
                optimize=True
            )
            
            file_size = os.path.getsize(self.filename) / (1024 * 1024)
            print(f"Random demo GIF saved: {self.filename} ({file_size:.1f} MB)")
            
        except Exception as e:
            print(f"Error saving random demo GIF: {e}")
        finally:
            self.frames = []

def create_random_action_demo():
    """
    Create a demonstration of agent taking random actions in the environment.
    This shows the environment visualization without any trained model.
    """
    print("=" * 60)
    print("RANDOM ACTION DEMO - DIABETES TREATMENT ENVIRONMENT")
    print("=" * 60)
    print("Demonstrating agent taking random actions (no trained model)")
    print("This shows the environment components and visualization")
    
    # Create environment
    env = DiabetesTreatmentEnv()
    
    # Initialize GIF recorder
    gif_recorder = RandomActionGIFRecorder()
    gif_recorder.start_recording()
    
    print("\nEnvironment Information:")
    print(f"Action Space: {env.action_space} (0=Up, 1=Down, 2=Left, 3=Right)")
    print(f"Observation Space: {env.observation_space}")
    print("Grid Items:")
    for pos, item in env.grid_items.items():
        print(f"  Position {pos}: {item}")
    
    try:
        obs, _ = env.reset()
        step_count = 0
        episode_reward = 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        action_names = ["Up", "Down", "Left", "Right"]
        
        print(f"\nStarting random action demonstration...")
        print(f"Initial state: Agent at ({env.agent_pos[0]}, {env.agent_pos[1]}), Sugar: {env.sugar_level:.1f} mg/dL")
        
        while step_count < 200:  # Run for 200 steps
            # Take random action
            action = env.action_space.sample()  # Random action
            action_counts[action] += 1
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            # Render and capture
            surface = env.render()
            if gif_recorder and surface:
                gif_recorder.capture_frame(surface)
            
            # Print occasional updates
            if step_count % 25 == 0:
                print(f"Step {step_count}: Action={action_names[action]}, "
                      f"Position=({env.agent_pos[0]}, {env.agent_pos[1]}), "
                      f"Sugar={env.sugar_level:.1f} mg/dL, "
                      f"Reward={reward:.2f}")
            
            step_count += 1
            time.sleep(0.2)  # Slower for better visualization
            
            # Reset if episode ends
            if done or truncated:
                print(f"Episode ended at step {step_count}, resetting...")
                obs, _ = env.reset()
                episode_reward = 0
        
        if gif_recorder and gif_recorder.recording:
            gif_recorder.stop_recording()
        
        print("\n" + "=" * 60)
        print("RANDOM ACTION DEMO RESULTS")
        print("=" * 60)
        print(f"Total steps: {step_count}")
        print(f"Final position: ({env.agent_pos[0]}, {env.agent_pos[1]})")
        print(f"Final sugar level: {env.sugar_level:.1f} mg/dL")
        print(f"Total reward: {episode_reward:.2f}")
        
        print("\nAction Distribution:")
        total_actions = sum(action_counts.values())
        for action, count in action_counts.items():
            percentage = (count / total_actions) * 100
            print(f"  {action_names[action]}: {count} times ({percentage:.1f}%)")
        
        print(f"\nDemonstration complete! Random demo GIF saved.")
        
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
        if gif_recorder and gif_recorder.recording:
            gif_recorder.stop_recording()
    except Exception as e:
        print(f"Demo error: {e}")
        if gif_recorder and gif_recorder.recording:
            gif_recorder.stop_recording()
    finally:
        env.close()

if __name__ == "__main__":
    create_random_action_demo()
