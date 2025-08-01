#!/usr/bin/env python3
"""
Main Application: Universal RL-Guided Diabetes Treatment Simulation

This application can load and use any trained RL model (A2C, PPO, REINFORCE, etc.)
to guide the agent's decisions. Simply specify the model path and type.
The simulation displays the GUI to show intelligent treatment decisions in real-time.
"""

# ========== CONFIGURATION SECTION ==========
DEFAULT_MODEL_PATH = "training/models/dqn/dqn_model"
DEFAULT_MODEL_TYPE = "auto"

# GIF Recording Configuration
RECORD_GIF = True
GIF_FILENAME = "diabetes_simulation.gif"
GIF_DURATION = 10
GIF_FPS = 10
# ===============================================

import sys
import os
import time
import argparse
import torch

# Import PIL for GIF creation
try:
    from PIL import Image
    import pygame
    PIL_AVAILABLE = True
    PYGAME_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    PYGAME_AVAILABLE = False

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'environment'))

# Import required libraries
try:
    from stable_baselines3 import A2C, PPO
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from environment.custom_env import DiabetesTreatmentEnv

# REINFORCE Policy Network Definition
class PolicyNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.fc(x)

class GIFRecorder:
    def __init__(self, filename="simulation.gif", duration=30, fps=10):
        # Ensure the GIF is saved in the root directory (where main.py is located)
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
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
        print(f"Started recording GIF: {self.filename} ({self.duration}s @ {self.fps}fps)")
        return True
        
    def capture_frame(self, surface):
        if not self.recording or not PIL_AVAILABLE or not PYGAME_AVAILABLE:
            return
        
        # Check if surface is valid
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
                
                if w > 800 or h > 600:
                    pil_image = pil_image.resize((min(800, w), min(600, h)), Image.Resampling.LANCZOS)
                
                self.frames.append(pil_image)
                self.last_capture_time = current_time
                
                if len(self.frames) % (self.fps * 5) == 0:
                    progress = (current_time / self.duration) * 100
                    print(f"Recording... {progress:.1f}% ({len(self.frames)} frames)")
                    
            except Exception as e:
                print(f"Error capturing frame: {e}")
                
    def stop_recording(self):
        if not self.recording:
            return
            
        self.recording = False
        
        if not self.frames:
            print("No frames captured for GIF")
            return
            
        try:
            print(f"Saving GIF with {len(self.frames)} frames...")
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
            print(f"GIF saved successfully: {self.filename} ({file_size:.1f} MB)")
            print(f"GIF Details: {len(self.frames)} frames, {self.fps} fps, {self.duration}s duration")
            
        except Exception as e:
            print(f"Error saving GIF: {e}")
        finally:
            self.frames = []

class UniversalRLGuidedEnvironment(DiabetesTreatmentEnv):
    def __init__(self, model_path=None, model_type="auto"):
        original_dir = os.getcwd()
        env_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'environment')
        os.chdir(env_dir)
        
        try:
            super().__init__()
        finally:
            os.chdir(original_dir)
        
        self.model = None
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.model_enabled = False
        
        if model_path:
            self.load_model()
        
        self.decisions_made = 0
        self.model_decisions = 0
        self.fallback_decisions = 0
        
        self._verify_images()
    
    def load_model(self):
        if not self.model_path:
            print("No model path specified. Using fallback logic.")
            self.model_enabled = False
            return
        
        if self.model_type == "auto":
            self.model_type = self._detect_model_type()
        
        try:
            if self.model_type in ["a2c", "ppo"]:
                self._load_stable_baselines_model()
            elif self.model_type == "reinforce":
                self._load_reinforce_model()
            else:
                print(f"Unknown model type: {self.model_type}")
                self.model_enabled = False
                return
                
            print(f"{self.model_type.upper()} model loaded successfully from {self.model_path}")
            self.model_enabled = True
            
        except Exception as e:
            print(f"Error loading {self.model_type.upper()} model: {e}")
            print("Using fallback logic instead.")
            self.model_enabled = False
    
    def _detect_model_type(self):
        model_path_lower = self.model_path.lower()
        
        if os.path.exists(f"{self.model_path}.pth"):
            return "reinforce"
        
        if os.path.exists(f"{self.model_path}.zip"):
            if "a2c" in model_path_lower:
                return "a2c"
            elif "ppo" in model_path_lower:
                return "ppo"
            else:
                return "a2c"
        
        if "reinforce" in model_path_lower:
            return "reinforce"
        elif "ppo" in model_path_lower:
            return "ppo"
        elif "a2c" in model_path_lower:
            return "a2c"
        
        print(f"Could not auto-detect model type for {self.model_path}")
        return "a2c"
    
    def _load_stable_baselines_model(self):
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("Stable Baselines3 not available")
        
        model_file = f"{self.model_path}.zip"
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        if self.model_type == "a2c":
            self.model = A2C.load(self.model_path)
        elif self.model_type == "ppo":
            self.model = PPO.load(self.model_path)
        else:
            raise ValueError(f"Unknown Stable Baselines model type: {self.model_type}")
    
    def _load_reinforce_model(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        model_file = f"{self.model_path}.pth"
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        checkpoint = torch.load(model_file, map_location='cpu')
        obs_size = checkpoint.get('obs_size', 4)
        n_actions = checkpoint.get('n_actions', 4)
        
        self.model = PolicyNet(obs_size, n_actions)
        self.model.load_state_dict(checkpoint['policy_state_dict'])
        self.model.eval()
    
    def _verify_images(self):
        env_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'environment')
        image_dir = os.path.join(env_dir, 'image')
        
        expected_images = ['doctor.png', 'insulin.png', 'candy.png', 'fruits.png', 
                          'nutrient.png', 'stop.png', 'patient.png', 'person.png', 'died.png']
        
        missing_images = []
        for img in expected_images:
            img_path = os.path.join(image_dir, img)
            if not os.path.exists(img_path):
                missing_images.append(img)
        
        if missing_images:
            print(f"Missing images: {missing_images} (will use fallback graphics)")
        else:
            print("All graphics loaded successfully")
    
    def _get_optimal_action(self):
        self.decisions_made += 1
        
        if self.model_enabled and self.model is not None:
            try:
                obs = self._get_observation()
                
                if self.model_type in ["a2c", "ppo"]:
                    action, _states = self.model.predict(obs, deterministic=True)
                    treatment = self._convert_sb3_action_to_treatment(action)
                elif self.model_type == "reinforce":
                    action = self._predict_reinforce_action(obs)
                    treatment = self._convert_sb3_action_to_treatment(action)
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")
                
                self.model_decisions += 1
                return treatment
                
            except Exception as e:
                print(f"{self.model_type.upper()} prediction error: {e}")
                self.fallback_decisions += 1
                return self._get_fallback_action()
        else:
            self.fallback_decisions += 1
            return self._get_fallback_action()
    
    def _predict_reinforce_action(self, obs):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            probs = self.model(obs_tensor)
            action = torch.argmax(probs, dim=1).item()
        return action
    
    def _convert_sb3_action_to_treatment(self, action):
        sugar = self.sugar_level
        
        if action == 0:
            if sugar > 150:
                return 'insulin'
            elif sugar > 120:
                return 'fruits'
            else:
                return 'insuline'
                
        elif action == 1:
            if sugar < 80:
                return 'candy'
            elif sugar < 100:
                return 'nutrient'
            else:
                return 'stop'
                
        elif action == 2:
            if sugar > 140:
                return 'insulin'
            else:
                return 'stop'
                
        elif action == 3:
            if sugar < 90:
                return 'candy'
            else:
                return 'insuline'
        
        return 'stop'
    
    def _get_fallback_action(self):
        if self.sugar_level < 60:
            return 'candy'
        elif self.sugar_level > 200:
            return 'insulin'
        
        hour_of_day = self.simulation_time % 24
        
        if 6 <= hour_of_day < 10:
            if self.sugar_level > 140:
                return 'insulin'
            elif self.sugar_level > 120:
                return 'insuline'
            elif 70 <= self.sugar_level <= 120:
                return 'stop'
            elif self.sugar_level < 80:
                return 'fruits'
                
        elif (12 <= hour_of_day < 13) or (18 <= hour_of_day < 19):
            if self.sugar_level > 150:
                return 'insulin'
            elif self.sugar_level > 130:
                return 'insuline'
            elif 80 <= self.sugar_level <= 130:
                return 'stop'
            elif self.sugar_level < 80:
                return 'nutrient'
                
        elif hour_of_day >= 22 or hour_of_day < 6:
            if self.sugar_level > 180:
                return 'insuline'
            elif self.sugar_level > 160:
                return 'stop'
            elif 80 <= self.sugar_level <= 160:
                return 'stop'
            elif self.sugar_level < 80:
                return 'fruits'
        else:
            if self.sugar_level > 160:
                return 'insulin'
            elif self.sugar_level > 130:
                return 'insuline'
            elif 80 <= self.sugar_level <= 130:
                return 'stop'
            elif self.sugar_level < 80:
                return 'nutrient'
                
        return 'stop'
    
    def render(self):
        if self.renderer is None:
            original_dir = os.getcwd()
            env_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'environment')
            os.chdir(env_dir)
            
            try:
                from rendering import DiabetesRenderer
                self.renderer = DiabetesRenderer(self)
            finally:
                os.chdir(original_dir)
        
        # Call the renderer and check if we should quit
        result = self.renderer.render()
        if result is False:
            return None  # Quit event received
        
        # Return the surface for GIF recording
        return self.renderer.screen
    
    def get_decision_stats(self):
        total = self.decisions_made
        if total == 0:
            return "No decisions made yet"
        
        model_percent = (self.model_decisions / total) * 100
        fallback_percent = (self.fallback_decisions / total) * 100
        
        model_name = self.model_type.upper() if self.model_enabled else "NONE"
        
        return (f"Decisions: {total} total | "
                f"{model_name}: {self.model_decisions} ({model_percent:.1f}%) | "
                f"Fallback: {self.fallback_decisions} ({fallback_percent:.1f}%)")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Universal RL-Guided Diabetes Treatment Simulation')
    parser.add_argument('--model', '-m', type=str, 
                       help='Path to model file (without extension)')
    parser.add_argument('--type', '-t', type=str, default='auto',
                       choices=['auto', 'a2c', 'ppo', 'reinforce'],
                       help='Model type (auto-detect if not specified)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available trained models')
    
    return parser.parse_args()

def list_available_models():
    """List all available trained models."""
    print("Available Models:")
    print("=" * 50)
    
    models_found = []
    
    # Check training directory for models
    training_dir = "training"
    if os.path.exists(training_dir):
        # Check for A2C models
        a2c_dir = os.path.join(training_dir, "models", "a2c")
        if os.path.exists(a2c_dir):
            for file in os.listdir(a2c_dir):
                if file.endswith('.zip'):
                    model_path = os.path.join(a2c_dir, file[:-4])  # Remove .zip
                    models_found.append(('A2C', model_path))
        
        # Check for PPO models
        ppo_dir = os.path.join(training_dir, "models", "ppo")
        if os.path.exists(ppo_dir):
            for file in os.listdir(ppo_dir):
                if file.endswith('.zip'):
                    model_path = os.path.join(ppo_dir, file[:-4])  # Remove .zip
                    models_found.append(('PPO', model_path))
        
        # Check for REINFORCE models
        models_dir = os.path.join(training_dir, "models")
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.pth'):
                    model_path = os.path.join(models_dir, file[:-4])  # Remove .pth
                    models_found.append(('REINFORCE', model_path))
    
    if models_found:
        for model_type, path in models_found:
            print(f"  {model_type:<10} | {path}")
        
        print("\nUsage examples:")
        print("  python main.py --model training/models/a2c/a2c_model")
        print("  python main.py --model training/models/reinforce_model --type reinforce")
        print("  python main.py --model training/models/ppo/ppo_model --type ppo")
    else:
        print("  No trained models found!")
        print("  Train models first:")
        print("    cd training && python a2c_training.py")
        print("    cd training && python ppo_training.py") 
        print("    cd training && python reinforce_training.py")

def run_universal_simulation(model_path=None, model_type="auto"):
    print("Universal RL-Guided Diabetes Treatment Simulation")
    print("=" * 60)
    
    env = UniversalRLGuidedEnvironment(model_path=model_path, model_type=model_type)
    
    if env.model_enabled:
        print(f"{env.model_type.upper()} model active - Making intelligent decisions")
    else:
        print("Using fallback logic - No model loaded")
    
    gif_recorder = None
    if RECORD_GIF:
        gif_recorder = GIFRecorder(
            filename=GIF_FILENAME,
            duration=GIF_DURATION,
            fps=GIF_FPS
        )
        gif_recorder.start_recording()
    
    print("Starting simulation...")
    print("=" * 60)
    
    try:
        obs, _ = env.reset()
        start_time = time.time()
        step_count = 0
        sugar_readings = []
        
        while True:
            obs, reward, done, truncated, info = env.step(0)
            surface = env.render()
            
            if gif_recorder and surface:
                gif_recorder.capture_frame(surface)
            
            step_count += 1
            sugar_readings.append(env.sugar_level)
            
            if step_count % 100 == 0:
                avg_sugar = sum(sugar_readings) / len(sugar_readings)
                status = "NORMAL" if 80 <= env.sugar_level <= 120 else "OUT_OF_RANGE"
                print(f"Step {step_count}: Sugar {env.sugar_level:.1f} mg/dL | {status}")
            
            time.sleep(0.1)
            
            if done or truncated:
                print("Simulation completed!")
                break
                
            sim_duration = time.time() - start_time
            max_duration = max(120, GIF_DURATION + 10) if RECORD_GIF else 120
            if sim_duration > max_duration:
                print("Time limit reached!")
                break
            
            if gif_recorder and not gif_recorder.recording and RECORD_GIF:
                print("GIF recording completed!")
        
        if gif_recorder and gif_recorder.recording:
            gif_recorder.stop_recording()
        
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)
        
        final_sugar = env.sugar_level
        avg_sugar = sum(sugar_readings) / len(sugar_readings)
        min_sugar = min(sugar_readings)
        max_sugar = max(sugar_readings)
        
        target_readings = [s for s in sugar_readings if 80 <= s <= 120]
        time_in_range = (len(target_readings) / len(sugar_readings)) * 100
        
        print(f"Final level: {final_sugar:.1f} mg/dL")
        print(f"Average level: {avg_sugar:.1f} mg/dL")
        print(f"Range: {min_sugar:.1f} - {max_sugar:.1f} mg/dL")
        print(f"Time in target range (80-120): {time_in_range:.1f}%")
        print(f"{env.get_decision_stats()}")
        
        if env.model_enabled:
            print(f"{env.model_type.upper()} guided {env.model_decisions} decisions")
        
        if time_in_range >= 70:
            print(f"EXCELLENT: {time_in_range:.1f}% time in target range!")
        elif time_in_range >= 50:
            print(f"GOOD: {time_in_range:.1f}% time in target range")
        else:
            print(f"NEEDS IMPROVEMENT: {time_in_range:.1f}% time in target range")
        
        if RECORD_GIF and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), GIF_FILENAME)):
            gif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), GIF_FILENAME)
            file_size = os.path.getsize(gif_path) / (1024 * 1024)
            print(f"GIF Recording: {GIF_FILENAME} ({file_size:.1f} MB)")
            print(f"GIF saved to: {gif_path}")
        
    except KeyboardInterrupt:
        print("Simulation stopped by user")
        if gif_recorder and gif_recorder.recording:
            print("Saving GIF before exit...")
            gif_recorder.stop_recording()
    except Exception as e:
        print(f"Simulation error: {e}")
        if gif_recorder and gif_recorder.recording:
            gif_recorder.stop_recording()
    finally:
        env.close()
        print("Simulation Complete!")

def main():
    print("Universal RL-Guided Diabetes Treatment System")
    print("=" * 50)
    
    args = parse_arguments()
    
    if args.list_models:
        list_available_models()
        return
    
    try:
        import pygame
    except ImportError:
        print("Pygame not available! Install with: pip install pygame")
        return
    
    if RECORD_GIF:
        if PIL_AVAILABLE and PYGAME_AVAILABLE:
            print(f"GIF Recording: ENABLED ({GIF_FILENAME}, {GIF_DURATION}s @ {GIF_FPS}fps)")
        else:
            print("GIF Recording: DISABLED (PIL/Pillow or Pygame not available)")
            print("   Install with: pip install pillow pygame")
    else:
        print("GIF Recording: DISABLED (set RECORD_GIF = True to enable)")
    
    if args.model:
        model_path = args.model
        model_type = args.type
        print(f"Using command line model: {model_path}")
    elif DEFAULT_MODEL_PATH:
        model_path = DEFAULT_MODEL_PATH
        model_type = DEFAULT_MODEL_TYPE
        print(f"Using configured model: {model_path}")
    else:
        model_path = None
        model_type = "auto"
        print("No model configured, using fallback logic")
    
    run_universal_simulation(model_path, model_type)

if __name__ == "__main__":
    main()


