import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time
import pygame
from rendering import DiabetesRenderer

class DiabetesTreatmentEnv(gym.Env):
    """
    Custom Gymnasium environment for personalized diabetes treatment simulation.
    
    A nurse (agent) navigates a 7x7 grid to provide appropriate diabetes treatment
    based on simulated patient blood sugar levels. Agent moves every 3 in-game hours.
    """
    
    def __init__(self):
        super(DiabetesTreatmentEnv, self).__init__()
        
        # Environment parameters
        self.grid_size = 7
        self.simulation_duration = 60.0  # 60 seconds = 24 hours simulation
        self.decision_interval = 7.5  # Agent moves every 7.5 seconds (3 in-game hours)
        self.hour_duration = 2.5  # 1 hour = 2.5 seconds real time
        
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: [agent_x, agent_y, sugar_level, time_hours]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 40, 0]),
            high=np.array([6, 6, 300, 24]),
            dtype=np.float32
        )
        
        # Grid items positions (row, col) -> item_type
        self.grid_items = {
            # Top row (row 0)
            (0, 1): 'insulin',     # High dosage
            (0, 3): 'stop',        # No dosage
            (0, 5): 'insuline',    # Low dosage
            # Bottom row (row 6)
            (6, 1): 'fruits',      # Low sugar treatment
            (6, 3): 'nutrient',    # Medium sugar treatment
            (6, 5): 'candy'        # High sugar treatment
        }
        
        # Treatment effects on blood sugar
        self.treatment_effects = {
            'insulin': -15,     # Strong sugar reduction
            'insuline': -8,     # Mild sugar reduction
            'stop': 0,          # No effect
            'fruits': 8,        # Small sugar increase
            'nutrient': 5,      # Moderate sugar increase
            'candy': 15         # High sugar increase
        }
        
        self.reset()
        
        # Initialize renderer
        self.renderer = None
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Random agent starting position
        self.agent_pos = [
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        ]
        
        # Initialize blood sugar simulation
        self.sugar_level = np.random.uniform(90, 110)  # Start near normal
        self.sugar_history = [self.sugar_level]
        self.time_history = [0]
        
        # Simulation tracking
        self.total_reward = 0
        self.start_time = time.time()
        self.last_decision_time = self.start_time
        self.simulation_time = 0  # In hours (0-24)
        self.step_count = 0
        
        # Action feedback
        self.last_action_cell = None
        self.action_flash_start = 0
        self.is_thinking = False
        self.thinking_start = 0
        
        # Sugar level dynamics
        self.sugar_trend = np.random.uniform(-0.5, 0.5)  # Natural trend
        self.last_treatment_effect = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in the environment."""
        current_time = time.time()
        
        # Only allow decisions every 7.5 seconds (3 in-game hours)
        if current_time - self.last_decision_time >= self.decision_interval:
            # Show thinking overlay
            self.is_thinking = True
            self.thinking_start = current_time
            
            # Wait 0.5 seconds for thinking animation
            time.sleep(0.5)
            
            # Execute movement
            old_pos = self.agent_pos.copy()
            self._move_agent(action)
            
            # Flash action cell
            self.last_action_cell = tuple(self.agent_pos)
            self.action_flash_start = current_time
            
            self.last_decision_time = current_time
            self.step_count += 1
            self.is_thinking = False
            
            # Apply treatment effect
            self._apply_treatment()
            
            # Calculate reward
            reward = self._calculate_reward(old_pos, self.agent_pos)
            self.total_reward += reward
            
            # Log action for debugging
            self._log_action(action, reward)
        else:
            reward = 0
        
        # Update simulation time continuously
        elapsed = current_time - self.start_time
        self.simulation_time = min((elapsed / self.simulation_duration) * 24, 24)
        
        # Update sugar level continuously
        self._update_sugar_level()
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Get current observation
        obs = self._get_observation()
        
        return obs, reward, done, False, {}
    
    def _move_agent(self, action):
        """Move agent based on action with smooth animation."""
        if action == 0:  # Up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # Down
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # Left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # Right
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
    
    def _apply_treatment(self):
        """Apply treatment effect based on current position."""
        pos_tuple = tuple(self.agent_pos)
        
        if pos_tuple in self.grid_items:
            item_type = self.grid_items[pos_tuple]
            self.last_treatment_effect = self.treatment_effects[item_type]
        else:
            self.last_treatment_effect = 0
    
    def _update_sugar_level(self):
        """Update blood sugar level with realistic simulation."""
        # Natural fluctuation (small random changes)
        natural_change = np.random.normal(0, 1.0)
        
        # Natural trend (sugar tends to rise without treatment)
        trend_change = self.sugar_trend * 0.2
        
        # Treatment effect (gradually applied)
        treatment_change = self.last_treatment_effect * 0.1
        self.last_treatment_effect *= 0.95  # Decay treatment effect
        
        # Apply all changes
        total_change = natural_change + trend_change + treatment_change
        self.sugar_level = np.clip(self.sugar_level + total_change, 40, 300)
        
        # Store history for plotting (every 2.5 seconds for smooth graph)
        if len(self.time_history) == 0 or self.simulation_time - self.time_history[-1] >= 1.0:
            self.sugar_history.append(self.sugar_level)
            self.time_history.append(self.simulation_time)
            
            # Keep history manageable
            if len(self.sugar_history) > 100:
                self.sugar_history = self.sugar_history[-100:]
                self.time_history = self.time_history[-100:]
    
    def _calculate_reward(self, old_pos, new_pos):
        """Calculate reward based on agent movement and current state."""
        reward = 0
        
        # Perfect sugar range bonus
        if 80 <= self.sugar_level <= 105:
            reward += 5
        
        # Treatment cell rewards
        new_pos_tuple = tuple(new_pos)
        if new_pos_tuple in self.grid_items:
            item_type = self.grid_items[new_pos_tuple]
            reward += self._get_treatment_reward(item_type)
        
        # Distance-based reward
        reward += self._get_distance_reward(new_pos)
        
        # Penalties for dangerous levels
        if self.sugar_level < 50 or self.sugar_level > 250:
            reward -= 20
        elif self.sugar_level < 70 or self.sugar_level > 180:
            reward -= 5
        
        return reward
    
    def _get_treatment_reward(self, item_type):
        """Get reward for choosing a specific treatment."""
        if item_type == 'insulin':  # High dosage
            if self.sugar_level > 150:
                return 10  # Correct for high sugar
            elif self.sugar_level < 80:
                return -10  # Wrong for low sugar
            else:
                return 0
                
        elif item_type == 'insuline':  # Low dosage
            if 120 < self.sugar_level <= 150:
                return 10  # Correct for moderately high sugar
            elif self.sugar_level < 80:
                return -10  # Wrong for low sugar
            else:
                return 2  # Preventive
                
        elif item_type == 'stop':  # No dosage
            if 80 <= self.sugar_level <= 120:
                return 10  # Good to not interfere when normal
            else:
                return -10  # Should be treating
                
        elif item_type == 'fruits':  # Low sugar treatment
            if self.sugar_level < 80:
                return 10  # Correct for low sugar
            elif self.sugar_level > 140:
                return -10  # Wrong for high sugar
            else:
                return 2
                
        elif item_type == 'nutrient':  # Medium sugar treatment
            if 60 <= self.sugar_level <= 90:
                return 10  # Good for mild low sugar
            elif self.sugar_level > 150:
                return -10  # Wrong for high sugar
            else:
                return 3
                
        elif item_type == 'candy':  # High sugar treatment
            if self.sugar_level < 60:
                return 10  # Emergency treatment
            elif self.sugar_level > 120:
                return -10  # Very wrong
            else:
                return -5  # Generally not ideal
        
        return 0
    
    def _get_distance_reward(self, pos):
        """Get reward based on distance to appropriate treatment."""
        if self.sugar_level > 150:
            # Need insulin
            target_positions = [(0, 1), (0, 5)]  # insulin, insuline
            min_distance = min([self._manhattan_distance(pos, tpos) for tpos in target_positions])
            return max(0, 2 - min_distance * 0.3)
            
        elif self.sugar_level < 80:
            # Need sugar
            sugar_positions = [(6, 1), (6, 3), (6, 5)]  # fruits, nutrient, candy
            min_distance = min([self._manhattan_distance(pos, spos) for spos in sugar_positions])
            return max(0, 2 - min_distance * 0.3)
            
        return 0
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _is_episode_done(self):
        """Check if episode should end."""
        # End if simulation time reaches 24 hours
        if self.simulation_time >= 24:
            return True
            
        # End if sugar level is critically dangerous
        if self.sugar_level < 40 or self.sugar_level > 300:
            return True
            
        return False
    
    def _get_observation(self):
        """Get current observation state."""
        return np.array([
            float(self.agent_pos[0]),
            float(self.agent_pos[1]),
            self.sugar_level,
            self.simulation_time
        ], dtype=np.float32)
    
    def _log_action(self, action, reward):
        """Log action details for debugging."""
        action_names = ['Up', 'Down', 'Left', 'Right']
        print(f"[{self.get_formatted_time()}] Agent moved {action_names[action]} to {self.agent_pos}, "
              f"Sugar: {self.sugar_level:.1f} mg/dL, Action: {self.get_current_action_text()}, "
              f"Reward: {reward:.1f}, Total: {self.total_reward:.1f}")
    
    def get_current_action_text(self):
        """Get current action description based on agent position."""
        pos_tuple = tuple(self.agent_pos)
        
        if pos_tuple in self.grid_items:
            item_type = self.grid_items[pos_tuple]
            action_map = {
                'insulin': 'High dosage',
                'insuline': 'Low dosage', 
                'stop': 'No dosage',
                'fruits': 'Low sugar treatment',
                'nutrient': 'Medium sugar treatment',
                'candy': 'High sugar treatment'
            }
            return action_map.get(item_type, 'No action')
        
        return 'No action'
    
    def get_patient_status(self):
        """Get patient status based on current sugar level."""
        if self.sugar_level < 40 or self.sugar_level > 300:
            return 'died'
        elif 80 <= self.sugar_level <= 105:
            return 'person'  # Perfect range
        else:
            return 'patient'  # Outside perfect range but alive
    
    def get_formatted_time(self):
        """Get formatted simulation time as HH:MM."""
        hours = int(self.simulation_time)
        minutes = int((self.simulation_time - hours) * 60)
        return f"{hours:02d}:{minutes:02d}"
    
    def should_flash_cell(self, row, col):
        """Check if a cell should flash light blue."""
        if self.last_action_cell is None:
            return False
        
        current_time = time.time()
        if current_time - self.action_flash_start > 0.5:  # Flash for 0.5 seconds
            return False
            
        return (row, col) == self.last_action_cell
    
    def render(self):
        """Render the environment."""
        if self.renderer is None:
            self.renderer = DiabetesRenderer(self)
        
        return self.renderer.render()
    
    def close(self):
        """Close the environment and renderer."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def run_random_simulation():
    """Run a simulation with random agent movements."""
    env = DiabetesTreatmentEnv()
    obs, _ = env.reset()
    
    print("🏥 Starting Diabetes Treatment Simulation...")
    print("📊 Agent makes decisions every 7.5 seconds (3 in-game hours)")
    print("⏱️ Simulation duration: 60 seconds (24 hours game time)")
    print("🔄 Close window to stop simulation\n")
    
    running = True
    
    while running:
        # Generate random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Render
        running = env.render()
        if not running:
            break
        
        # Check if simulation should end
        if done:
            print(f"\n🏁 Episode ended: Sugar level {env.sugar_level:.1f} mg/dL")
            break
        
        # Small delay to control frame rate
        time.sleep(0.1)
    
    print(f"\n📈 Simulation Results:")
    print(f"   Final sugar level: {env.sugar_level:.1f} mg/dL")
    print(f"   Total reward: {env.total_reward:.1f}")
    print(f"   Patient status: {env.get_patient_status()}")
    print(f"   Decisions made: {env.step_count}")
    
    # Save screenshot
    if env.renderer:
        env.renderer.save_screenshot("diabetes_simulation_final.png")
    
    env.close()


if __name__ == "__main__":
    run_random_simulation()