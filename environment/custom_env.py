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
        self.decision_interval = 7.5  # Agent makes decisions every 7.5 seconds (3 in-game hours)
        self.hour_duration = 2.5  # 1 hour = 2.5 seconds real time
        
        # Movement timing
        self.movement_duration = 4.0  # 4 seconds to reach treatment (slower movement)
        self.treatment_duration = 1.5  # 1.5 seconds at treatment location
        # No return duration - instant return to center
        
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: [agent_x, agent_y, sugar_level, time_hours]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 40, 0]),
            high=np.array([6, 6, 300, 24]),
            dtype=np.float32
        )
        
        # Grid items positions (row, col) -> item_type - placed in corners
        self.grid_items = {
            # Corner positions
            (0, 0): 'insulin',     # Top-left: High dosage
            (0, 6): 'insuline',    # Top-right: Low dosage
            (6, 0): 'stop',        # Bottom-left: No dosage
            (6, 6): 'candy',       # Bottom-right: High sugar treatment
            # Additional strategic positions
            (0, 3): 'fruits',      # Top-center: Low sugar treatment
            (6, 3): 'nutrient',    # Bottom-center: Medium sugar treatment
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
        
        # Agent always starts in the center of the grid (3, 3)
        self.agent_pos = [3, 3]  # Center position in 7x7 grid
        
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
        
        # Movement tracking for smooth animation
        self.is_moving = False
        self.target_pos = None
        self.movement_start_time = 0
        self.movement_path = []
        self.current_path_index = 0
        self.move_progress = 0.0  # Progress percentage for rendering
        self.cycle_complete = True  # Ready for next decision cycle
        self.current_cycle_reward = 0  # Track reward for current cycle
        
        # Sugar level dynamics
        self.sugar_trend = np.random.uniform(-0.5, 0.5)  # Natural trend
        self.last_treatment_effect = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in the environment with intelligent movement."""
        current_time = time.time()
        
        # Handle ongoing movement animation
        if self.is_moving:
            self._update_movement_animation()
            reward = 0
        # Only allow new decisions when cycle is complete and 3 hours have passed
        elif self.cycle_complete and current_time - self.last_decision_time >= self.decision_interval:
            # Show thinking overlay
            self.is_thinking = True
            self.thinking_start = current_time
            
            # Wait 0.5 seconds for thinking animation
            time.sleep(0.5)
            
            # Determine optimal treatment based on sugar level and time
            optimal_treatment = self._get_optimal_action()
            
            # Start movement to target location
            self._start_movement_to_target(optimal_treatment)
            
            self.last_decision_time = current_time
            self.step_count += 1
            self.is_thinking = False
            self.cycle_complete = False  # Mark cycle as in progress
            self.current_cycle_reward = 0  # Reset cycle reward
            
            # Log action for debugging
            self._log_action_intelligent(optimal_treatment, 0)
            reward = 0
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
        """Get reward for choosing a specific treatment (+10 correct, -10 wrong)."""
        if item_type == 'insulin':  # High dosage
            if self.sugar_level > 150:
                return 10  # Correct for high sugar
            else:
                return -10  # Wrong for normal/low sugar
                
        elif item_type == 'insuline':  # Low dosage
            if 120 < self.sugar_level <= 150:
                return 10  # Correct for moderately high sugar
            else:
                return -10  # Wrong for other levels
                
        elif item_type == 'stop':  # No dosage
            if 80 <= self.sugar_level <= 120:
                return 10  # Good to not interfere when normal
            else:
                return -10  # Should be treating
                
        elif item_type == 'fruits':  # Low sugar treatment
            if self.sugar_level < 80:
                return 10  # Correct for low sugar
            else:
                return -10  # Wrong for normal/high sugar
                
        elif item_type == 'nutrient':  # Medium sugar treatment
            if 60 <= self.sugar_level <= 90:
                return 10  # Good for mild low sugar
            else:
                return -10  # Wrong for other levels
                
        elif item_type == 'candy':  # High sugar treatment
            if self.sugar_level < 60:
                return 10  # Emergency treatment
            else:
                return -10  # Wrong for normal/high sugar
        
        return 0
    
    def _get_distance_reward(self, pos):
        """Get reward based on distance to appropriate treatment."""
        if self.sugar_level > 150:
            # Need insulin - corner positions
            target_positions = [(0, 0), (0, 6)]  # insulin, insuline
            min_distance = min([self._manhattan_distance(pos, tpos) for tpos in target_positions])
            return max(0, 2 - min_distance * 0.3)
            
        elif self.sugar_level < 80:
            # Need sugar - corner and edge positions
            sugar_positions = [(6, 6), (0, 3), (6, 3)]  # candy, fruits, nutrient
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
    
    def _get_optimal_action(self):
        """Determine the optimal treatment based on current sugar level and time."""
        # Critical situations - immediate action needed
        if self.sugar_level < 60:  # Severe hypoglycemia
            return 'candy'  # Emergency high sugar treatment
        elif self.sugar_level > 200:  # Severe hyperglycemia
            return 'insulin'  # Strong insulin needed
        
        # Time-based treatment decisions
        hour_of_day = self.simulation_time % 24
        
        # Morning (6-10): More aggressive treatment due to dawn phenomenon
        if 6 <= hour_of_day < 10:
            if self.sugar_level > 140:
                return 'insulin'
            elif self.sugar_level > 120:
                return 'insuline'
            elif 70 <= self.sugar_level <= 120:
                return 'stop'
            elif self.sugar_level < 80:
                return 'fruits'
                
        # Meal times (12-13, 18-19): Preventive care
        elif (12 <= hour_of_day < 13) or (18 <= hour_of_day < 19):
            if self.sugar_level > 150:
                return 'insulin'
            elif self.sugar_level > 130:
                return 'insuline'
            elif 80 <= self.sugar_level <= 130:
                return 'stop'  # No intervention needed before meals
            elif self.sugar_level < 80:
                return 'nutrient'
                
        # Night time (22-6): Conservative approach
        elif hour_of_day >= 22 or hour_of_day < 6:
            if self.sugar_level > 180:
                return 'insuline'  # Gentler approach at night
            elif self.sugar_level > 160:
                return 'stop'  # Monitor only
            elif 70 <= self.sugar_level <= 160:
                return 'stop'
            elif self.sugar_level < 70:
                return 'fruits'
                
        # Regular hours (10-12, 13-18, 19-22): Standard treatment
        else:
            if self.sugar_level > 160:
                return 'insulin'
            elif self.sugar_level > 130:
                return 'insuline'
            elif 80 <= self.sugar_level <= 130:
                return 'stop'
            elif 60 <= self.sugar_level < 80:
                return 'fruits'
            else:
                return 'nutrient'
    
    def _start_movement_to_target(self, target_treatment):
        """Start smooth movement to target treatment location."""
        target_positions = {
            'insulin': (0, 0),    # Top-left
            'insuline': (0, 6),   # Top-right
            'stop': (6, 0),       # Bottom-left
            'candy': (6, 6),      # Bottom-right
            'fruits': (0, 3),     # Top-center
            'nutrient': (6, 3)    # Bottom-center
        }
        
        if target_treatment in target_positions:
            self.target_pos = list(target_positions[target_treatment])
        else:
            self.target_pos = [3, 3]  # Default to center
        
        # Verify the target position matches the grid items
        target_tuple = tuple(self.target_pos)
        if target_tuple in self.grid_items:
            expected_item = self.grid_items[target_tuple]
            if expected_item != target_treatment:
                print(f"WARNING: Position mismatch! Expected {target_treatment} at {target_tuple}, but found {expected_item}")
        
        # Create movement path (step by step)
        self.movement_path = self._calculate_path(self.agent_pos, self.target_pos)
        self.current_path_index = 0
        self.is_moving = True
        self.movement_start_time = time.time()
        
        print(f"\nAgent starting movement from {self.agent_pos} to {self.target_pos} for {target_treatment}")
        print(f"Target treatment position: {target_positions[target_treatment]} (row, col)")
        print(f"Path length: {len(self.movement_path)} steps (will take 4 seconds)")
        print(f"Verification: Grid at {target_tuple} contains '{self.grid_items.get(target_tuple, 'EMPTY')}'")
        
        # Show the planned path
        if self.movement_path:
            path_str = " -> ".join([str(pos) for pos in self.movement_path])
            print(f"Planned path: {self.agent_pos} -> {path_str}")
    
    def _calculate_path(self, start, end):
        """Calculate step-by-step path from start to end position."""
        path = []
        current = start.copy()
        
        # Move row-wise first, then column-wise (Manhattan path)
        while current[0] != end[0]:
            if current[0] < end[0]:
                current[0] += 1
            else:
                current[0] -= 1
            path.append(current.copy())
        
        while current[1] != end[1]:
            if current[1] < end[1]:
                current[1] += 1
            else:
                current[1] -= 1
            path.append(current.copy())
        
        return path
    
    def _update_movement_animation(self):
        """Update agent position during smooth movement animation."""
        current_time = time.time()
        elapsed = current_time - self.movement_start_time
        
        # Moving to treatment location (4 seconds)
        if len(self.movement_path) == 0:
            # No path means we're already at target, finish movement phase
            self._finish_movement_to_treatment()
            return
        
        # Calculate how many steps should be completed by now
        steps_per_second = len(self.movement_path) / self.movement_duration
        target_step = int(elapsed * steps_per_second)
        
        # Move to the appropriate step in the path
        if target_step < len(self.movement_path):
            # Update agent position to current step
            old_pos = self.agent_pos.copy()
            self.agent_pos = self.movement_path[target_step].copy()
            self.current_path_index = target_step
            
            # Show movement progress (only when position actually changes)
            if old_pos != self.agent_pos:
                pos_tuple = tuple(self.agent_pos)
                treatment_here = self.grid_items.get(pos_tuple, 'empty')
                print(f"Agent moving step-by-step to {self.agent_pos} (contains: {treatment_here})")
            
            # Calculate movement progress percentage
            progress = (target_step + 1) / len(self.movement_path)
            self.move_progress = progress
            
        else:
            # Movement complete - agent has reached the TARGET TREATMENT
            self.agent_pos = self.target_pos.copy()
            self.move_progress = 1.0
            
            # Verify we're at the correct treatment
            pos_tuple = tuple(self.agent_pos)
            treatment_here = self.grid_items.get(pos_tuple, 'EMPTY')
            print(f"Agent reached TARGET treatment at {self.agent_pos} (treatment: {treatment_here})")
            
            # Complete the movement cycle immediately
            self._finish_movement_to_treatment()
    
    def _calculate_position_reward(self):
        """Calculate reward for current position during movement."""
        pos_tuple = tuple(self.agent_pos)
        
        # Empty grid = 0 reward
        if pos_tuple not in self.grid_items:
            return 0
        
        # Treatment grid - check if it's correct
        item_type = self.grid_items[pos_tuple]
        return self._get_treatment_reward(item_type)
    
    def _end_cycle_immediately(self):
        """End current cycle immediately and return to center."""
        self.is_moving = False
        
        # Apply final treatment effect if on treatment cell
        self._apply_treatment()
        
        # Add cycle reward to total
        self.total_reward += self.current_cycle_reward
        
        # Flash current cell
        self.last_action_cell = tuple(self.agent_pos)
        self.action_flash_start = time.time()
        
        print(f"Cycle ended at {self.agent_pos}. Cycle reward: {self.current_cycle_reward}")
        
        # Instantly return to center
        self.agent_pos = [3, 3]
        self.cycle_complete = True
        
        print(f"Agent instantly returned to center. Ready for next cycle.")
    
    def _finish_movement_to_treatment(self):
        """Complete the movement to treatment, apply reward, and instantly return to center."""
        self.is_moving = False
        
        # CRITICAL: Ensure agent is at exact target position
        self.agent_pos = self.target_pos.copy()
        
        # Verify agent is at the correct treatment location
        pos_tuple = tuple(self.agent_pos)
        actual_treatment = self.grid_items.get(pos_tuple, 'EMPTY')
        
        # Flash action cell to show treatment was applied
        self.last_action_cell = pos_tuple
        self.action_flash_start = time.time()
        
        # Apply treatment effect
        self._apply_treatment()
        
        # Calculate and apply reward for this ONE treatment only
        if pos_tuple in self.grid_items:
            treatment_reward = self._get_treatment_reward(actual_treatment)
            self.current_cycle_reward = treatment_reward  # Only one reward per cycle
            self.total_reward += treatment_reward
            
            print(f"TREATMENT COMPLETED:")
            print(f"   Agent position: {self.agent_pos}")
            print(f"   Treatment: '{actual_treatment}'")
            print(f"   Reward: {treatment_reward}")
            print(f"   Total reward: {self.total_reward}")
        
        # Brief pause to show the overlap and treatment (0.3s to see the visual)
        time.sleep(0.3)
        
        # Instantly return to center after treatment
        self.agent_pos = [3, 3]
        self.cycle_complete = True
        
        print(f"Treatment '{actual_treatment}' applied successfully!")
        print(f"Agent instantly returned to center {self.agent_pos}. Ready for next cycle.")
        print("-" * 60)  # Separator for clarity
    
    def _move_agent_to_target(self, target_treatment):
        """Move agent directly to the target treatment location."""
        target_positions = {
            'insulin': (0, 0),    # Top-left
            'insuline': (0, 6),   # Top-right
            'stop': (6, 0),       # Bottom-left
            'candy': (6, 6),      # Bottom-right
            'fruits': (0, 3),     # Top-center
            'nutrient': (6, 3)    # Bottom-center
        }
        
        if target_treatment in target_positions:
            target_pos = target_positions[target_treatment]
            self.agent_pos = [target_pos[0], target_pos[1]]
        else:
            # Default to center if target not found
            self.agent_pos = [3, 3]
    
    def _gradual_return_to_center(self):
        """Gradually move agent back to center between decision intervals."""
        current_time = time.time()
        elapsed_since_decision = current_time - self.last_decision_time
        
        # Start returning to center after staying at treatment for 4 seconds
        if elapsed_since_decision >= 4.0:
            center = [3, 3]
            
            # Move one step closer to center if not already there
            if self.agent_pos != center:
                # Move towards center gradually
                if self.agent_pos[0] < center[0]:
                    self.agent_pos[0] += 1
                elif self.agent_pos[0] > center[0]:
                    self.agent_pos[0] -= 1
                elif self.agent_pos[1] < center[1]:
                    self.agent_pos[1] += 1
                elif self.agent_pos[1] > center[1]:
                    self.agent_pos[1] -= 1
    
    def _log_action_intelligent(self, treatment, reward):
        """Log intelligent action details for debugging."""
        treatment_names = {
            'insulin': 'High Insulin Dose',
            'insuline': 'Low Insulin Dose',
            'stop': 'Monitor Only',
            'fruits': 'Fruits (Light Sugar)',
            'nutrient': 'Nutrients (Medium Sugar)',
            'candy': 'Candy (Emergency Sugar)'
        }
        
        treatment_name = treatment_names.get(treatment, 'Unknown')
        
        print(f"[{self.get_formatted_time()}] Agent chose {treatment_name} at {self.agent_pos}, "
              f"Sugar: {self.sugar_level:.1f} mg/dL, Reward: {reward:.1f}, Total: {self.total_reward:.1f}")
    
    def get_current_action_text(self):
        """Get current action description based on agent position."""
        pos_tuple = tuple(self.agent_pos)
        
        if pos_tuple in self.grid_items:
            item_type = self.grid_items[pos_tuple]
            action_map = {
                'insulin': 'High Insulin Dose',
                'insuline': 'Low Insulin Dose', 
                'stop': 'Monitor Only',
                'fruits': 'Fruits (Light Sugar)',
                'nutrient': 'Nutrients (Medium Sugar)',
                'candy': 'Candy (Emergency Sugar)'
            }
            return action_map.get(item_type, 'No action')
        
        return 'Moving to Center'
    
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
    """Run a simulation with intelligent agent movements."""
    env = DiabetesTreatmentEnv()
    obs, _ = env.reset()
    
    print("Starting Intelligent Diabetes Treatment Simulation...")
    print("Every 3 in-game hours (7.5 seconds real-time):")
    print("  1. Agent thinks and decides optimal treatment (0.5s)")
    print("  2. Agent moves step-by-step to exact treatment location (4s)")
    print("  3. Agent stays at treatment location (1.5s)")
    print("  4. Rewards: +10 correct treatment, -10 wrong treatment, 0 empty grid")
    print("  5. Cycle ends when +/-10 reward reached OR target reached")
    print("  6. Agent instantly returns to center")
    print("Simulation duration: 60 seconds (24 hours game time)")
    print("Close window to stop simulation\n")
    
    running = True
    
    while running:
        # The agent now makes intelligent decisions automatically
        # We still need to call step() but the action parameter is ignored
        action = 0  # Dummy action, not used in intelligent mode
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Render
        running = env.render()
        if not running:
            break
        
        # Check if simulation should end
        if done:
            print(f"\nEpisode ended: Sugar level {env.sugar_level:.1f} mg/dL")
            break
        
        # Small delay to control frame rate
        time.sleep(0.1)
    
    print(f"\nSimulation Results:")
    print(f"   Final sugar level: {env.sugar_level:.1f} mg/dL")
    print(f"   Total reward: {env.total_reward:.1f}")
    print(f"   Patient status: {env.get_patient_status()}")
    print(f"   Treatment cycles completed: {env.step_count}")
    
    # Save screenshot
    if env.renderer:
        env.renderer.save_screenshot("diabetes_final_state.png")
    
    env.close()


if __name__ == "__main__":
    run_random_simulation()