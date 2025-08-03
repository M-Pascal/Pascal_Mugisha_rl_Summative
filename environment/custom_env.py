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
        reward = 0
        
        # Handle ongoing movement animation
        if self.is_moving:
            self._update_movement_animation()
            # Apply continuous health state reward during movement
            reward = self._get_health_state_reward() * 0.1  # Scaled down for continuous updates
        # Only allow new decisions when cycle is complete and 3 hours have passed
        elif self.cycle_complete and current_time - self.last_decision_time >= self.decision_interval:
            # Show thinking overlay
            self.is_thinking = True
            self.thinking_start = current_time
            
            # Wait 0.5 seconds for thinking animation
            time.sleep(0.5)
            
            # Determine optimal treatment based on sugar level and time
            optimal_treatment = self._get_optimal_action()
            
            # Calculate old position for reward calculation
            old_pos = self.agent_pos.copy()
            
            # Start movement to target location
            self._start_movement_to_target(optimal_treatment)
            
            # Calculate reward for this decision
            reward = self._calculate_reward(old_pos, self.agent_pos)
            
            # Track cumulative reward
            self.total_reward += reward
            self.current_cycle_reward += reward
            
            self.last_decision_time = current_time
            self.step_count += 1
            self.is_thinking = False
            self.cycle_complete = False  # Mark cycle as in progress
            
            # Log action for debugging
            self._log_action_intelligent(optimal_treatment, reward)
        
        # Update simulation time continuously
        elapsed = current_time - self.start_time
        self.simulation_time = min((elapsed / self.simulation_duration) * 24, 24)
        
        # Update sugar level continuously
        self._update_sugar_level()
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Add episode completion bonus/penalty
        if done:
            completion_reward = self._get_episode_completion_reward()
            reward += completion_reward
            self.total_reward += completion_reward
        
        # Get current observation
        obs = self._get_observation()
        
        return obs, reward, done, False, {'total_reward': self.total_reward}
    
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
        """Calculate sophisticated reward based on diabetes management quality."""
        reward = 0
        
        # Core health state rewards (most important)
        reward += self._get_health_state_reward()
        
        # Treatment appropriateness rewards
        new_pos_tuple = tuple(new_pos)
        if new_pos_tuple in self.grid_items:
            item_type = self.grid_items[new_pos_tuple]
            reward += self._get_treatment_appropriateness_reward(item_type)
        
        # Time efficiency rewards
        reward += self._get_urgency_reward(new_pos)
        
        # Sugar level stability rewards
        reward += self._get_stability_reward()
        
        # Critical safety penalties
        reward += self._get_safety_penalties()
        
        return reward
    
    def _get_health_state_reward(self):
        """Primary reward based on current health state quality."""
        # Optimal range (80-120 mg/dL) - highest reward
        if 80 <= self.sugar_level <= 120:
            return 50  # High reward for maintaining good control
        
        # Acceptable range (70-140 mg/dL) - moderate reward
        elif 70 <= self.sugar_level <= 140:
            return 20  # Good management
        
        # Mild concern ranges (60-70 or 140-180 mg/dL)
        elif 60 <= self.sugar_level <= 180:
            return 5  # Slight positive for not being dangerous
        
        # Warning ranges (50-60 or 180-250 mg/dL)
        elif 50 <= self.sugar_level <= 250:
            return -10  # Negative for poor control
        
        # Dangerous ranges (below 50 or above 250 mg/dL)
        else:
            return -50  # Heavy penalty for dangerous levels
    
    def _get_treatment_appropriateness_reward(self, item_type):
        """Reward based on how appropriate the chosen treatment is."""
        current_sugar = self.sugar_level
        
        # Calculate the expected effectiveness of this treatment
        expected_effect = self.treatment_effects[item_type]
        target_range_center = 100  # Ideal sugar level
        
        # Predict where sugar level would go with this treatment
        predicted_level = current_sugar + (expected_effect * 2)  # Estimate full effect
        
        # Distance from target after treatment
        distance_from_target_before = abs(current_sugar - target_range_center)
        distance_from_target_after = abs(predicted_level - target_range_center)
        
        # Reward improvement in getting closer to target
        improvement = distance_from_target_before - distance_from_target_after
        
        # Scale reward based on improvement
        if improvement > 20:
            return 30  # Excellent treatment choice
        elif improvement > 10:
            return 20  # Good treatment choice
        elif improvement > 0:
            return 10  # Mild improvement
        elif improvement == 0:
            return 0   # No change
        else:
            return -20  # Makes things worse
    
    def _get_urgency_reward(self, pos):
        """Reward based on how quickly urgent situations are addressed."""
        # Critical urgency - immediate treatment needed
        if self.sugar_level < 60 or self.sugar_level > 200:
            # Check if agent is moving toward appropriate treatment
            if self.sugar_level < 60:  # Need sugar fast
                sugar_positions = [(6, 6), (0, 3), (6, 3)]  # candy, fruits, nutrient
                closest_dist = min([self._manhattan_distance(pos, spos) for spos in sugar_positions])
                if closest_dist == 0:  # At treatment location
                    return 25  # High reward for quick critical response
                elif closest_dist <= 2:
                    return 10  # Moving toward treatment
                else:
                    return -15  # Too slow for critical situation
            
            else:  # Need insulin fast (sugar > 200)
                insulin_positions = [(0, 0), (0, 6)]  # insulin, insuline
                closest_dist = min([self._manhattan_distance(pos, ipos) for ipos in insulin_positions])
                if closest_dist == 0:
                    return 25  # Quick critical response
                elif closest_dist <= 2:
                    return 10  # Moving toward treatment
                else:
                    return -15  # Too slow
        
        # Moderate urgency
        elif self.sugar_level < 70 or self.sugar_level > 150:
            return 5  # Small bonus for addressing moderate issues
        
        return 0  # No urgency bonus needed
    
    def _get_stability_reward(self):
        """Reward for maintaining stable sugar levels over time."""
        if len(self.sugar_history) < 3:
            return 0
        
        # Look at recent sugar level changes
        recent_levels = self.sugar_history[-3:]
        
        # Calculate variance (lower is better)
        variance = np.var(recent_levels)
        
        # Reward stability
        if variance < 25:  # Very stable
            return 15
        elif variance < 100:  # Moderately stable
            return 8
        elif variance < 400:  # Somewhat unstable
            return 0
        else:  # Very unstable
            return -10
    
    def _get_safety_penalties(self):
        """Heavy penalties for dangerous situations."""
        penalty = 0
        
        # Extreme danger zones
        if self.sugar_level < 40:
            penalty -= 100  # Severe hypoglycemia - life threatening
        elif self.sugar_level > 300:
            penalty -= 100  # Severe hyperglycemia - life threatening
        
        # Extended time in dangerous ranges
        if len(self.sugar_history) >= 5:
            recent_dangerous = sum(1 for level in self.sugar_history[-5:] 
                                 if level < 60 or level > 200)
            if recent_dangerous >= 3:
                penalty -= 30  # Extended time in danger zone
        
        return penalty
    
    def _get_episode_completion_reward(self):
        """Calculate final reward based on overall episode performance."""
        if len(self.sugar_history) < 5:
            return 0
        
        # Calculate time spent in different ranges
        total_steps = len(self.sugar_history)
        optimal_time = sum(1 for level in self.sugar_history if 80 <= level <= 120)
        good_time = sum(1 for level in self.sugar_history if 70 <= level <= 140)
        dangerous_time = sum(1 for level in self.sugar_history if level < 60 or level > 200)
        critical_time = sum(1 for level in self.sugar_history if level < 40 or level > 300)
        
        # Calculate percentages
        optimal_pct = optimal_time / total_steps
        good_pct = good_time / total_steps
        dangerous_pct = dangerous_time / total_steps
        critical_pct = critical_time / total_steps
        
        # Base completion reward
        completion_reward = 0
        
        # Reward for time in optimal range
        if optimal_pct > 0.7:
            completion_reward += 100  # Excellent diabetes management
        elif optimal_pct > 0.5:
            completion_reward += 50   # Good management
        elif optimal_pct > 0.3:
            completion_reward += 20   # Fair management
        
        # Reward for staying in acceptable range
        if good_pct > 0.8:
            completion_reward += 50   # Very good overall control
        elif good_pct > 0.6:
            completion_reward += 25   # Good overall control
        
        # Penalties for dangerous time
        if dangerous_pct > 0.3:
            completion_reward -= 100  # Too much time in danger
        elif dangerous_pct > 0.1:
            completion_reward -= 50   # Some dangerous periods
        
        # Heavy penalty for critical situations
        if critical_pct > 0.05:  # More than 5% in critical range
            completion_reward -= 200  # Very poor management
        
        # Bonus for completing full 24-hour simulation
        if self.simulation_time >= 23.5:  # Almost full day
            completion_reward += 50   # Completion bonus
        
        return completion_reward
    
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
        """Check if episode should end with more sophisticated conditions."""
        # End if simulation time reaches 24 hours (successful completion)
        if self.simulation_time >= 24:
            return True
        
        # End if sugar level is life-threatening for extended period
        if self.sugar_level < 30 or self.sugar_level > 350:
            return True  # Immediate danger - emergency stop
        
        # End if sugar level has been critically dangerous for too long
        if len(self.sugar_history) >= 3:
            recent_critical = sum(1 for level in self.sugar_history[-3:] 
                                if level < 40 or level > 300)
            if recent_critical >= 3:
                return True  # 3 consecutive critical readings
        
        # End if agent is stuck (no progress for too long)
        if self.step_count > 20:  # After reasonable number of steps
            # Check if sugar level management is completely failing
            if len(self.sugar_history) >= 10:
                recent_levels = self.sugar_history[-10:]
                # If all recent levels are outside acceptable range
                all_bad = all(level < 60 or level > 220 for level in recent_levels)
                if all_bad:
                    return True  # Complete management failure
        
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
        
        # Move row-wise first, then column-wise
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
        return self._get_treatment_appropriateness_reward(item_type)
    
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
            treatment_reward = self._get_treatment_appropriateness_reward(actual_treatment)
            self.current_cycle_reward = treatment_reward
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
        
        # Call the renderer and check if we should quit
        result = self.renderer.render()
        if result is False:
            return None  # Quit event received
        
        # Return the surface for GIF recording
        return self.renderer.screen
    
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