"""
custom_env.py - Diabetes Treatment Simulation Environment

This module creates a custom environment for diabetes management simulation.
It handles the core game logic, state management, and glucose dynamics.
"""

import numpy as np
import random
import time
from typing import Dict, Tuple, List
from enum import Enum

class GlucoseState(Enum):
    """Enumeration for different glucose level states"""
    VERY_LOW = "very_low"      # < 50
    LOW = "low"                # 50-70
    NORMAL = "normal"          # 70-120
    ELEVATED = "elevated"      # 120-180
    HIGH = "high"              # 180-250
    VERY_HIGH = "very_high"    # > 250

class TeddyExpression(Enum):
    """Enumeration for teddy bear expressions"""
    HAPPY = "happy"
    NEUTRAL = "neutral"
    DIZZY = "dizzy"
    GRUMPY = "grumpy"
    VERY_SICK = "very_sick"

class DiabetesEnvironment:
    """
    Core environment class for diabetes treatment simulation.
    Manages glucose levels, insulin effects, meal events, and scoring.
    """
    
    def __init__(self, 
                 initial_glucose: float = 100.0,
                 time_multiplier: float = 144.0,
                 start_hour: int = 6):
        """
        Initialize the diabetes simulation environment.
        
        Args:
            initial_glucose: Starting glucose level (mg/dL)
            time_multiplier: Game time speed (1 real second = time_multiplier/60 game minutes)
            start_hour: Starting hour of the day (0-23)
        """
        # Core state variables
        self.glucose_level = initial_glucose
        self.target_glucose = initial_glucose
        self.glucose_history = [initial_glucose]
        
        # Insulin management
        self.active_insulin = 0.0
        self.insulin_history = []
        self.last_insulin_time = 0
        
        # Time management
        self.start_time = time.time()
        self.time_multiplier = time_multiplier
        self.start_hour = start_hour
        self.current_game_time = start_hour * 3600  # Convert to seconds
        
        # Meal system
        self.last_meal_time = 0
        self.next_meal_time = self._schedule_next_meal()
        self.meal_history = []
        
        # Scoring system
        self.score = 0
        self.time_in_range = 0
        self.total_time = 0
        
        # Glucose dynamics parameters
        self.base_glucose_rise_rate = 0.3  # mg/dL per update
        self.insulin_sensitivity = 1.0     # Individual sensitivity factor
        self.carb_ratio = 12.0            # Grams of carbs per unit of insulin
        self.correction_factor = 50.0      # mg/dL drop per unit of insulin
        
        # State tracking
        self.glucose_trend = "steady"
        self.last_glucose_values = [initial_glucose] * 5
        
        # Simulation parameters
        self.update_frequency = 60  # Updates per second
        self.glucose_noise_factor = 0.5  # Random variation
        
    def reset(self) -> Dict:
        """Reset the environment to initial state"""
        self.__init__()
        return self.get_state()
    
    def step(self, insulin_units: int = 0) -> Tuple[Dict, float, bool]:
        """
        Advance the simulation by one step.
        
        Args:
            insulin_units: Units of insulin to administer (0, 2, 5, or 10)
            
        Returns:
            Tuple of (new_state, reward, done)
        """
        # Apply insulin if administered
        if insulin_units > 0:
            self._administer_insulin(insulin_units)
        
        # Update time
        self._update_time()
        
        # Check for meal events
        self._check_meal_events()
        
        # Update glucose dynamics
        self._update_glucose_dynamics()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update statistics
        self._update_statistics()
        
        # Check if simulation should end (24 hours passed)
        done = self._check_done()
        
        return self.get_state(), reward, done
    
    def _administer_insulin(self, units: int):
        """Apply insulin effect to the system"""
        if units in [0, 2, 5, 10]:
            self.active_insulin += units
            self.insulin_history.append({
                'time': self.current_game_time,
                'units': units,
                'glucose_before': self.glucose_level
            })
            
            # Immediate glucose drop (rapid-acting insulin)
            immediate_drop = units * (self.correction_factor / 4)  # 25% immediate effect
            self.target_glucose -= immediate_drop
            
            self.last_insulin_time = self.current_game_time
    
    def _update_time(self):
        """Update game time based on real time elapsed"""
        current_real_time = time.time()
        elapsed_real_time = current_real_time - self.start_time
        
        # Convert real time to game time
        game_seconds_elapsed = elapsed_real_time * self.time_multiplier
        self.current_game_time = (self.start_hour * 3600) + game_seconds_elapsed
        self.total_time += 1
    
    def _check_meal_events(self):
        """Check if a meal event should occur"""
        if self.current_game_time >= self.next_meal_time:
            self._trigger_meal_event()
            self.next_meal_time = self._schedule_next_meal()
    
    def _trigger_meal_event(self):
        """Trigger a random meal event"""
        # Random meal size (carbohydrates in grams)
        carb_amount = random.uniform(30, 80)
        
        # Calculate glucose rise (simplified carb counting)
        glucose_rise = carb_amount * 3  # ~3 mg/dL per gram of carbs
        
        self.target_glucose += glucose_rise
        
        self.meal_history.append({
            'time': self.current_game_time,
            'carbs': carb_amount,
            'glucose_impact': glucose_rise
        })
    
    def _schedule_next_meal(self) -> float:
        """Schedule the next meal event"""
        # Random interval between 30-90 minutes (in game time)
        interval_minutes = random.uniform(30, 90)
        interval_seconds = interval_minutes * 60
        return self.current_game_time + interval_seconds
    
    def _update_glucose_dynamics(self):
        """Update glucose level based on various factors"""
        # Natural glucose rise (liver glucose production)
        self.target_glucose += self.base_glucose_rise_rate
        
        # Insulin effect (ongoing action)
        if self.active_insulin > 0:
            # Insulin decay and glucose reduction
            insulin_effect = self.active_insulin * 2.5  # Units per hour effect
            glucose_drop = insulin_effect * (self.correction_factor / 3600)  # Per second
            
            self.target_glucose -= glucose_drop
            
            # Insulin decay (3-4 hour duration for rapid-acting)
            decay_rate = 0.0003  # Per update cycle
            self.active_insulin = max(0, self.active_insulin - decay_rate)
        
        # Add some physiological noise
        noise = random.gauss(0, self.glucose_noise_factor)
        self.target_glucose += noise
        
        # Smooth transition to target glucose
        glucose_diff = self.target_glucose - self.glucose_level
        self.glucose_level += glucose_diff * 0.05  # Smooth transition
        
        # Physiological bounds
        self.glucose_level = max(20, min(500, self.glucose_level))
        self.target_glucose = max(20, min(500, self.target_glucose))
        
        # Update glucose history and trend
        self._update_glucose_trend()
    
    def _update_glucose_trend(self):
        """Update glucose trend based on recent values"""
        self.glucose_history.append(self.glucose_level)
        if len(self.glucose_history) > 100:  # Keep last 100 values
            self.glucose_history.pop(0)
        
        # Update recent values for trend calculation
        self.last_glucose_values.append(self.glucose_level)
        if len(self.last_glucose_values) > 5:
            self.last_glucose_values.pop(0)
        
        # Calculate trend
        if len(self.last_glucose_values) >= 3:
            recent_change = self.last_glucose_values[-1] - self.last_glucose_values[-3]
            if recent_change > 5:
                self.glucose_trend = "rising"
            elif recent_change < -5:
                self.glucose_trend = "falling"
            else:
                self.glucose_trend = "steady"
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current glucose level"""
        glucose = self.glucose_level
        
        if 70 <= glucose <= 120:  # Target range
            reward = 10
            self.time_in_range += 1
        elif 50 <= glucose < 70 or 120 < glucose <= 180:  # Acceptable range
            reward = -5
        elif 180 < glucose <= 250:  # High but manageable
            reward = -15
        elif glucose > 250:  # Dangerously high
            reward = -25
        else:  # glucose < 50 - Dangerously low
            reward = -30
        
        return reward
    
    def _update_statistics(self):
        """Update running statistics"""
        self.score += self._calculate_reward()
    
    def _check_done(self) -> bool:
        """Check if simulation should end"""
        # End after 24 game hours
        game_hours_elapsed = (self.current_game_time - (self.start_hour * 3600)) / 3600
        return game_hours_elapsed >= 24
    
    def get_state(self) -> Dict:
        """Get current state of the environment"""
        return {
            'glucose_level': self.glucose_level,
            'target_glucose': self.target_glucose,
            'glucose_state': self._get_glucose_state(),
            'teddy_expression': self._get_teddy_expression(),
            'glucose_trend': self.glucose_trend,
            'active_insulin': self.active_insulin,
            'score': self.score,
            'time_in_range_percent': (self.time_in_range / max(1, self.total_time)) * 100,
            'game_time': self._get_game_time_string(),
            'game_time_raw': self.current_game_time,
            'meal_pending': self.current_game_time >= (self.next_meal_time - 300),  # 5 min warning
            'glucose_history': self.glucose_history.copy(),
            'recent_meals': self.meal_history[-3:],  # Last 3 meals
            'recent_insulin': self.insulin_history[-3:]  # Last 3 insulin doses
        }
    
    def _get_glucose_state(self) -> GlucoseState:
        """Determine glucose state category"""
        glucose = self.glucose_level
        
        if glucose < 50:
            return GlucoseState.VERY_LOW
        elif glucose < 70:
            return GlucoseState.LOW
        elif glucose <= 120:
            return GlucoseState.NORMAL
        elif glucose <= 180:
            return GlucoseState.ELEVATED
        elif glucose <= 250:
            return GlucoseState.HIGH
        else:
            return GlucoseState.VERY_HIGH
    
    def _get_teddy_expression(self) -> TeddyExpression:
        """Determine teddy bear expression based on glucose"""
        glucose = self.glucose_level
        
        if glucose < 70:
            return TeddyExpression.DIZZY
        elif glucose > 250:
            return TeddyExpression.VERY_SICK
        elif glucose > 180:
            return TeddyExpression.GRUMPY
        elif 70 <= glucose <= 120:
            return TeddyExpression.HAPPY
        else:
            return TeddyExpression.NEUTRAL
    
    def _get_game_time_string(self) -> str:
        """Convert game time to readable string"""
        total_seconds = int(self.current_game_time % (24 * 3600))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        
        am_pm = "AM" if hours < 12 else "PM"
        display_hours = hours if hours <= 12 else hours - 12
        if display_hours == 0:
            display_hours = 12
            
        return f"{display_hours}:{minutes:02d} {am_pm}"
    
    def get_statistics(self) -> Dict:
        """Get detailed statistics about the simulation"""
        return {
            'total_score': self.score,
            'time_in_range_percent': (self.time_in_range / max(1, self.total_time)) * 100,
            'total_insulin_used': sum([dose['units'] for dose in self.insulin_history]),
            'total_meals': len(self.meal_history),
            'average_glucose': np.mean(self.glucose_history) if self.glucose_history else 0,
            'glucose_std': np.std(self.glucose_history) if len(self.glucose_history) > 1 else 0,
            'current_glucose_state': self._get_glucose_state(),
            'simulation_time_hours': (self.current_game_time - (self.start_hour * 3600)) / 3600
        }
    
    def simulate_rapid_test(self, insulin_sequence: List[int], steps: int = 1000) -> List[Dict]:
        """
        Run a rapid simulation with a sequence of insulin actions for testing.
        
        Args:
            insulin_sequence: List of insulin units to administer at each step
            steps: Number of simulation steps to run
            
        Returns:
            List of states at each step
        """
        states = []
        
        for i in range(steps):
            insulin_units = insulin_sequence[i % len(insulin_sequence)] if insulin_sequence else 0
            state, reward, done = self.step(insulin_units)
            states.append(state.copy())
            
            if done:
                break
                
        return states

# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = DiabetesEnvironment()
    
    print("=== Diabetes Simulation Environment Test ===")
    print(f"Initial state: {env.get_state()}")
    
    # Run a few simulation steps
    for i in range(10):
        # Randomly administer insulin sometimes
        insulin = random.choice([0, 0, 0, 2, 5]) if random.random() < 0.3 else 0
        
        state, reward, done = env.step(insulin)
        
        print(f"Step {i+1}:")
        print(f"  Glucose: {state['glucose_level']:.1f} mg/dL ({state['glucose_state'].value})")
        print(f"  Teddy: {state['teddy_expression'].value}")
        print(f"  Trend: {state['glucose_trend']}")
        print(f"  Score: {state['score']}")
        print(f"  Time: {state['game_time']}")
        print(f"  Reward: {reward}")
        print()
        
        if done:
            print("Simulation completed!")
            break
    
    # Print final statistics
    print("=== Final Statistics ===")
    stats = env.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")