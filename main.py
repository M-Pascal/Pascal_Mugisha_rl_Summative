#!/usr/bin/env python3
"""
Main Application: DQN-Guided Diabetes Treatment Simulation

This application integrates a trained DQN model into the existing environment
to guide the agent's decisions instead of using random or hardcoded actions.
The simulation displays the GUI to show intelligent treatment decisions in real-time.
"""

import sys
import os
import time

# Add environment directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'environment'))

# Import required libraries
try:
    from stable_baselines3 import DQN
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

class DQNGuidedEnvironment(DiabetesTreatmentEnv):
    """
    Enhanced environment that uses DQN model for decision making
    instead of the hardcoded logic.
    """
    
    def __init__(self, model_path="training/models/dqn/dqn_diabetes_final"):
        """Initialize with DQN model integration."""
        # Change to environment directory to ensure image paths work correctly
        original_dir = os.getcwd()
        env_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'environment')
        os.chdir(env_dir)
        
        try:
            super().__init__()
        finally:
            # Change back to original directory
            os.chdir(original_dir)
        
        self.dqn_model = None
        self.model_path = model_path
        self.dqn_enabled = False
        
        # Load DQN model
        self.load_dqn_model()
        
        # Statistics tracking
        self.decisions_made = 0
        self.dqn_decisions = 0
        self.fallback_decisions = 0
        
        # Verify image loading
        self._verify_images()
    
    def _verify_images(self):
        """Verify that images are available in the expected location."""
        env_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'environment')
        image_dir = os.path.join(env_dir, 'image')
        
        expected_images = ['doctor.png', 'insulin.png', 'candy.png', 'fruits.png', 
                          'nutrient.png', 'stop.png', 'patient.png', 'person.png', 'died.png']
        
        print(f"Checking images in: {image_dir}")
        missing_images = []
        found_images = []
        
        for img in expected_images:
            img_path = os.path.join(image_dir, img)
            if os.path.exists(img_path):
                found_images.append(img)
            else:
                missing_images.append(img)
        
        print(f"Found images: {found_images}")
        if missing_images:
            print(f"Missing images: {missing_images}")
            print("Note: Missing images will use colored squares as fallback")
        else:
            print("All required images found! Environment will display with proper graphics.")
    
    def load_dqn_model(self):
        """Load the trained DQN model."""
        model_file = f"{self.model_path}.zip"
        
        if not os.path.exists(model_file):
            print(f"DQN model not found at: {model_file}")
            print("Using fallback hardcoded logic. Train model with: cd training && python dqn_training.py")
            self.dqn_enabled = False
            return
        
        try:
            self.dqn_model = DQN.load(self.model_path)
            self.dqn_enabled = True
            print(f"DQN model loaded successfully! Agent will use AI guidance.")
        except Exception as e:
            print(f"Error loading DQN model: {e}")
            print("Using fallback hardcoded logic.")
            self.dqn_enabled = False
    
    def _get_optimal_action(self):
        """
        Override the original method to use DQN model for decision making.
        Falls back to original logic if DQN model is not available.
        """
        self.decisions_made += 1
        
        if self.dqn_enabled and self.dqn_model is not None:
            try:
                # Get current observation for DQN model
                obs = self._get_observation()
                
                # Get action from DQN model
                dqn_action, _states = self.dqn_model.predict(obs, deterministic=True)
                
                # Convert DQN action (0=Up, 1=Down, 2=Left, 3=Right) to treatment
                treatment = self._convert_dqn_action_to_treatment(dqn_action)
                
                self.dqn_decisions += 1
                
                print(f"DQN Decision: Action {dqn_action} -> {treatment} (Sugar: {self.sugar_level:.1f} mg/dL)")
                return treatment
                
            except Exception as e:
                print(f"DQN prediction error: {e}. Using fallback logic.")
                self.fallback_decisions += 1
                return self._get_fallback_action()
        else:
            # Use original hardcoded logic as fallback
            self.fallback_decisions += 1
            return self._get_fallback_action()
    
    def _convert_dqn_action_to_treatment(self, dqn_action):
        """
        Convert DQN action (0-3) to treatment based on current sugar level and action direction.
        This maps movement actions to appropriate treatments.
        """
        # Get current sugar level for context
        sugar = self.sugar_level
        
        # Map actions to treatments based on sugar level and movement direction
        if dqn_action == 0:  # Up movement
            if sugar > 150:
                return 'insulin'    # Go to top-left for strong insulin
            elif sugar > 120:
                return 'fruits'     # Go to top-center for fruits
            else:
                return 'insuline'   # Go to top-right for mild insulin
                
        elif dqn_action == 1:  # Down movement  
            if sugar < 80:
                return 'candy'      # Go to bottom-right for emergency sugar
            elif sugar < 100:
                return 'nutrient'   # Go to bottom-center for nutrients
            else:
                return 'stop'       # Go to bottom-left for no treatment
                
        elif dqn_action == 2:  # Left movement
            if sugar > 140:
                return 'insulin'    # Go to top-left for strong insulin
            else:
                return 'stop'       # Go to bottom-left for no treatment
                
        elif dqn_action == 3:  # Right movement
            if sugar < 90:
                return 'candy'      # Go to bottom-right for emergency sugar
            else:
                return 'insuline'   # Go to top-right for mild insulin
        
        return 'stop'  # Default fallback
    
    def _get_fallback_action(self):
        """Original hardcoded logic as fallback."""
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
            elif 80 <= self.sugar_level <= 160:
                return 'stop'
            elif self.sugar_level < 80:
                return 'fruits'
                
        # Default day time (10-12, 13-18, 19-22)
        else:
            if self.sugar_level > 160:
                return 'insulin'
            elif self.sugar_level > 130:
                return 'insuline'
            elif 80 <= self.sugar_level <= 130:
                return 'stop'
            elif self.sugar_level < 80:
                return 'nutrient'
                
        return 'stop'  # Default fallback
    
    def render(self):
        """Override render to ensure proper image display."""
        # Ensure renderer is initialized in correct directory
        if self.renderer is None:
            original_dir = os.getcwd()
            env_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'environment')
            os.chdir(env_dir)
            
            try:
                from rendering import DiabetesRenderer
                self.renderer = DiabetesRenderer(self)
                print(f"Renderer initialized in directory: {os.getcwd()}")
            finally:
                os.chdir(original_dir)
        
        return self.renderer.render()
    
    def get_decision_stats(self):
        """Get statistics about decision making."""
        total = self.decisions_made
        if total == 0:
            return "No decisions made yet"
        
        dqn_percent = (self.dqn_decisions / total) * 100
        fallback_percent = (self.fallback_decisions / total) * 100
        
        return (f"Decisions: {total} total | "
                f"DQN: {self.dqn_decisions} ({dqn_percent:.1f}%) | "
                f"Fallback: {self.fallback_decisions} ({fallback_percent:.1f}%)")

def run_dqn_guided_simulation():
    """Run the simulation with DQN-guided environment."""
    print("DQN-Guided Diabetes Treatment Simulation")
    print("=" * 60)
    print("Agent decisions will be guided by trained DQN model")
    print("Watch the GUI for real-time intelligent treatment decisions")
    print("Goal: Maintain blood sugar levels between 80-120 mg/dL")
    print("=" * 60)
    
    # Create DQN-guided environment
    env = DQNGuidedEnvironment()
    
    if env.dqn_enabled:
        print("DQN model active - Agent will make intelligent decisions!")
    else:
        print("Using fallback logic - Train DQN model for better performance")
    
    print("\nStarting simulation...")
    print("GUI window will show the environment")
    print("Watch the agent navigate and make treatment decisions")
    print("Simulation runs for 60 seconds (24 in-game hours)")
    print("=" * 60)
    
    try:
        # Reset environment
        obs, _ = env.reset()
        
        # Track statistics
        start_time = time.time()
        step_count = 0
        sugar_readings = []
        
        print(f"\nInitial State:")
        print(f"   Sugar Level: {env.sugar_level:.1f} mg/dL")
        print(f"   Time: {env.simulation_time:.1f} hours")
        print(f"   Agent Position: {env.agent_pos}")
        print()
        
        # Run simulation
        while True:
            # Step the environment (it will use DQN for decisions automatically)
            obs, reward, done, truncated, info = env.step(0)  # Action parameter is ignored
            
            # Render the GUI
            env.render()
            
            # Track statistics
            step_count += 1
            sugar_readings.append(env.sugar_level)
            
            # Print periodic updates
            if step_count % 50 == 0:
                avg_sugar = sum(sugar_readings) / len(sugar_readings)
                status = "NORMAL" if 80 <= env.sugar_level <= 120 else "OUT_OF_RANGE"
                print(f"Step {step_count:3d}: Sugar {env.sugar_level:6.1f} mg/dL {status} | "
                      f"Time {env.simulation_time:4.1f}h | Avg {avg_sugar:5.1f}")
                print(f"   {env.get_decision_stats()}")
            
            # Small delay for smooth visualization
            time.sleep(0.1)
            
            # Check if simulation is complete
            if done or truncated:
                print("Simulation completed!")
                break
                
            # Safety check for long runs
            if time.time() - start_time > 120:  # 2 minutes max
                print("Time limit reached!")
                break
        
        # Final statistics
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)
        
        final_sugar = env.sugar_level
        avg_sugar = sum(sugar_readings) / len(sugar_readings)
        min_sugar = min(sugar_readings)
        max_sugar = max(sugar_readings)
        
        # Time in target range
        target_readings = [s for s in sugar_readings if 80 <= s <= 120]
        time_in_range = (len(target_readings) / len(sugar_readings)) * 100
        
        print(f"Blood Sugar Analysis:")
        print(f"   Final level: {final_sugar:.1f} mg/dL")
        print(f"   Average level: {avg_sugar:.1f} mg/dL")
        print(f"   Range: {min_sugar:.1f} - {max_sugar:.1f} mg/dL")
        print(f"   Time in target range (80-120): {time_in_range:.1f}%")
        print(f"   Total readings: {len(sugar_readings)}")
        
        print(f"\nDecision Analysis:")
        print(f"   {env.get_decision_stats()}")
        
        if env.dqn_enabled:
            print(f"   DQN model successfully guided {env.dqn_decisions} decisions!")
        
        # Performance assessment
        if time_in_range >= 70:
            print(f"\nEXCELLENT: {time_in_range:.1f}% time in target range!")
        elif time_in_range >= 50:
            print(f"\nGOOD: {time_in_range:.1f}% time in target range")
        else:
            print(f"\nNEEDS IMPROVEMENT: Only {time_in_range:.1f}% time in target range")
        
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    except Exception as e:
        print(f"\nSimulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        env.close()
        print("\nSimulation Complete!")

def main():
    """Main application entry point."""
    print("DQN-Guided Diabetes Treatment System")
    print("=" * 50)
    print("Starting simulation with DQN-guided intelligent agent...")
    print("GUI will display the environment with real-time treatment decisions")
    print("=" * 50)
    
    # Check dependencies quickly
    try:
        import pygame
    except ImportError:
        print("Pygame not available! Install with: pip install pygame")
        return
    
    # Run the simulation immediately
    run_dqn_guided_simulation()

if __name__ == "__main__":
    main()


