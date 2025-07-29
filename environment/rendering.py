import pygame
import numpy as np
import os
import math
import time

class DiabetesRenderer:
    """
    Pygame-based renderer for the Diabetes Treatment Environment.
    
    Displays the simulation in 1/4 screen size with balanced left (grid) and 
    right (info panel) sections, proper spacing, and animated elements.
    """
    
    def __init__(self, env):
        self.env = env
        
        # Get screen dimensions and calculate window size
        pygame.init()
        info = pygame.display.Info()
        screen_width = info.current_w
        screen_height = info.current_h
        
        # Set window to a larger size to accommodate all content
        self.window_width = screen_width // 2
        self.window_height = int(screen_height * 0.7)  # Increased from 50% to 70% of screen height
        
        # Layout parameters
        self.padding = 20
        self.section_spacing = 15  # Reduced from 25 to fit more content
        self.title_height = 50
        self.content_spacing = 40  # New: spacing between left and right content
        
        # Grid parameters (left side)
        available_grid_height = self.window_height - self.title_height - self.padding * 2
        self.cell_size = min(60, available_grid_height // 7)  # Max 60px, scaled to fit
        self.grid_width = 7 * self.cell_size
        self.grid_height = 7 * self.cell_size
        
        # Side panel (right side) - with better spacing
        self.side_panel_width = self.window_width - self.grid_width - self.padding * 2 - self.content_spacing
        self.side_panel_height = available_grid_height
        
        # Initialize pygame window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Diabetes Treatment Simulation")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (220, 220, 220)
        self.DARK_GRAY = (64, 64, 64)
        self.GREEN = (0, 180, 0)
        self.LIGHT_GREEN = (150, 255, 150)
        self.DARK_GREEN = (0, 120, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 100, 255)
        self.LIGHT_BLUE = (173, 216, 230)  # For flashing
        self.ORANGE = (255, 165, 0)
        self.YELLOW = (255, 255, 0)
        self.PURPLE = (128, 0, 128)
        
        # Range colors for sugar chart
        self.NORMAL_RANGE_COLOR = (200, 255, 200, 100)  # Light green with alpha
        self.PERFECT_RANGE_COLOR = (150, 255, 150, 150)  # Darker green with alpha
        self.DANGER_COLOR = (255, 200, 200, 100)  # Light red with alpha
        
        # Fonts
        self.title_font = pygame.font.Font(None, 32)
        self.subtitle_font = pygame.font.Font(None, 20)  # Reduced from 24 to 20
        self.info_font = pygame.font.Font(None, 18)      # Reduced from 20 to 18
        self.small_font = pygame.font.Font(None, 14)     # Reduced from 16 to 14
        
        # Load images
        self.images = self._load_images()
        
        # Positioning
        self.grid_x = self.padding
        self.grid_y = self.title_height + self.padding
        self.panel_x = self.grid_x + self.grid_width + self.content_spacing  # Updated to use content_spacing
        self.panel_y = self.grid_y
        
        # Chart dimensions - adjusted for smaller panel with space for Y-axis labels
        self.chart_width = self.side_panel_width - 45  # More space for Y-axis labels outside chart
        self.chart_height = 100  # Reduced from 120 to 100
        
    def _load_images(self):
        """Load all required images with professional fallbacks."""
        images = {}
        image_configs = {
            'doctor': ('image/doctor.png', self.BLUE, 'DOC'),
            'insulin': ('image/insulin.png', self.GREEN, 'INS'),
            'insuline': ('image/insuline.png', self.LIGHT_GREEN, 'LOW'),
            'stop': ('image/stop.png', self.RED, 'STP'),
            'fruits': ('image/fruits.png', self.ORANGE, 'FRT'),
            'candy': ('image/candy.png', self.YELLOW, 'CND'),
            'nutrient': ('image/nutrient.png', (139, 69, 19), 'NUT'),
            'person': ('image/person.png', self.GREEN, 'OK'),
            'patient': ('image/patient.png', self.ORANGE, 'CHK'),
            'died': ('image/died.png', self.RED, 'DIE')
        }
        
        for name, (filepath, fallback_color, text_fallback) in image_configs.items():
            try:
                if os.path.exists(filepath):
                    img = pygame.image.load(filepath)
                    if name in ['person', 'patient', 'died']:
                        images[name] = pygame.transform.scale(img, (60, 60))
                    else:
                        images[name] = pygame.transform.scale(img, (self.cell_size - 10, self.cell_size - 10))
                else:
                    # Create professional fallback
                    size = 60 if name in ['person', 'patient', 'died'] else self.cell_size - 10
                    surf = pygame.Surface((size, size))
                    surf.fill(fallback_color)
                    
                    # Add border
                    pygame.draw.rect(surf, self.BLACK, surf.get_rect(), 2)
                    
                    # Add text
                    font = pygame.font.Font(None, max(16, size // 3))
                    try:
                        text = font.render(text_fallback, True, self.WHITE)
                    except:
                        text = font.render(name[:3].upper(), True, self.WHITE)
                    
                    text_rect = text.get_rect(center=(size//2, size//2))
                    surf.blit(text, text_rect)
                    images[name] = surf
                    
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
                # Minimal fallback
                size = 60 if name in ['person', 'patient', 'died'] else self.cell_size - 10
                surf = pygame.Surface((size, size))
                surf.fill(self.GRAY)
                images[name] = surf
        
        return images
    
    def render(self):
        """Main render function."""
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Draw main components
        self._draw_title()
        self._draw_thinking_overlay()
        self._draw_grid()
        self._draw_side_panel()
        
        # Update display
        pygame.display.flip()
        
        return True
    
    def _draw_title(self):
        """Draw the main title."""
        title_surface = self.title_font.render("Personalized Diabetes Treatment", True, self.BLACK)
        title_rect = title_surface.get_rect(center=(self.window_width // 2, 25))
        self.screen.blit(title_surface, title_rect)
    
    def _draw_thinking_overlay(self):
        """Draw thinking overlay when agent is deciding."""
        if self.env.is_thinking:
            overlay = pygame.Surface((self.window_width, self.window_height))
            overlay.set_alpha(100)
            overlay.fill((200, 200, 200))
            self.screen.blit(overlay, (0, 0))
            
            thinking_text = self.title_font.render("THINKING...", True, self.PURPLE)
            thinking_rect = thinking_text.get_rect(center=(self.window_width // 2, self.window_height // 2))
            self.screen.blit(thinking_text, thinking_rect)
    
    def _draw_grid(self):
        """Draw the 7x7 grid with items, agent, and flashing effects."""
        # Grid background
        grid_rect = pygame.Rect(self.grid_x, self.grid_y, self.grid_width, self.grid_height)
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, grid_rect)
        pygame.draw.rect(self.screen, self.BLACK, grid_rect, 2)
        
        # Draw cells
        for row in range(7):
            for col in range(7):
                cell_x = self.grid_x + col * self.cell_size
                cell_y = self.grid_y + row * self.cell_size
                cell_rect = pygame.Rect(cell_x, cell_y, self.cell_size, self.cell_size)
                
                # Check if cell should flash
                if self.env.should_flash_cell(row, col):
                    pygame.draw.rect(self.screen, self.LIGHT_BLUE, cell_rect)
                
                # Draw cell border
                pygame.draw.rect(self.screen, self.DARK_GRAY, cell_rect, 1)
                
                # Draw treatment items with colored borders
                if (row, col) in self.env.grid_items:
                    item_type = self.env.grid_items[(row, col)]
                    
                    # Colored border hint
                    border_colors = {
                        'insulin': self.GREEN,
                        'insuline': self.LIGHT_GREEN,
                        'stop': self.RED,
                        'fruits': self.ORANGE,
                        'nutrient': (139, 69, 19),
                        'candy': self.YELLOW
                    }
                    
                    border_color = border_colors.get(item_type, self.GRAY)
                    pygame.draw.rect(self.screen, border_color, cell_rect, 3)
                    
                    # Draw item image
                    if item_type in self.images:
                        img_x = cell_x + 5
                        img_y = cell_y + 5
                        self.screen.blit(self.images[item_type], (img_x, img_y))
        
        # Draw agent (doctor)
        agent_x = self.grid_x + self.env.agent_pos[1] * self.cell_size + 5
        agent_y = self.grid_y + self.env.agent_pos[0] * self.cell_size + 5
        self.screen.blit(self.images['doctor'], (agent_x, agent_y))
    
    def _draw_side_panel(self):
        """Draw the complete side panel."""
        # Panel background
        panel_rect = pygame.Rect(self.panel_x - 5, self.panel_y - 5, 
                               self.side_panel_width + 10, self.side_panel_height + 10)
        pygame.draw.rect(self.screen, (245, 245, 245), panel_rect)
        pygame.draw.rect(self.screen, self.DARK_GRAY, panel_rect, 2)
        
        current_y = self.panel_y + 10
        
        # 1. Blood Sugar Graph
        current_y = self._draw_sugar_graph(current_y)
        current_y += self.section_spacing
        
        # 2. Patient Status
        current_y = self._draw_patient_status(current_y)
        current_y += self.section_spacing
        
        # 3. Rewards Panel
        current_y = self._draw_rewards_panel(current_y)
        current_y += self.section_spacing
        
        # 4. Current Time
        current_y = self._draw_time_panel(current_y)
        current_y += self.section_spacing
        
        # 5. Current Action
        current_y = self._draw_current_action(current_y)
    
    def _draw_section_header(self, title, y):
        """Draw a section header with underline."""
        header_surface = self.subtitle_font.render(title, True, self.DARK_GRAY)
        self.screen.blit(header_surface, (self.panel_x, y))
        
        # Underline
        header_width = header_surface.get_width()
        pygame.draw.line(self.screen, self.DARK_GRAY,
                        (self.panel_x, y + 18),  # Reduced from 22 to 18
                        (self.panel_x + header_width, y + 18), 1)
        
        return y + 25  # Reduced from 30 to 25
    
    def _draw_sugar_graph(self, start_y):
        """Draw animated blood sugar level graph."""
        y = self._draw_section_header("Blood Sugar Level (mg/dL)", start_y)
        
        # Chart area positioned with space for Y-axis labels on the left
        chart_x = self.panel_x + 30  # Leave space for Y-axis labels
        chart_rect = pygame.Rect(chart_x, y, self.chart_width, self.chart_height)
        pygame.draw.rect(self.screen, self.WHITE, chart_rect)
        pygame.draw.rect(self.screen, self.BLACK, chart_rect, 1)
        
        # Chart parameters
        sugar_min, sugar_max = 0, 200  # Changed from 300 to 200
        normal_min, normal_max = 70, 120
        perfect_min, perfect_max = 80, 105
        
        def sugar_to_y(sugar_value):
            return y + self.chart_height - int((sugar_value / sugar_max) * self.chart_height)
        
        # Draw danger zones (dashed lines)
        danger_high_y = sugar_to_y(200)  # Changed from 300 to 200
        danger_low_y = sugar_to_y(40)
        
        self._draw_dashed_line(chart_x, danger_high_y, 
                              chart_x + self.chart_width, danger_high_y, self.RED)
        self._draw_dashed_line(chart_x, danger_low_y, 
                              chart_x + self.chart_width, danger_low_y, self.BLUE)
        
        # Draw range backgrounds
        normal_y1 = sugar_to_y(normal_max)
        normal_y2 = sugar_to_y(normal_min)
        perfect_y1 = sugar_to_y(perfect_max)
        perfect_y2 = sugar_to_y(perfect_min)
        
        # Normal range (light green band)
        normal_rect = pygame.Rect(chart_x, normal_y1, self.chart_width, normal_y2 - normal_y1)
        normal_surface = pygame.Surface((self.chart_width, normal_y2 - normal_y1))
        normal_surface.set_alpha(100)
        normal_surface.fill(self.LIGHT_GREEN)
        self.screen.blit(normal_surface, (chart_x, normal_y1))
        
        # Perfect range (darker green inside)
        perfect_rect = pygame.Rect(chart_x, perfect_y1, self.chart_width, perfect_y2 - perfect_y1)
        perfect_surface = pygame.Surface((self.chart_width, perfect_y2 - perfect_y1))
        perfect_surface.set_alpha(150)
        perfect_surface.fill(self.DARK_GREEN)
        self.screen.blit(perfect_surface, (chart_x, perfect_y1))
        
        # Y-axis labels (positioned outside the chart area but inside the panel)
        for sugar_val in [0, 25, 50, 75, 100, 125, 150, 175, 200]:  # Changed scale for 0-200 range
            label_y = sugar_to_y(sugar_val)
            if label_y >= y and label_y <= y + self.chart_height:
                # Draw tick mark connecting to chart
                pygame.draw.line(self.screen, self.GRAY,
                               (chart_x - 5, label_y), (chart_x, label_y))
                
                # Position labels outside the chart area
                label_text = self.small_font.render(str(sugar_val), True, self.BLACK)
                self.screen.blit(label_text, (chart_x - 25, label_y - 6))
        
        # Draw smooth sugar curve
        if len(self.env.sugar_history) > 1:
            points = []
            for i, sugar in enumerate(self.env.sugar_history):
                if len(self.env.time_history) > i:
                    time_ratio = self.env.time_history[i] / 24.0 if self.env.time_history[i] > 0 else 0
                    x_pos = chart_x + int(time_ratio * self.chart_width)
                    y_pos = sugar_to_y(sugar)
                    points.append((x_pos, y_pos))
            
            if len(points) > 1:
                # Draw smooth line
                pygame.draw.lines(self.screen, self.RED, False, points, 2)
                
                # Draw current point
                if points:
                    pygame.draw.circle(self.screen, self.RED, points[-1], 4)
        
        # Current sugar level display
        current_sugar_text = self.small_font.render(f"Current Reading: {self.env.sugar_level:.1f} mg/dL", True, self.BLACK)
        self.screen.blit(current_sugar_text, (self.panel_x, y + self.chart_height + 3))  # Reduced spacing
        
        return y + self.chart_height + 20  # Reduced from 25 to 20
    
    def _draw_dashed_line(self, x1, y1, x2, y2, color):
        """Draw a dashed line."""
        dash_length = 5
        gap_length = 3
        total_length = abs(x2 - x1)
        
        for i in range(0, int(total_length), dash_length + gap_length):
            start_x = x1 + i
            end_x = min(x1 + i + dash_length, x2)
            if start_x < x2:
                pygame.draw.line(self.screen, color, (start_x, y1), (end_x, y2), 2)
    
    def _draw_patient_status(self, start_y):
        """Draw patient status with icon and description."""
        y = self._draw_section_header("Person Status", start_y)
        
        status = self.env.get_patient_status()
        
        # Status icon
        icon_x = self.panel_x + 10
        icon_y = y
        
        if status in self.images:
            self.screen.blit(self.images[status], (icon_x, icon_y))
        
        # Status text with color coding
        status_texts = {
            'person': 'Healthy (Perfect Range)',
            'patient': 'Needs Monitoring',
            'died': 'CRITICAL DANGER!'
        }
        
        status_colors = {
            'person': self.DARK_GREEN,
            'patient': self.ORANGE,
            'died': self.RED
        }
        
        status_text = status_texts.get(status, 'Unknown')
        status_color = status_colors.get(status, self.BLACK)
        
        text_surface = self.info_font.render(status_text, True, status_color)
        self.screen.blit(text_surface, (icon_x + 70, icon_y + 15))  # Reduced vertical offset
        
        return y + 60  # Reduced from 70 to 60
    
    def _draw_rewards_panel(self, start_y):
        """Draw accumulated rewards display."""
        y = self._draw_section_header("Rewards Panel", start_y)
        
        # Total rewards with color coding
        reward_color = self.DARK_GREEN if self.env.total_reward >= 0 else self.RED
        reward_text = self.info_font.render(f"Total Rewards: {self.env.total_reward:.1f}", True, reward_color)
        self.screen.blit(reward_text, (self.panel_x, y))
        
        # Reward bar visualization
        bar_width = self.chart_width - 15  # Reduced from 20 to 15
        bar_height = 12  # Reduced from 15 to 12
        bar_x = self.panel_x
        bar_y = y + 20  # Reduced from 25 to 20
        
        # Background bar
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.BLACK, (bar_x, bar_y, bar_width, bar_height), 1)
        
        # Reward fill (normalized to reasonable range -100 to +100)
        normalized_reward = max(-100, min(100, self.env.total_reward))
        if normalized_reward != 0:
            fill_width = int((abs(normalized_reward) / 100) * (bar_width // 2))
            if normalized_reward > 0:
                fill_rect = (bar_x + bar_width // 2, bar_y, fill_width, bar_height)
                pygame.draw.rect(self.screen, self.GREEN, fill_rect)
            else:
                fill_rect = (bar_x + bar_width // 2 - fill_width, bar_y, fill_width, bar_height)
                pygame.draw.rect(self.screen, self.RED, fill_rect)
        
        # Center line
        center_x = bar_x + bar_width // 2
        pygame.draw.line(self.screen, self.BLACK, (center_x, bar_y), (center_x, bar_y + bar_height), 2)
        
        return y + 40  # Reduced from 50 to 40
    
    def _draw_time_panel(self, start_y):
        """Draw current time with progress bar."""
        y = self._draw_section_header("Current Time", start_y)
        
        # Digital clock display
        time_str = self.env.get_formatted_time()
        time_text = self.info_font.render(f"Time: {time_str}", True, self.BLACK)
        self.screen.blit(time_text, (self.panel_x, y))
        
        # Progress bar for 24-hour simulation
        progress = min(self.env.simulation_time / 24.0, 1.0)
        bar_width = self.chart_width - 15  # Reduced from 20 to 15
        bar_height = 16  # Reduced from 20 to 16
        bar_x = self.panel_x
        bar_y = y + 20  # Reduced from 25 to 20
        
        # Background
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.BLACK, (bar_x, bar_y, bar_width, bar_height), 1)
        
        # Progress fill
        if progress > 0:
            fill_width = int(bar_width * progress)
            pygame.draw.rect(self.screen, self.BLUE, (bar_x, bar_y, fill_width, bar_height))
        
        # 3-hour markers
        for hour in range(3, 24, 3):
            marker_x = bar_x + int((hour / 24.0) * bar_width)
            pygame.draw.line(self.screen, self.BLACK, 
                           (marker_x, bar_y), (marker_x, bar_y + bar_height), 2)
        
        # Progress percentage
        progress_text = self.small_font.render(f"{progress*100:.1f}%", True, self.BLACK)
        self.screen.blit(progress_text, (bar_x + bar_width + 5, bar_y + 3))  # Adjusted vertical alignment
        
        return y + 40  # Reduced from 50 to 40
    
    def _draw_current_action(self, start_y):
        """Draw current action taken by agent."""
        y = self._draw_section_header("Current Action Taken", start_y)
        
        # Get current action
        action_text = self.env.get_current_action_text()
        
        # Action box with color coding
        box_width = self.side_panel_width - 15  # Reduced from 20 to 15
        box_height = 25  # Reduced from 30 to 25
        box_x = self.panel_x
        box_y = y
        
        # Color mapping for actions
        action_colors = {
            'High dosage': (255, 220, 220),          # Light red
            'Low dosage': (220, 255, 220),           # Light green
            'No dosage': (220, 220, 255),            # Light blue
            'Low sugar treatment': (255, 255, 200),  # Light yellow
            'Medium sugar treatment': (255, 235, 200), # Light orange
            'High sugar treatment': (255, 220, 255), # Light pink
            'No action': (240, 240, 240)             # Light gray
        }
        
        box_color = action_colors.get(action_text, (240, 240, 240))
        
        # Draw action box
        pygame.draw.rect(self.screen, box_color, (box_x, box_y, box_width, box_height))
        pygame.draw.rect(self.screen, self.BLACK, (box_x, box_y, box_width, box_height), 2)
        
        # Action text
        action_surface = self.info_font.render(action_text, True, self.BLACK)
        text_rect = action_surface.get_rect(center=(box_x + box_width//2, box_y + box_height//2))
        self.screen.blit(action_surface, text_rect)
        
        # Contextual advice based on sugar level
        advice_y = y + box_height + 8  # Reduced from 10 to 8
        sugar_level = self.env.sugar_level
        
        if sugar_level < 70:
            advice = "LOW: Need glucose urgently"
            advice_color = self.RED
        elif sugar_level > 150:
            advice = "HIGH: Need insulin treatment"
            advice_color = self.RED
        elif 80 <= sugar_level <= 105:
            advice = "PERFECT: Maintain current status"
            advice_color = self.DARK_GREEN
        else:
            advice = "NORMAL: Continue monitoring"
            advice_color = self.BLUE
        
        advice_surface = self.small_font.render(advice, True, advice_color)
        self.screen.blit(advice_surface, (self.panel_x, advice_y))
        
        return advice_y + 18  # Reduced from 20 to 18
    
    def save_screenshot(self, filename="diabetes_simulation.png"):
        """Save current screen as image."""
        try:
            pygame.image.save(self.screen, filename)
            print(f"Screenshot saved: {filename}")
        except Exception as e:
            print(f"Error saving screenshot: {e}")
    
    def close(self):
        """Close pygame and clean up."""
        pygame.quit()


# Demo function for testing
def run_demo():
    """Run demonstration of the diabetes treatment environment."""
    print("Initializing Diabetes Treatment Demo...")
    
    # Import here to avoid circular import
    from custom_env import DiabetesTreatmentEnv
    
    env = DiabetesTreatmentEnv()
    obs, _ = env.reset()
    
    print("Demo Controls:")
    print("   - Agent decides every 7.5 seconds (3 game hours)")
    print("   - Watch for 'THINKING...' overlay")
    print("   - Cells flash light blue when selected")
    print("   - Close window to stop")
    print()
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Random action for demo
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Render
        running = env.render()
        
        # Check completion
        if done:
            print(f"Simulation completed!")
            break
        
        # Control frame rate (30 FPS)
        clock.tick(30)
    
    # Final statistics
    print(f"\nFinal Results:")
    print(f"   Sugar Level: {env.sugar_level:.1f} mg/dL")
    print(f"   Patient Status: {env.get_patient_status()}")
    print(f"   Total Reward: {env.total_reward:.1f}")
    print(f"   Decisions Made: {env.step_count}")
    
    # Save final screenshot
    if env.renderer:
        env.renderer.save_screenshot("diabetes_final_state.png")
    
    env.close()
    print("Demo completed successfully!")


if __name__ == "__main__":
    run_demo()