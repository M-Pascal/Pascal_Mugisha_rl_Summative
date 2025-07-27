"""
rendering.py - Visualization GUI Components for Diabetes Simulator

This module handles all the visual rendering and user interface components
for the diabetes treatment simulation using Pygame.
"""

import pygame
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from custom_env import DiabetesEnvironment, TeddyExpression, GlucoseState

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (46, 204, 113)
YELLOW = (241, 196, 15)
RED = (231, 76, 60)
BLUE = (52, 152, 219)
GRAY = (149, 165, 166)
LIGHT_GRAY = (236, 240, 241)
DARK_GRAY = (52, 73, 94)
PURPLE = (155, 89, 182)
ORANGE = (230, 126, 34)
LIGHT_BLUE = (174, 214, 241)

class TeddyBearRenderer:
    """Handles rendering of the animated teddy bear character"""
    
    def __init__(self, x: int, y: int, size: int = 120):
        self.x = x
        self.y = y
        self.size = size
        self.animation_time = 0
        self.expression = TeddyExpression.HAPPY
        self.expression_transition = 0.0
        
    def update(self, expression: TeddyExpression, dt: float):
        """Update animation and expression"""
        self.animation_time += dt
        
        # Smooth expression transition
        if self.expression != expression:
            self.expression = expression
            self.expression_transition = 0.0
        else:
            self.expression_transition = min(1.0, self.expression_transition + dt * 3)
    
    def draw(self, screen: pygame.Surface):
        """Draw the teddy bear with current expression"""
        # Calculate animation offset
        bob_offset = math.sin(self.animation_time * 2) * 3
        shake_offset = 0
        
        if self.expression == TeddyExpression.DIZZY:
            shake_offset = math.sin(self.animation_time * 8) * 2
        
        x = self.x + shake_offset
        y = self.y + bob_offset
        
        # Draw shadow
        shadow_color = (0, 0, 0, 50)
        shadow_surface = pygame.Surface((self.size * 2, 40), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, shadow_color, (0, 10, self.size * 2, 20))
        screen.blit(shadow_surface, (x - self.size, y + 80))
        
        # Draw teddy bear body
        body_color = (139, 69, 19)
        pygame.draw.circle(screen, body_color, (int(x), int(y + 40)), 60)
        
        # Draw teddy bear head
        pygame.draw.circle(screen, body_color, (int(x), int(y)), 50)
        
        # Draw ears
        pygame.draw.circle(screen, body_color, (int(x - 35), int(y - 25)), 20)
        pygame.draw.circle(screen, body_color, (int(x + 35), int(y - 25)), 20)
        pygame.draw.circle(screen, (160, 82, 45), (int(x - 35), int(y - 25)), 12)
        pygame.draw.circle(screen, (160, 82, 45), (int(x + 35), int(y - 25)), 12)
        
        # Draw arms with slight animation
        arm_offset = math.sin(self.animation_time) * 2
        pygame.draw.circle(screen, body_color, (int(x - 50), int(y + 20 + arm_offset)), 25)
        pygame.draw.circle(screen, body_color, (int(x + 50), int(y + 20 - arm_offset)), 25)
        
        # Draw legs
        pygame.draw.circle(screen, body_color, (int(x - 25), int(y + 85)), 20)
        pygame.draw.circle(screen, body_color, (int(x + 25), int(y + 85)), 20)
        
        # Draw face based on expression
        self._draw_face(screen, x, y)
        
        # Draw snout
        pygame.draw.circle(screen, (160, 82, 45), (int(x), int(y + 10)), 15)
        
        # Draw expression-specific effects
        self._draw_expression_effects(screen, x, y)
    
    def _draw_face(self, screen: pygame.Surface, x: float, y: float):
        """Draw facial features based on current expression"""
        if self.expression == TeddyExpression.HAPPY:
            # Happy eyes
            pygame.draw.circle(screen, BLACK, (int(x - 15), int(y - 10)), 8)
            pygame.draw.circle(screen, BLACK, (int(x + 15), int(y - 10)), 8)
            pygame.draw.circle(screen, WHITE, (int(x - 12), int(y - 12)), 3)
            pygame.draw.circle(screen, WHITE, (int(x + 18), int(y - 12)), 3)
            # Smile
            pygame.draw.arc(screen, BLACK, (x - 10, y + 5, 20, 15), 0, math.pi, 3)
            
        elif self.expression == TeddyExpression.DIZZY:
            # Dizzy eyes (X shape)
            pygame.draw.line(screen, BLACK, (x - 20, y - 15), (x - 10, y - 5), 3)
            pygame.draw.line(screen, BLACK, (x - 10, y - 15), (x - 20, y - 5), 3)
            pygame.draw.line(screen, BLACK, (x + 10, y - 15), (x + 20, y - 5), 3)
            pygame.draw.line(screen, BLACK, (x + 20, y - 15), (x + 10, y - 5), 3)
            # Wavy mouth
            for i in range(5):
                offset = math.sin(i + self.animation_time * 2) * 2
                pygame.draw.circle(screen, BLACK, (int(x - 8 + i * 4), int(y + 15 + offset)), 2)
                
        elif self.expression == TeddyExpression.GRUMPY:
            # Angry eyes
            pygame.draw.circle(screen, BLACK, (int(x - 15), int(y - 10)), 8)
            pygame.draw.circle(screen, BLACK, (int(x + 15), int(y - 10)), 8)
            pygame.draw.circle(screen, RED, (int(x - 15), int(y - 10)), 6)
            pygame.draw.circle(screen, RED, (int(x + 15), int(y - 10)), 6)
            # Frown
            pygame.draw.arc(screen, BLACK, (x - 10, y + 20, 20, 15), math.pi, 2 * math.pi, 3)
            # Angry eyebrows
            pygame.draw.line(screen, BLACK, (x - 25, y - 20), (x - 5, y - 15), 3)
            pygame.draw.line(screen, BLACK, (x + 5, y - 15), (x + 25, y - 20), 3)
            
        elif self.expression == TeddyExpression.VERY_SICK:
            # Sick eyes (spirals)
            self._draw_spiral_eye(screen, x - 15, y - 10)
            self._draw_spiral_eye(screen, x + 15, y - 10)
            # Sick mouth
            pygame.draw.ellipse(screen, BLACK, (x - 8, y + 10, 16, 8))
            # Sick color tint
            sick_surface = pygame.Surface((100, 100), pygame.SRCALPHA)
            pygame.draw.circle(sick_surface, (0, 255, 0, 30), (50, 50), 50)
            screen.blit(sick_surface, (x - 50, y - 50))
            
        else:  # NEUTRAL
            # Normal eyes
            pygame.draw.circle(screen, BLACK, (int(x - 15), int(y - 10)), 8)
            pygame.draw.circle(screen, BLACK, (int(x + 15), int(y - 10)), 8)
            pygame.draw.circle(screen, WHITE, (int(x - 12), int(y - 12)), 3)
            pygame.draw.circle(screen, WHITE, (int(x + 18), int(y - 12)), 3)
            # Straight mouth
            pygame.draw.line(screen, BLACK, (x - 10, y + 15), (x + 10, y + 15), 3)
    
    def _draw_spiral_eye(self, screen: pygame.Surface, cx: float, cy: float):
        """Draw a spiral eye for sick expression"""
        pygame.draw.circle(screen, BLACK, (int(cx), int(cy)), 8)
        pygame.draw.circle(screen, WHITE, (int(cx), int(cy)), 6)
        
        # Draw spiral
        for i in range(10):
            angle = i * 0.5 + self.animation_time * 2
            radius = i * 0.5
            x = cx + math.cos(angle) * radius
            y = cy + math.sin(angle) * radius
            pygame.draw.circle(screen, BLACK, (int(x), int(y)), 1)
    
    def _draw_expression_effects(self, screen: pygame.Surface, x: float, y: float):
        """Draw additional effects based on expression"""
        if self.expression == TeddyExpression.DIZZY:
            # Draw floating stars
            for i in range(3):
                angle = self.animation_time + i * 2.1
                star_x = x + math.cos(angle) * 60
                star_y = y - 60 + math.sin(angle * 0.7) * 10
                self._draw_star(screen, star_x, star_y, 8)
        
        elif self.expression == TeddyExpression.VERY_SICK:
            # Draw sweat drops
            for i in range(2):
                drop_x = x + (-20 if i == 0 else 20)
                drop_y = y - 30 + math.sin(self.animation_time * 3 + i) * 5
                pygame.draw.circle(screen, LIGHT_BLUE, (int(drop_x), int(drop_y)), 4)
                pygame.draw.circle(screen, BLUE, (int(drop_x), int(drop_y)), 2)
    
    def _draw_star(self, screen: pygame.Surface, x: float, y: float, size: int):
        """Draw a star shape"""
        points = []
        for i in range(10):
            angle = i * math.pi / 5
            if i % 2 == 0:
                radius = size
            else:
                radius = size // 2
            point_x = x + math.cos(angle) * radius
            point_y = y + math.sin(angle) * radius
            points.append((point_x, point_y))
        
        pygame.draw.polygon(screen, YELLOW, points)

class GlucoseMeterRenderer:
    """Handles rendering of the glucose meter and related displays"""
    
    def __init__(self, x: int, y: int, width: int = 80, height: int = 400):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.glucose_level = 100.0
        self.target_glucose = 100.0
        self.trend = "steady"
        self.animation_time = 0
        
    def update(self, state: Dict, dt: float):
        """Update meter state"""
        self.glucose_level = state['glucose_level']
        self.target_glucose = state['target_glucose']
        self.trend = state['glucose_trend']
        self.animation_time += dt
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw the glucose meter"""
        # Draw meter background
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height), 3)
        
        # Draw target ranges as background colors
        self._draw_target_zones(screen)
        
        # Draw glucose level bar
        self._draw_glucose_bar(screen)
        
        # Draw scale markers
        self._draw_scale(screen, font)
        
        # Draw current reading
        self._draw_current_reading(screen, font)
        
        # Draw trend indicator
        self._draw_trend_indicator(screen, font)
    
    def _draw_target_zones(self, screen: pygame.Surface):
        """Draw colored zones for different glucose ranges"""
        max_glucose = 400
        zones = [
            (0, 70, RED),      # Hypoglycemic
            (70, 120, GREEN),  # Target range
            (120, 180, YELLOW), # Elevated
            (180, 400, RED)    # High
        ]
        
        for min_val, max_val, color in zones:
            start_y = self.y + self.height - (max_val / max_glucose * self.height)
            zone_height = (max_val - min_val) / max_glucose * self.height
            
            # Create semi-transparent overlay
            zone_surface = pygame.Surface((self.width - 6, zone_height), pygame.SRCALPHA)
            zone_surface.fill((*color, 30))
            screen.blit(zone_surface, (self.x + 3, start_y))
    
    def _draw_glucose_bar(self, screen: pygame.Surface):
        """Draw the main glucose level bar"""
        max_glucose = 400
        fill_height = min(self.glucose_level / max_glucose * self.height, self.height - 6)
        
        # Get color based on glucose level
        color = self._get_glucose_color()
        
        # Draw the bar with gradient effect
        bar_rect = pygame.Rect(self.x + 3, self.y + self.height - fill_height - 3, 
                              self.width - 6, fill_height)
        pygame.draw.rect(screen, color, bar_rect)
        
        # Add shine effect
        if fill_height > 10:
            shine_rect = pygame.Rect(self.x + 5, self.y + self.height - fill_height - 1, 
                                   self.width - 12, 6)
            shine_color = tuple(min(255, c + 50) for c in color)
            pygame.draw.rect(screen, shine_color, shine_rect)
        
        # Draw target glucose level indicator
        target_y = self.y + self.height - (self.target_glucose / max_glucose * self.height)
        pygame.draw.line(screen, BLACK, (self.x - 5, target_y), (self.x + self.width + 5, target_y), 2)
        pygame.draw.polygon(screen, BLACK, [(self.x - 8, target_y - 3), (self.x - 2, target_y), (self.x - 8, target_y + 3)])
    
    def _draw_scale(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw scale markers and labels"""
        max_glucose = 400
        
        # Major markers every 50 mg/dL
        for glucose_val in range(0, 401, 50):
            y_pos = self.y + self.height - (glucose_val / max_glucose * self.height)
            
            # Draw marker line
            marker_length = 15 if glucose_val % 100 == 0 else 10
            pygame.draw.line(screen, BLACK, 
                           (self.x + self.width - marker_length, y_pos), 
                           (self.x + self.width, y_pos), 2)
            
            # Draw label for major markers
            if glucose_val % 100 == 0:
                text = font.render(str(glucose_val), True, BLACK)
                screen.blit(text, (self.x + self.width + 5, y_pos - 10))
    
    def _draw_current_reading(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw current glucose reading"""
        # Main glucose reading
        glucose_text = f"{int(self.glucose_level)} mg/dL"
        text_surface = font.render(glucose_text, True, BLACK)
        text_rect = text_surface.get_rect()
        text_rect.centerx = self.x + self.width // 2
        text_rect.bottom = self.y - 10
        
        # Background for readability
        bg_rect = text_rect.inflate(10, 4)
        pygame.draw.rect(screen, WHITE, bg_rect)
        pygame.draw.rect(screen, BLACK, bg_rect, 1)
        
        screen.blit(text_surface, text_rect)
        
        # Status text
        status_text = self._get_status_text()
        status_color = self._get_glucose_color()
        status_surface = font.render(status_text, True, status_color)
        status_rect = status_surface.get_rect()
        status_rect.centerx = self.x + self.width // 2
        status_rect.top = text_rect.bottom + 5
        screen.blit(status_surface, status_rect)
    
    def _draw_trend_indicator(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw glucose trend arrow"""
        arrow_x = self.x + self.width + 60
        arrow_y = self.y - 30
        
        if self.trend == "rising":
            arrow_text = "⬆️"
            color = RED
        elif self.trend == "falling":
            arrow_text = "⬇️"
            color = BLUE
        else:
            arrow_text = "➡️"
            color = GREEN
        
        # Animate the arrow
        offset = math.sin(self.animation_time * 3) * 2
        arrow_surface = font.render(arrow_text, True, color)
        screen.blit(arrow_surface, (arrow_x, arrow_y + offset))
        
        # Trend text
        trend_text = font.render(self.trend.upper(), True, color)
        screen.blit(trend_text, (arrow_x, arrow_y + 25))
    
    def _get_glucose_color(self):
        """Get color based on glucose level"""
        if 70 <= self.glucose_level <= 120:
            return GREEN
        elif 120 < self.glucose_level <= 180:
            return YELLOW
        else:
            return RED
    
    def _get_status_text(self):
        """Get status text based on glucose level"""
        glucose = self.glucose_level
        if glucose < 50:
            return "CRITICAL LOW"
        elif glucose < 70:
            return "LOW"
        elif glucose <= 120:
            return "NORMAL"
        elif glucose <= 180:
            return "ELEVATED"
        elif glucose <= 250:
            return "HIGH"
        else:
            return "CRITICAL HIGH"

class ButtonRenderer:
    """Handles rendering of interactive buttons"""
    
    def __init__(self, x: int, y: int, width: int, height: int, text: str, insulin_units: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.insulin_units = insulin_units
        self.is_hovered = False
        self.is_pressed = False
        self.click_animation = 0.0
        
    def update(self, dt: float):
        """Update button animations"""
        if self.click_animation > 0:
            self.click_animation -= dt * 5
            self.click_animation = max(0, self.click_animation)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse events and return True if clicked"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_pressed = True
                self.click_animation = 1.0
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_pressed = False
        elif event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        
        return False
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw the button with animations"""
        # Calculate visual effects
        scale = 0.95 if self.is_pressed else 1.0
        if self.click_animation > 0:
            scale -= self.click_animation * 0.1
        
        # Calculate button rect with scaling
        scaled_width = int(self.rect.width * scale)
        scaled_height = int(self.rect.height * scale)
        scaled_x = self.rect.centerx - scaled_width // 2
        scaled_y = self.rect.centery - scaled_height // 2
        scaled_rect = pygame.Rect(scaled_x, scaled_y, scaled_width, scaled_height)
        
        # Choose colors
        if self.is_pressed:
            bg_color = GRAY
            border_color = DARK_GRAY
        elif self.is_hovered:
            bg_color = LIGHT_GRAY
            border_color = BLUE
        else:
            bg_color = WHITE
            border_color = BLACK
        
        # Draw button
        pygame.draw.rect(screen, bg_color, scaled_rect)
        pygame.draw.rect(screen, border_color, scaled_rect, 3)
        
        # Add glow effect if hovered
        if self.is_hovered:
            glow_rect = scaled_rect.inflate(6, 6)
            glow_surface = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*BLUE, 30), glow_surface.get_rect())
            screen.blit(glow_surface, glow_rect.topleft)
        
        # Draw text
        text_color = WHITE if self.is_pressed else BLACK
        text_surface = font.render(self.text, True, text_color)
        text_rect = text_surface.get_rect(center=scaled_rect.center)
        screen.blit(text_surface, text_rect)
        
        # Draw insulin icon
        if self.insulin_units > 0:
            icon_x = scaled_rect.right - 15
            icon_y = scaled_rect.top + 5
            pygame.draw.circle(screen, BLUE, (icon_x, icon_y), 4)
            pygame.draw.rect(screen, BLUE, (icon_x - 2, icon_y - 8, 4, 6))

class HistoryGraphRenderer:
    """Renders glucose history graph"""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.glucose_history = []
        
    def update(self, glucose_history: List[float]):
        """Update glucose history data"""
        self.glucose_history = glucose_history[-100:]  # Keep last 100 points
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw the glucose history graph"""
        # Draw background
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height), 2)
        
        if len(self.glucose_history) < 2:
            return
        
        # Draw target range
        self._draw_target_range(screen)
        
        # Draw glucose line
        self._draw_glucose_line(screen)
        
        # Draw title
        title = font.render("Glucose History", True, BLACK)
        screen.blit(title, (self.x + 5, self.y - 25))
    
    def _draw_target_range(self, screen: pygame.Surface):
        """Draw target glucose range as background"""
        min_glucose, max_glucose = 50, 300
        target_min, target_max = 70, 120
        
        # Calculate positions
        range_top = self.y + (1 - (target_max - min_glucose) / (max_glucose - min_glucose)) * self.height
        range_bottom = self.y + (1 - (target_min - min_glucose) / (max_glucose - min_glucose)) * self.height
        range_height = range_bottom - range_top
        
        # Draw target range
        target_surface = pygame.Surface((self.width - 4, range_height), pygame.SRCALPHA)
        target_surface.fill((*GREEN, 50))
        screen.blit(target_surface, (self.x + 2, range_top))
    
    def _draw_glucose_line(self, screen: pygame.Surface):
        """Draw the glucose level line graph"""
        if len(self.glucose_history) < 2:
            return
        
        min_glucose, max_glucose = 50, 300
        points = []
        
        for i, glucose in enumerate(self.glucose_history):
            x = self.x + (i / (len(self.glucose_history) - 1)) * (self.width - 4) + 2
            y = self.y + (1 - (glucose - min_glucose) / (max_glucose - min_glucose)) * (self.height - 4) + 2
            y = max(self.y + 2, min(self.y + self.height - 2, y))
            points.append((x, y))
        
        # Draw line
        if len(points) > 1:
            pygame.draw.lines(screen, RED, False, points, 3)
        
        # Draw current point
        if points:
            pygame.draw.circle(screen, RED, (int(points[-1][0]), int(points[-1][1])), 5)
            pygame.draw.circle(screen, WHITE, (int(points[-1][0]), int(points[-1][1])), 3)

class NotificationRenderer:
    """Handles rendering of notifications and alerts"""
    
    def __init__(self):
        self.notifications = []
        
    def add_notification(self, text: str, color: Tuple[int, int, int] = BLACK, duration: float = 3.0):
        """Add a new notification"""
        self.notifications.append({
            'text': text,
            'color': color,
            'duration': duration,
            'time_left': duration,
            'alpha': 255
        })
    
    def update(self, dt: float):
        """Update notifications"""
        for notification in self.notifications[:]:
            notification['time_left'] -= dt
            
            # Fade out in last second
            if notification['time_left'] < 1.0:
                notification['alpha'] = int(255 * notification['time_left'])
            
            # Remove expired notifications
            if notification['time_left'] <= 0:
                self.notifications.remove(notification)
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw all active notifications"""
        y_offset = 50
        
        for notification in self.notifications:
            # Create text surface with alpha
            text_surface = font.render(notification['text'], True, notification['color'])
            
            if notification['alpha'] < 255:
                # Apply alpha to surface
                alpha_surface = pygame.Surface(text_surface.get_size(), pygame.SRCALPHA)
                alpha_surface.fill((255, 255, 255, notification['alpha']))
                text_surface.blit(alpha_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            
            # Draw background
            text_rect = text_surface.get_rect()
            text_rect.centerx = SCREEN_WIDTH // 2
            text_rect.y = y_offset
            
            bg_rect = text_rect.inflate(20, 10)
            bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surface.fill((255, 255, 255, min(200, notification['alpha'])))
            screen.blit(bg_surface, bg_rect)
            
            pygame.draw.rect(screen, notification['color'], bg_rect, 2)
            screen.blit(text_surface, text_rect)
            
            y_offset += 40

class DiabetesSimulatorGUI:
    """Main GUI class that orchestrates all rendering components"""
    
    def __init__(self):
        # Initialize Pygame
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Diabetes Treatment Simulator")
        self.clock = pygame.time.Clock()
        
        # Initialize environment
        self.env = DiabetesEnvironment()
        
        # Initialize fonts
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 20)
        self.large_font = pygame.font.Font(None, 48)
        
        # Initialize renderers
        self.teddy_renderer = TeddyBearRenderer(250, 300)
        self.glucose_meter = GlucoseMeterRenderer(900, 150)
        self.history_graph = HistoryGraphRenderer(500, 450, 400, 200)
        self.notification_renderer = NotificationRenderer()
        
        # Initialize buttons
        self.buttons = [
            ButtonRenderer(50, 500, 100, 60, "0 Units", 0),
            ButtonRenderer(170, 500, 100, 60, "2 Units", 2),
            ButtonRenderer(290, 500, 100, 60, "5 Units", 5),
            ButtonRenderer(410, 500, 100, 60, "10 Units", 10)
        ]
        
        # Game state
        self.running = True
        self.last_meal_notification = 0
        self.last_insulin_dose = 0
        
    def handle_events(self):
        """Handle all pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            # Handle button clicks
            for button in self.buttons:
                if button.handle_event(event):
                    # Administer insulin
                    state, reward, done = self.env.step(button.insulin_units)
                    
                    # Add notification
                    if button.insulin_units > 0:
                        self.notification_renderer.add_notification(
                            f"Administered {button.insulin_units} units of insulin",
                            BLUE, 2.0
                        )
                        self.last_insulin_dose = button.insulin_units
                    
                    # Check for meal events
                    if state.get('meal_pending', False) and self.last_meal_notification < state['game_time_raw'] - 300:
                        self.notification_renderer.add_notification(
                            "Meal time! Glucose will rise soon", ORANGE, 3.0
                        )
                        self.last_meal_notification = state['game_time_raw']
    
    def update(self, dt: float):
        """Update all game components"""
        # Step environment if no input
        state, reward, done = self.env.step(0)
        
        # Update renderers
        self.teddy_renderer.update(state['teddy_expression'], dt)
        self.glucose_meter.update(state, dt)
        self.history_graph.update(state['glucose_history'])
        self.notification_renderer.update(dt)
        
        for button in self.buttons:
            button.update(dt)
        
        # Check for critical glucose levels
        glucose = state['glucose_level']
        if glucose < 70 and not hasattr(self, 'low_glucose_warned'):
            self.notification_renderer.add_notification(
                "WARNING: Low glucose detected!", RED, 4.0
            )
            self.low_glucose_warned = True
        elif glucose >= 70:
            self.low_glucose_warned = False
        
        if glucose > 250 and not hasattr(self, 'high_glucose_warned'):
            self.notification_renderer.add_notification(
                "CRITICAL: Very high glucose!", RED, 4.0
            )
            self.high_glucose_warned = True
        elif glucose <= 250:
            self.high_glucose_warned = False
        
        return state, done
    
    def draw_ui(self, state: Dict):
        """Draw the user interface"""
        # Clear screen
        self.screen.fill(WHITE)
        
        # Draw title
        title_text = self.title_font.render("Diabetes Treatment Simulator", True, BLACK)
        self.screen.blit(title_text, (20, 20))
        
        # Draw game stats
        stats_y = 70
        stats = [
            f"Score: {state['score']}",
            f"Time: {state['game_time']}",
            f"Time in Range: {state['time_in_range_percent']:.1f}%",
            f"Active Insulin: {state['active_insulin']:.1f} units"
        ]
        
        for i, stat in enumerate(stats):
            color = GREEN if "Time in Range" in stat and state['time_in_range_percent'] > 70 else BLACK
            text = self.font.render(stat, True, color)
            self.screen.blit(text, (20, stats_y + i * 30))
        
        # Draw instructions
        instructions = [
            "Goal: Keep glucose in 70-120 mg/dL range",
            "Click buttons to administer insulin",
            "Watch for meal notifications",
            "Monitor trends and teddy's expression"
        ]
        
        instruction_y = 300
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, DARK_GRAY)
            self.screen.blit(text, (20, instruction_y + i * 25))
        
        # Draw glucose status
        glucose_state = state['glucose_state']
        status_messages = {
            'very_low': ("EMERGENCY: Severe Hypoglycemia", RED),
            'low': ("CAUTION: Low Glucose", YELLOW),
            'normal': ("EXCELLENT: In Target Range", GREEN),
            'elevated': ("ELEVATED: Monitor Closely", YELLOW),
            'high': ("HIGH: Insulin Needed", RED),
            'very_high': ("CRITICAL: Dangerous Level", RED)
        }
        
        status_text, status_color = status_messages.get(glucose_state.value, ("Unknown", BLACK))
        status_surface = self.font.render(status_text, True, status_color)
        status_rect = status_surface.get_rect()
        status_rect.centerx = SCREEN_WIDTH // 2
        status_rect.y = 100
        
        # Status background
        bg_rect = status_rect.inflate(20, 10)
        pygame.draw.rect(self.screen, WHITE, bg_rect)
        pygame.draw.rect(self.screen, status_color, bg_rect, 2)
        self.screen.blit(status_surface, status_rect)
        
        # Draw insulin button label
        button_label = self.font.render("Administer Insulin:", True, BLACK)
        self.screen.blit(button_label, (50, 470))
    
    def draw(self, state: Dict):
        """Draw all components"""
        # Draw UI elements
        self.draw_ui(state)
        
        # Draw main components
        self.teddy_renderer.draw(self.screen)
        self.glucose_meter.draw(self.screen, self.font)
        self.history_graph.draw(self.screen, self.small_font)
        
        # Draw buttons
        for button in self.buttons:
            button.draw(self.screen, self.font)
        
        # Draw notifications
        self.notification_renderer.draw(self.screen, self.font)
        
        # Update display
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        last_time = pygame.time.get_ticks()
        
        while self.running:
            # Calculate delta time
            current_time = pygame.time.get_ticks()
            dt = (current_time - last_time) / 1000.0
            last_time = current_time
            
            # Handle events
            self.handle_events()
            
            # Update game state
            state, done = self.update(dt)
            
            # Draw everything
            self.draw(state)
            
            # Control frame rate
            self.clock.tick(FPS)
            
            # Check if simulation is complete
            if done:
                self.notification_renderer.add_notification(
                    "24-hour simulation complete!", GREEN, 5.0
                )
                # Could add end screen here
        
        pygame.quit()

# Main execution
if __name__ == "__main__":
    print("=== Diabetes Treatment Simulator ===")
    print("Starting GUI application...")
    
    try:
        simulator = DiabetesSimulatorGUI()
        simulator.run()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Error running simulator: {e}")
        import traceback
        traceback.print_exc()
    
    print("Simulation ended.")
    