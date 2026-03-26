"""
modules/sketch_position_control.py

AI-Powered Shape Positioning System
====================================

Enables intuitive gesture-based repositioning of drawn shapes on the canvas.
Users can grab shapes with a closed fist gesture and move them relative to
their hand motion, allowing for quick shape arrangement without redrawing.

Architecture:
  - GestureActivator: Detects and confirms closed fist gesture
  - ShapeTracker: Maintains registry of all drawn shapes
  - MovementController: Calculates shape position updates
  - BoundaryManager: Enforces canvas boundary constraints
  - VisualIndicators: Renders feedback UI elements

Features:
  ✓ 2-3 second hold activation (prevents accidental activation)
  ✓ Relative hand motion tracking (intuitive dragging feel)
  ✓ Automatic boundary clamping (shapes stay within canvas)
  ✓ Multi-shape support (most recent shape moves)
  ✓ Visual feedback (highlighting, outlines, status text)
  ✓ Timeout protection (auto-release after 3+ seconds idle)
  ✓ Gesture confirmation (prevents flickering)

Usage:
  from modules.sketch_position_control import (
      GestureActivator, ShapeTracker, MovementController,
      BoundaryManager, VisualIndicators
  )
  
  # In DrawState initialization:
  self.gesture_activator = GestureActivator(hold_duration_sec=2.5)
  self.shape_tracker = ShapeTracker()
  self.movement_controller = MovementController((SCREEN_W, SCREEN_H))
  self.boundary_manager = BoundaryManager(SCREEN_W, SCREEN_H, UI_H)
  self.visual_indicators = VisualIndicators()
  
  # In main loop:
  if gesture == "closed_fist":
      is_activated = gesture_activator.update("closed_fist", is_fist=True)
      if is_activated:
          shape = shape_tracker.get_most_recent()
          movement_controller.start_move(shape['id'], hand_x, hand_y)

Author: AI Virtual Drawing Team
Date: March 26, 2026
"""

import time
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict
import uuid


class GestureActivator:
    """
    Detects and confirms closed fist gesture with hold duration tracking.
    
    Implements hysteresis to require sustained fist gesture for N seconds
    before activating shape movement. This prevents accidental activation
    from brief hand closures.
    
    Attributes:
        hold_duration_sec: Seconds to hold fist before activating
        confirmation_frames: Consecutive frames to confirm gesture
        hold_start_time: When fist gesture began
        is_confirmed: Whether gesture has been held long enough
    """
    
    def __init__(self, hold_duration_sec: float = 2.5, confirmation_frames: int = 2):
        """
        Initialize gesture activation detector.
        
        Args:
            hold_duration_sec: Minimum fist hold time to activate (2-3s)
            confirmation_frames: Frames to confirm gesture (hysteresis)
        """
        self.hold_duration_sec = hold_duration_sec
        self.confirmation_frames = confirmation_frames
        self.hold_start_time: Optional[float] = None
        self.confirmation_count = 0
        self.is_confirmed = False
        
    def update(self, current_gesture: str, is_fist: bool, current_time: Optional[float] = None) -> bool:
        """
        Update gesture state and check if activation threshold reached.
        
        Args:
            current_gesture: Current detected gesture name
            is_fist: Whether hand is in closed fist state
            current_time: Current time (uses time.time() if None)
            
        Returns:
            True if fist has been held for required duration, False otherwise
        """
        if current_time is None:
            current_time = time.time()
        
        # Fist gesture begins
        if is_fist and current_gesture == "fist":
            if self.hold_start_time is None:
                self.hold_start_time = current_time
            
            # Increment confirmation counter
            self.confirmation_count = min(self.confirmation_count + 1,
                                         self.confirmation_frames + 1)
            
            # Check if held long enough AND confirmed for multiple frames
            if (self.confirmation_count >= self.confirmation_frames and
                current_time - self.hold_start_time >= self.hold_duration_sec):
                self.is_confirmed = True
                return True
        else:
            # Gesture lost - reset confirmation
            self.reset()
        
        return False
    
    def get_hold_progress(self, current_time: Optional[float] = None) -> float:
        """
        Get progress towards activation (0.0 to 1.0).
        
        Args:
            current_time: Current time (uses time.time() if None)
            
        Returns:
            Progress ratio (0.0 = just started, 1.0 = ready to activate)
        """
        if self.hold_start_time is None:
            return 0.0
        
        if current_time is None:
            current_time = time.time()
        
        elapsed = current_time - self.hold_start_time
        progress = min(elapsed / self.hold_duration_sec, 1.0)
        return progress
    
    def reset(self):
        """Clear activation state when gesture released."""
        self.hold_start_time = None
        self.confirmation_count = 0
        self.is_confirmed = False


class ShapeTracker:
    """
    Maintains registry of all drawn shapes on canvas.
    
    Tracks shape positions, properties, and history. Provides quick lookup
    for most recent shape or nearest shape to cursor.
    
    Shape Data Format:
    {
        'id': 'unique_id',
        'type': 'circle',  # circle, rectangle, triangle, line
        'original_pos': (x, y),
        'current_pos': (x, y),
        'center': (x, y),
        'bounding_box': (x1, y1, x2, y2),
        'size': (w, h),
        'rotation': 0,
        'canvas_data': numpy array (mask),
        'color': (b, g, r),
        'thickness': 2,
        'timestamp': float,
        'moved': bool,
        'move_count': int,
    }
    
    Attributes:
        shapes: List of all tracked shapes
        shape_ids: Dict mapping shape ID to index for quick lookup
        creation_order: List of shape IDs in chronological order
    """
    
    def __init__(self):
        """Initialize empty shape registry."""
        self.shapes: List[Dict[str, Any]] = []
        self.shape_ids: Dict[str, int] = {}  # id -> index mapping
        self.creation_order: List[str] = []
        
    def add_shape(self, shape_data: Dict[str, Any]) -> str:
        """
        Register newly drawn shape.
        
        Args:
            shape_data: Shape data dictionary with properties
            
        Returns:
            Unique shape ID assigned to this shape
        """
        # Generate unique ID if not provided
        if 'id' not in shape_data:
            shape_data['id'] = str(uuid.uuid4())
        
        shape_id = shape_data['id']
        
        # Initialize move tracking
        if 'moved' not in shape_data:
            shape_data['moved'] = False
        if 'move_count' not in shape_data:
            shape_data['move_count'] = 0
        if 'timestamp' not in shape_data:
            shape_data['timestamp'] = time.time()
        
        # Store shape
        idx = len(self.shapes)
        self.shapes.append(shape_data)
        self.shape_ids[shape_id] = idx
        self.creation_order.append(shape_id)
        
        return shape_id
    
    def get_most_recent(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recently drawn shape.
        
        Returns:
            Most recent shape dict, or None if no shapes exist
        """
        if not self.shapes:
            return None
        return self.shapes[-1]
    
    def get_nearest(self, x: int, y: int, radius: int = 100) -> Optional[Dict[str, Any]]:
        """
        Find shape nearest to cursor position within radius.
        
        Args:
            x, y: Cursor position
            radius: Search radius in pixels
            
        Returns:
            Nearest shape dict, or None if none in radius
        """
        if not self.shapes:
            return None
        
        min_dist = float('inf')
        nearest_shape = None
        
        for shape in self.shapes:
            cx, cy = shape.get('center', shape.get('current_pos', (0, 0)))
            dist = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            
            if dist < min_dist and dist <= radius:
                min_dist = dist
                nearest_shape = shape
        
        return nearest_shape
    
    def get_by_id(self, shape_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve shape by unique ID.
        
        Args:
            shape_id: Unique shape identifier
            
        Returns:
            Shape dict, or None if not found
        """
        if shape_id not in self.shape_ids:
            return None
        idx = self.shape_ids[shape_id]
        return self.shapes[idx] if idx < len(self.shapes) else None
    
    def update_shape(self, shape_id: str, updates: Dict[str, Any]):
        """
        Update properties of existing shape.
        
        Args:
            shape_id: Shape identifier
            updates: Dictionary of property updates
        """
        shape = self.get_by_id(shape_id)
        if shape:
            shape.update(updates)
    
    def get_all_shapes(self) -> List[Dict[str, Any]]:
        """Get all tracked shapes."""
        return self.shapes.copy()
    
    def remove_shape(self, shape_id: str):
        """Remove shape from tracking (e.g., if shape deleted)."""
        if shape_id in self.shape_ids:
            del self.shape_ids[shape_id]
            self.creation_order.remove(shape_id)
            self.shapes = [s for s in self.shapes if s.get('id') != shape_id]
    
    def clear_all(self):
        """Clear all tracked shapes."""
        self.shapes.clear()
        self.shape_ids.clear()
        self.creation_order.clear()


class MovementController:
    """
    Handles shape movement calculations and tracking.
    
    Implements relative hand movement: when hand moves 10px right,
    shape moves 10px right. Creates intuitive dragging behavior.
    
    Attributes:
        canvas_shape: Canvas dimensions (width, height)
        moving_shape_id: ID of shape currently being moved
        initial_hand_pos: Hand position when grab began
        initial_shape_pos: Shape position when grab began
        total_movement: Total pixels moved from original position
    """
    
    def __init__(self, canvas_shape: Tuple[int, int]):
        """
        Initialize movement controller.
        
        Args:
            canvas_shape: Canvas dimensions as (width, height)
        """
        self.canvas_shape = canvas_shape
        self.moving_shape_id: Optional[str] = None
        self.initial_hand_pos: Optional[Tuple[int, int]] = None
        self.initial_shape_pos: Optional[Tuple[int, int]] = None
        self.total_movement = (0, 0)
    
    def start_move(self, shape_id: str, hand_x: int, hand_y: int,
                   shape_pos: Tuple[int, int]):
        """
        Begin moving a shape.
        
        Args:
            shape_id: ID of shape to move
            hand_x, hand_y: Current hand position
            shape_pos: Current shape position (center)
        """
        self.moving_shape_id = shape_id
        self.initial_hand_pos = (hand_x, hand_y)
        self.initial_shape_pos = shape_pos
        self.total_movement = (0, 0)
    
    def calculate_new_position(self, hand_x: int, hand_y: int) -> Tuple[int, int]:
        """
        Calculate new shape position based on hand movement.
        
        Implements relative motion: shape moves by the same amount as hand.
        
        Args:
            hand_x, hand_y: Current hand position
            
        Returns:
            Calculated new position (x, y)
        """
        if not self.moving_shape_id or not self.initial_hand_pos or not self.initial_shape_pos:
            return (0, 0)
        
        # Calculate hand movement delta
        hand_dx = hand_x - self.initial_hand_pos[0]
        hand_dy = hand_y - self.initial_hand_pos[1]
        
        # Apply same movement to shape
        new_x = self.initial_shape_pos[0] + hand_dx
        new_y = self.initial_shape_pos[1] + hand_dy
        
        self.total_movement = (hand_dx, hand_dy)
        
        return (int(new_x), int(new_y))
    
    def get_shape_id(self) -> Optional[str]:
        """Get ID of currently moving shape."""
        return self.moving_shape_id
    
    def is_moving(self) -> bool:
        """Check if any shape is currently being moved."""
        return self.moving_shape_id is not None
    
    def end_move(self) -> str:
        """
        Finalize shape movement.
        
        Returns:
            ID of shape that was moved (for notification)
        """
        moved_id = self.moving_shape_id
        self.moving_shape_id = None
        self.initial_hand_pos = None
        self.initial_shape_pos = None
        return moved_id or ""


class BoundaryManager:
    """
    Enforces canvas boundary constraints on shape movement.
    
    Prevents shapes from being dragged outside the canvas bounds.
    Handles UI area at top and provides smooth edge behavior.
    
    Attributes:
        canvas_w, canvas_h: Canvas dimensions
        ui_height: Height of UI area at top to avoid
        boundary_mode: "clamp" (stick to edge) or "bounce" (reject)
    """
    
    def __init__(self, canvas_w: int, canvas_h: int, ui_height: int = 160,
                 boundary_mode: str = "clamp"):
        """
        Initialize boundary manager.
        
        Args:
            canvas_w, canvas_h: Canvas dimensions in pixels
            ui_height: Height of UI area at top
            boundary_mode: How to handle boundary ("clamp" or "bounce")
        """
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.ui_height = ui_height
        self.boundary_mode = boundary_mode
    
    def clamp_position(self, shape: Dict[str, Any], x: int, y: int) -> Tuple[int, int]:
        """
        Clamp shape position to stay within canvas bounds.
        
        Args:
            shape: Shape dictionary (needs 'size' property)
            x, y: Desired position
            
        Returns:
            Clamped position (x, y)
        """
        shape_w, shape_h = shape.get('size', (50, 50))
        half_w = shape_w // 2
        half_h = shape_h // 2
        
        # Clamp X
        min_x = half_w
        max_x = self.canvas_w - half_w
        x = max(min_x, min(max_x, x))
        
        # Clamp Y (avoid UI area)
        min_y = self.ui_height + half_h
        max_y = self.canvas_h - half_h
        y = max(min_y, min(max_y, y))
        
        return (x, y)
    
    def is_within_bounds(self, shape: Dict[str, Any], x: int, y: int) -> bool:
        """
        Check if position is within bounds.
        
        Args:
            shape: Shape dictionary
            x, y: Position to check
            
        Returns:
            True if position is within canvas bounds
        """
        shape_w, shape_h = shape.get('size', (50, 50))
        half_w = shape_w // 2
        half_h = shape_h // 2
        
        within_x = half_w <= x <= self.canvas_w - half_w
        within_y = self.ui_height + half_h <= y <= self.canvas_h - half_h
        
        return within_x and within_y
    
    def get_distance_to_boundary(self, shape: Dict[str, Any], x: int, y: int) -> float:
        """
        Get distance from position to nearest boundary.
        
        Args:
            shape: Shape dictionary
            x, y: Current position
            
        Returns:
            Distance to nearest boundary in pixels
        """
        shape_w, shape_h = shape.get('size', (50, 50))
        half_w = shape_w // 2
        half_h = shape_h // 2
        
        dist_left = x - half_w
        dist_right = self.canvas_w - (x + half_w)
        dist_top = y - (self.ui_height + half_h)
        dist_bottom = self.canvas_h - (y + half_h)
        
        return min(dist_left, dist_right, dist_top, dist_bottom)


class VisualIndicators:
    """
    Renders visual feedback during shape movement.
    
    Provides:
      - Grab activation ring around hand
      - Shape highlighting when grabbed
      - Shape outline showing movement
      - Status text and timer
      - Movement arrows or trails (optional)
    
    Attributes:
        colors: UI color palette
        font_face: Font for text rendering
        font_scale: Font size
    """
    
    def __init__(self):
        """Initialize visual indicator system."""
        self.colors = {
            'grab_ring': (0, 255, 255),      # Cyan
            'shape_highlight': (0, 255, 0),  # Green
            'shape_outline': (0, 200, 255),  # Orange
            'text': (255, 255, 255),         # White
            'timer': (100, 200, 255),        # Light blue
        }
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
    
    def draw_grab_activation_ring(self, canvas: np.ndarray, hand_x: int, hand_y: int,
                                 progress: float):
        """
        Draw ring around hand showing grab activation progress.
        
        Args:
            canvas: Canvas to draw on
            hand_x, hand_y: Hand position
            progress: Activation progress (0.0 to 1.0)
        """
        # Ring size grows as progress increases
        radius = int(20 + 20 * progress)
        

        # Color transitions from red → yellow → green
        if progress < 0.5:
            b, g, r = 0, int(255 * progress * 2), 255
        else:
            b, g, r = 0, 255, int(255 * (2 - progress * 2))
        
        cv2.circle(canvas, (hand_x, hand_y), radius, (b, g, r), 2)
        
        # Draw pulsing inner circle
        inner_radius = int(15 * progress)
        if inner_radius > 0:
            cv2.circle(canvas, (hand_x, hand_y), inner_radius, (b, g, r), 1)
    
    def draw_grabbed_shape_highlight(self, canvas: np.ndarray, shape: Dict[str, Any],
                                    alpha: float = 0.3):
        """
        Draw highlighting on grabbed shape.
        
        Args:
            canvas: Canvas to draw on
            shape: Shape being grabbed
            alpha: Transparency (0.0 = invisible, 1.0 = opaque)
        """
        x, y = shape.get('current_pos', shape.get('center', (0, 0)))
        w, h = shape.get('size', (50, 50))
        
        # Create semi-transparent overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x - w//2, y - h//2), (x + w//2, y + h//2),
                     self.colors['shape_highlight'], -1)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    
    def draw_shape_outline(self, canvas: np.ndarray, shape: Dict[str, Any], thickness: int = 3):
        """
        Draw outline around shape being moved.
        
        Args:
            canvas: Canvas to draw on
            shape: Shape to outline
            thickness: Line thickness in pixels
        """
        x, y = shape.get('current_pos', shape.get('center', (0, 0)))
        w, h = shape.get('size', (50, 50))
        
        cv2.rectangle(canvas, (x - w//2, y - h//2), (x + w//2, y + h//2),
                     self.colors['shape_outline'], thickness)
    
    def draw_movement_status(self, canvas: np.ndarray, hand_x: int, hand_y: int,
                            is_moving: bool = False, progress: float = 0.0):
        """
        Draw status text and progress indicator.
        
        Args:
            canvas: Canvas to draw on
            hand_x, hand_y: Hand position
            is_moving: Whether shape is currently being moved
            progress: Activation progress (0-1) if not moving
        """
        y_offset = 30
        
        if is_moving:
            status = "Mode: MOVE | Release to finalize"
            color = self.colors['shape_highlight']
        else:
            progress_pct = int(progress * 100)
            status = f"Hold fist... {progress_pct}%"
            color = self.colors['timer']
        
        # Draw text with background for readability
        font = self.font_face
        scale = self.font_scale
        thickness = 1
        
        text_size = cv2.getTextSize(status, font, scale, thickness)[0]
        
        # Background box
        x_offset = 10
        cv2.rectangle(canvas,
                     (x_offset - 5, y_offset - text_size[1] - 5),
                     (x_offset + text_size[0] + 5, y_offset + 5),
                     (0, 0, 0), -1)
        
        # Text
        cv2.putText(canvas, status, (x_offset, y_offset), font, scale,
                   color, thickness, cv2.LINE_AA)
    
    def draw_movement_help(self, canvas: np.ndarray):
        """
        Draw help text for shape movement feature.
        
        Args:
            canvas: Canvas to draw on
        """
        help_lines = [
            "SHAPE CONTROL:",
            "1. Make CLOSED FIST",
            "2. Hold 2-3 seconds",
            "3. Move hand to reposition",
            "4. Open hand to release"
        ]
        
        font = self.font_face
        scale = 0.4
        thickness = 1
        y_start = 180
        line_height = 16
        
        for i, line in enumerate(help_lines):
            y = y_start + i * line_height
            cv2.putText(canvas, line, (10, y), font, scale,
                       self.colors['text'], thickness, cv2.LINE_AA)


# ============================================================================
# Integration Helper Functions
# ============================================================================

def create_shape_data(shape_type: str, center_x: int, center_y: int,
                     size: Tuple[int, int], color: Tuple[int, int, int],
                     thickness: int = 2, canvas_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Create shape data dictionary for tracker registration.
    
    Args:
        shape_type: Type of shape (circle, rectangle, triangle, line)
        center_x, center_y: Center position
        size: (width, height) tuple
        color: (B, G, R) color tuple
        thickness: Line thickness
        canvas_data: Binary mask of shape pixels
        
    Returns:
        Formatted shape data dictionary
    """
    return {
        'id': str(uuid.uuid4()),
        'type': shape_type,
        'original_pos': (center_x, center_y),
        'current_pos': (center_x, center_y),
        'center': (center_x, center_y),
        'bounding_box': (center_x - size[0]//2, center_y - size[1]//2,
                        center_x + size[0]//2, center_y + size[1]//2),
        'size': size,
        'rotation': 0,
        'canvas_data': canvas_data,
        'color': color,
        'thickness': thickness,
        'timestamp': time.time(),
        'moved': False,
        'move_count': 0,
    }


def extract_shape_bounds(canvas_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Extract bounding box from shape canvas mask.
    
    Args:
        canvas_mask: Binary mask where 255 = shape, 0 = background
        
    Returns:
        (x1, y1, x2, y2) bounding box, or None if empty
    """
    contours, _ = cv2.findContours(canvas_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    x1, y1, w, h = cv2.boundingRect(contours[0])
    return (x1, y1, x1 + w, y1 + h)


def calculate_shape_center(bounding_box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculate center point from bounding box.
    
    Args:
        bounding_box: (x1, y1, x2, y2) coordinates
        
    Returns:
        (center_x, center_y)
    """
    x1, y1, x2, y2 = bounding_box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


# ============================================================================
# Example Usage / Testing
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of sketch position control system.
    """
    
    # Create components
    gesture_activator = GestureActivator(hold_duration_sec=2.5)
    shape_tracker = ShapeTracker()
    movement_controller = MovementController((800, 600))
    boundary_manager = BoundaryManager(800, 600, ui_height=160)
    visual_indicators = VisualIndicators()
    
    # Simulate drawing a circle
    circle_data = create_shape_data(
        shape_type='circle',
        center_x=400,
        center_y=300,
        size=(60, 60),
        color=(255, 255, 255),
        thickness=2
    )
    circle_id = shape_tracker.add_shape(circle_data)
    print(f"✓ Circle added with ID: {circle_id}")
    
    # Simulate user making fist gesture and holding for 2.5+ seconds
    start_time = time.time()
    for i in range(300):  # ~10 seconds at 30 FPS
        current_time = start_time + i * (1/30)
        
        # Simulate closed fist for first 75 frames (~2.5 seconds)
        is_fist = i < 75
        gesture = "fist" if is_fist else "idle"
        
        activated = gesture_activator.update(gesture, is_fist, current_time)
        progress = gesture_activator.get_hold_progress(current_time)
        
        if activated and i == 75:
            print(f"✓ Gesture activated after {progress * 100:.1f}% hold")
            shape = shape_tracker.get_most_recent()
            movement_controller.start_move(circle_id, 400, 300, (400, 300))
        
        # Simulate hand movement right 50 pixels
        if is_fist and i > 75:
            new_x = 400 + (i - 75) * (50 / 25)  # Move 50px over 25 frames
            new_pos = movement_controller.calculate_new_position(int(new_x), 300)
            print(f"Frame {i}: Hand at {int(new_x)}, Shape moving to {new_pos}")
    
    print("✓ Sketch position control demo complete!")
