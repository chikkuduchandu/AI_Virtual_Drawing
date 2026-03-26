# Phase 3 & 5 Implementation Guide
## Image Mapping Improvement + Sketch Position Control Integration

**Date**: March 26, 2026  
**Version**: 1.0  
**Status**: Ready for Implementation  

---

## Overview

This document provides step-by-step implementation instructions for:
1. **Phase 3**: Image Mapping ML Improvement (Shape Recognition)
2. **Phase 5**: Sketch Position Control (Gesture-Based Shape Movement)

Both phases are designed to integrate seamlessly without affecting existing functionality.

---

## TABLE OF CONTENTS

1. [Phase 3: ML Implementation](#phase-3-ml-implementation)
2. [Phase 5: Sketch Control Implementation](#phase-5-sketch-control-implementation)
3. [Integration Points](#integration-points)
4. [Testing Procedures](#testing-procedures)
5. [Troubleshooting](#troubleshooting)

---

# PHASE 3: ML IMPLEMENTATION

## Step 1: Enhanced Dataset Generation

### File: `utils/dataset_generator.py`

**Objective**: Increase training data from 4,000 to 20,000 samples with better augmentation.

**Current Code Structure**:
```python
# Current generator creates:
# - 1000 samples per class
# - Basic augmentation (noise, affine)
# - Classes: circle, square, triangle, line

def generate_shape(shape_type, img_size=28, noise_level=0.1):
    """Generate synthetic shape image"""
    # Current implementation
```

**Required Changes**:

1. **Increase sample count**
   ```python
   NUM_SAMPLES_PER_CLASS = 5000  # was 1000
   ```

2. **Add augmentation variations**
   ```python
   def _generate_with_augmentation(shape, num_samples):
       """Generate multiple variations of same shape"""
       variations = []
       
       # Rotation variation: -45 to +45 degrees
       for angle in np.linspace(-45, 45, 5):
           img = rotate_shape(shape, angle)
           variations.append(img)
       
       # Scale variation: 0.7x to 1.3x
       for scale in np.linspace(0.7, 1.3, 4):
           img = scale_shape(shape, scale)
           variations.append(img)
       
       # Thickness variation: 1-3 pixels
       for thickness in [1, 2, 3]:
           img = draw_with_thickness(shape, thickness)
           variations.append(img)
       
       # Skew/shear: ±15 degrees
       for skew in np.linspace(-15, 15, 3):
           img = skew_shape(shape, skew)
           variations.append(img)
       
       return variations
   ```

3. **Implement in dataset generation**
   ```python
   def generate_training_dataset(output_dir='data/shapes'):
       """Generate 20,000 training samples total"""
       
       for shape_class in ['circle', 'square', 'triangle', 'line']:
           for i in range(NUM_SAMPLES_PER_CLASS):
               # Generate base shape
               img = generate_shape(shape_class)
               
               # Apply augmentation
               for aug_type in ['rotate', 'scale', 'thickness', 'skew', 'noise']:
                   aug_img = apply_augmentation(img, aug_type)
                   save_training_sample(aug_img, shape_class, i)
   ```

**Time Estimate**: 4 hours

---

## Step 2: Create Enhanced CNN Architecture

### File: `ml/drawing_cnn_v2.py` (NEW)

**Objective**: Implement improved CNN with 4 conv blocks, BatchNorm, and progressive dropout.

**Complete Implementation**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class DrawingCNNv2(nn.Module):
    """
    Enhanced CNN for shape classification.
    
    Architecture:
    - Conv Block 1: 1→16 channels
    - Conv Block 2: 16→32 channels (MaxPool)
    - Conv Block 3: 32→64 channels (MaxPool)
    - Conv Block 4: 64→128 channels (MaxPool)
    - Dense: 256 units
    - Output: num_classes
    
    Features:
    - Batch normalization at each layer
    - Progressive dropout (0.2→0.4)
    - Confidence tracking
    """
    
    def __init__(self, num_classes: int = 4):
        super(DrawingCNNv2, self).__init__()
        
        # Block 1: Initial feature extraction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.2)
        
        # Block 2: Edge/corner detection
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Block 3: Mid-level features
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.3)
        
        # Block 4: High-level features
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout_fc = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)
        
        # For confidence tracking
        self.last_confidence = 0.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network"""
        
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x: torch.Tensor) -> tuple:
        """
        Get prediction and confidence score.
        
        Returns:
            (class_idx, confidence_score)
        """
        with torch.no_grad():
            output = self.forward(x)
            probs = F.softmax(output, dim=1)
            confidence, class_idx = torch.max(probs, 1)
        
        self.last_confidence = confidence.item()
        return int(class_idx.item()), float(confidence.item())


class DrawingClassifierV2:
    """Wrapper for model loading and inference"""
    
    def __init__(self, model_path: str = 'ml/drawing_cnn_v2.pkl',
                 class_labels: list = None, device: str = 'cpu'):
        self.model_path = Path(model_path)
        self.device = device
        self.class_labels = class_labels or ['circle', 'square', 'triangle', 'line']
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        import pickle
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image: torch.Tensor) -> tuple:
        """
        Predict shape class and confidence.
        
        Args:
            image: Input image tensor (28x28 or compatible)
            
        Returns:
            (class_name, confidence_score)
        """
        # Ensure correct shape and device
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            output = self.model.forward(image)
            probs = F.softmax(output, dim=1)
            confidence, class_idx = torch.max(probs, 1)
        
        class_name = self.class_labels[int(class_idx.item())]
        confidence_score = float(confidence.item())
        
        return class_name, confidence_score
```

**Key Files**:
- `ml/drawing_cnn_v2.py` - NEW (complete module above)

**Time Estimate**: 6 hours

---

## Step 3: Create Enhanced Training Script

### File: `train_drawing_cnn_v2.py` (NEW)

**Objective**: Train CNN on 20,000 samples with proper validation and early stopping.

**Implementation**:

```python
#!/usr/bin/env python3
"""
Training script for enhanced CNN shape classifier.

Trains DrawingCNNv2 on synthetic shape dataset with:
- 20,000 samples (5000 per class)
- 150 epochs
- Learning rate decay
- Early stopping
- Validation monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import pickle
import time

from ml.drawing_cnn_v2 import DrawingCNNv2
from utils.dataset_generator import generate_training_dataset

# Configuration
EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EARLY_STOPPING_PATIENCE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = total_loss / len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

def train_cnn_v2(data_dir='data/shapes', output_path='ml/drawing_cnn_v2.pkl'):
    """
    Main training function.
    
    Args:
        data_dir: Directory containing training data
        output_path: Where to save trained model
    """
    
    print("[1/4] Generating training dataset...")
    X_train, y_train, class_names = generate_training_dataset(data_dir)
    
    # Split into train/val (80/20)
    val_split = int(0.8 * len(X_train))
    X_train, X_val = X_train[:val_split], X_train[val_split:]
    y_train, y_val = y_train[:val_split], y_train[val_split:]
    
    # Create data loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(),
                                  torch.from_numpy(y_train).long())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(),
                               torch.from_numpy(y_val).long())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("[2/4] Initializing model...")
    model = DrawingCNNv2(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                          weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    print("[3/4] Training model...")
    best_val_acc = 0
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                           optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    training_time = time.time() - start_time
    
    # Load best model and save
    print("[4/4] Saving model...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✓ Training complete!")
    print(f"  - Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  - Training time: {training_time/60:.1f} minutes")
    print(f"  - Model saved to: {output_path}")
    
    return model, best_val_acc

if __name__ == "__main__":
    import sys
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'ml/drawing_cnn_v2.pkl'
    train_cnn_v2(output_path=output_path)
```

**Run Command**:
```bash
python train_drawing_cnn_v2.py
```

**Time Estimate**: 4 hours (script creation + execution monitoring)

---

## Step 4: Model Integration

### File: `utils/shape_mlp_ai.py` (MODIFY)

**Objective**: Switch from MLP to CNN v2 with hybrid verification.

**Changes**:

1. **Add CNN import**:
```python
from ml.drawing_cnn_v2 import DrawingClassifierV2

_CNN_V2_CLASSIFIER = None

def get_cnn_v2_classifier():
    """Lazy load CNN v2 model"""
    global _CNN_V2_CLASSIFIER
    if _CNN_V2_CLASSIFIER is None:
        try:
            _CNN_V2_CLASSIFIER = DrawingClassifierV2(
                model_path='ml/drawing_cnn_v2.pkl',
                class_labels=['circle', 'square', 'triangle', 'line']
            )
        except Exception as e:
            print(f"Warning: CNN v2 load failed: {e}, using MLP fallback")
            _CNN_V2_CLASSIFIER = None
    return _CNN_V2_CLASSIFIER
```

2. **Create enhanced detection function**:
```python
def detect_and_snap_mlp_enhanced(raw_pts, canvas_shape):
    """
    Enhanced shape detection with CNN v2 + hybrid verification.
    
    Fallback chain:
    1. CNN v2 (confidence > 0.85) → Accept
    2. CNN v2 (confidence 0.70-0.85) + geometric verification → Confirm
    3. Rule-based fallback → Use geometric heuristics
    """
    
    if not raw_pts or len(raw_pts) < 20:
        return None, None
    
    # Preprocess stroke to 28x28
    processed_image = _preprocess_stroke(raw_pts, canvas_shape)
    if processed_image is None:
        return None, None
    
    # Try CNN v2 first
    cnn_clf = get_cnn_v2_classifier()
    if cnn_clf:
        try:
            shape_pred, confidence = cnn_clf.predict(processed_image)
            
            if confidence > 0.85:
                # High confidence - accept directly
                return shape_pred, _make_clean_shape(shape_pred, raw_pts)
            
            elif confidence > 0.70:
                # Medium confidence - verify with geometry
                verified = _verify_prediction_geometry(raw_pts, shape_pred)
                if verified:
                    return verified, _make_clean_shape(verified, raw_pts)
            
            # Fall through to rule-based
        except Exception as e:
            print(f"CNN prediction error: {e}")
    
    # Fallback to rule-based system
    return detect_and_snap(raw_pts, canvas_shape)

def _verify_prediction_geometry(raw_pts, predicted_shape):
    """
    Cross-check CNN prediction with geometric heuristics.
    
    Returns verified shape name or None if geometry contradicts.
    """
    circularity = _calculate_circularity(raw_pts)
    corners = _count_corners(raw_pts)
    straightness = _calculate_straightness(raw_pts)
    
    # Verification rules
    if predicted_shape == "circle":
        if circularity > 0.80:
            return "circle"
    elif predicted_shape == "rectangle" or predicted_shape == "square":
        if 3 <= corners <= 5 and circularity < 0.75:
            return "rectangle"
    elif predicted_shape == "triangle":
        if 2.5 <= corners <= 3.5:
            return "triangle"
    elif predicted_shape == "line":
        if straightness > 0.80:
            return "line"
    
    # Verification failed
    return None
```

**Time Estimate**: 4 hours

---

# PHASE 5: SKETCH CONTROL IMPLEMENTATION

## Integration Steps

### Step 1: Import and Initialize

**File**: `modules/drawing_2d.py` (ADD imports at top)

```python
# After existing imports
from modules.sketch_position_control import (
    GestureActivator, ShapeTracker, MovementController,
    BoundaryManager, VisualIndicators, create_shape_data
)
```

### Step 2: Initialize in DrawState

**File**: `modules/drawing_2d.py` (in `DrawState.__init__`)

Add after existing initialization:

```python
# Sketch position control
self.sketch_move_enabled = True  # Check config
self.gesture_activator = GestureActivator(hold_duration_sec=2.5)
self.shape_tracker = ShapeTracker()
self.movement_controller = MovementController((SCREEN_W, SCREEN_H))
self.boundary_manager = BoundaryManager(SCREEN_W, SCREEN_H, UI_H)
self.visual_indicators = VisualIndicators()
self.is_moving_shape = False
self.shape_move_timeout = 0.0
```

### Step 3: Register Shapes on Snap

**File**: `modules/drawing_2d.py` (in `try_snap_shape()`)

After shape snapping succeeds, add:

```python
# Register shape for position control
shape_data = create_shape_data(
    shape_type=snapped_label,
    center_x=snippet_x,
    center_y=snippet_y,
    size=(roi_w, roi_h),
    color=self.color,
    thickness=self.thickness,
    canvas_data=mask_data
)
shape_id = self.shape_tracker.add_shape(shape_data)
```

### Step 4: Handle Grab Gesture in Main Loop

**File**: `modules/drawing_2d.py` (in gesture handling section)

Add new gesture handling block:

```python
# ── Sketch Position Control ─────────────────────────────────────
if self.sketch_move_enabled and gesture == "closed_fist" and not was_prev:
    # Try to activate shape movement
    is_activated = self.gesture_activator.update("closed_fist",
                                                is_fist=True,
                                                current_time=now)
    progress = self.gesture_activator.get_hold_progress(now)
    
    # Draw grab activation indicator
    self.visual_indicators.draw_grab_activation_ring(
        frame, ix, iy, progress
    )
    
    if is_activated and not self.is_moving_shape:
        # Grab gesture confirmed - start moving shape
        shape = self.shape_tracker.get_most_recent()
        if shape:
            self.movement_controller.start_move(
                shape['id'], ix, iy, shape['current_pos']
            )
            self.is_moving_shape = True
            self.shape_move_timeout = now + 3.0  # 3 second timeout
            show_status("Shape grabbed! Move hand to reposition.")

elif self.is_moving_shape and gesture == "closed_fist":
    # Update shape position as hand moves
    if now > self.shape_move_timeout:
        # Timeout - release shape
        self.is_moving_shape = False
        show_status("Shape released (timeout).")
    else:
        # Calculate new position
        new_pos = self.movement_controller.calculate_new_position(ix, iy)
        shape = self.shape_tracker.get_most_recent()
        
        if shape and new_pos:
            # Apply boundary constraints
            final_pos = self.boundary_manager.clamp_position(
                shape, new_pos[0], new_pos[1]
            )
            
            # Update shape position
            self.shape_tracker.update_shape(shape['id'], {
                'current_pos': final_pos,
                'center': final_pos,
                'moved': True
            })
            
            # Redraw shape at new position
            self._redraw_shape_at_position(ds, shape, final_pos)

elif self.is_moving_shape and gesture != "closed_fist":
    # Hand opened - release shape
    self.is_moving_shape = False
    self.movement_controller.end_move()
    show_status("Shape released.", 2.0)
```

### Step 5: Add Shape Redraw Function

**File**: `modules/drawing_2d.py` (add new method)

```python
def _redraw_shape_at_position(self, ds, shape, new_pos):
    """
    Erase shape from old position and redraw at new position.
    
    Args:
        ds: DrawState instance
        shape: Shape data dictionary
        new_pos: New (x, y) position
    """
    x, y = new_pos
    w, h = shape.get('size', (50, 50))
    
    # Erase old position
    old_x, old_y = shape.get('current_pos', (x, y))
    cv2.rectangle(ds.canvas,
                 (int(old_x - w//2), int(old_y - h//2)),
                 (int(old_x + w//2), int(old_y + h//2)),
                 (0, 0, 0), -1)
    
    # Redraw at new position
    if shape['type'] == 'circle':
        cv2.circle(ds.canvas, (int(x), int(y)), w//2,
                  shape['color'], shape['thickness'])
    else:
        cv2.rectangle(ds.canvas,
                     (int(x - w//2), int(y - h//2)),
                     (int(x + w//2), int(y + h//2)),
                     shape['color'], shape['thickness'])
```

### Step 6: Add Config Settings

**File**: `core/config.py` (add at end)

```python
# Sketch position control settings
SKETCH_MOVE_ENABLED = True
SKETCH_MOVE_GESTURE = "closed_fist"
SKETCH_MOVE_HOLD_TIME = 2.5  # seconds
SKETCH_MOVE_IDLE_TIMEOUT = 3.0  # seconds
SKETCH_MOVE_VISUAL_STYLE = "outline"  # outline, highlight, both
SKETCH_MOVE_BOUNDARY_MODE = "clamp"  # clamp or bounce
```

**Time Estimate**: 6 hours

---

# INTEGRATION POINTS

## Code Flow Diagram

```
Main Drawing Loop
    ↓
Gesture Detection (existing)
    ├─→ "draw" gesture → Draw on canvas (existing)
    ├─→ "open_palm" → Clear canvas (existing)
    ├─→ "closed_fist" → NEW: Check shape move
    │       ↓
    │   GestureActivator.update()
    │       ├─→ Hold < 2.5s: Show progress ring
    │       ├─→ Hold ≥ 2.5s: Activate grab
    │       └─→ Return True if activated
    │
    └─→ If is_moving_shape:
            ↓
        MovementController.calculate_new_position()
            ├─→ Hand delta = current_hand - initial_hand
            ├─→ New shape pos = original_pos + hand_delta
            └─→ Return new position
                    ↓
            BoundaryManager.clamp_position()
                ├─→ Ensure within canvas
                └─→ Return clamped (x, y)
                        ↓
            ShapeTracker.update_shape()
                ├─→ Update shape position
                └─→ Update moved/move_count flags
                        ↓
            Canvas.redraw()
                ├─→ Erase old position
                └─→ Draw at new position
                        ↓
            VisualIndicators.draw_*()
                ├─→ Draw outline
                ├─→ Draw status text
                └─→ Draw highlights
```

---

## Data Flow: Shape Registration to Movement

```
1. Shape Drawn (modules/drawing_2d.py)
   ↓
2. try_snap_shape() called
   ↓
3. Shape snapped successfully
   ├─→ Get shape_type, position, size, color
   ├─→ Extract canvas mask
   └─→ Create shape_data dict
       ↓
4. ShapeTracker.add_shape(shape_data)
   ├─→ Assign unique ID
   ├─→ Store in shapes[] list
   └─→ Return shape_id
       ↓
5. User makes closed fist
   ├─→ GestureActivator.update()
   ├─→ Hold for 2.5 seconds
   └─→ Returns True (activation confirmed)
       ↓
6. MovementController.start_move()
   ├─→ Store initial hand position
   ├─→ Store initial shape position
   └─→ Set moving_shape_id
       ↓
7. User moves hand (relative motion)
   ├─→ MovementController.calculate_new_position()
   ├─→ hand_delta = new_hand - initial_hand
   ├─→ shape_new_pos = shape_initial + hand_delta
   └─→ Return new_pos
       ↓
8. BoundaryManager.clamp_position()
   ├─→ Check bounds
   ├─→ Clamp if exceeds (keep in canvas)
   └─→ Return final_pos
       ↓
9. ShapeTracker.update_shape()
   ├─→ Update current_pos
   ├─→ Update center
   └─→ Set moved=True, increment move_count
       ↓
10. Canvas redraw
    ├─→ Erase old position
    └─→ Draw at new position
        ↓
11. User opens hand
    ├─→ MovementController.end_move()
    ├─→ GestureActivator.reset()
    ├─→ is_moving_shape = False
    └─→ Shape finalized at new location
```

---

# TESTING PROCEDURES

## Unit Tests

### Phase 3: ML Tests

**File**: `test_shape_mapping_v2.py` (NEW)

```python
import unittest
import torch
from ml.drawing_cnn_v2 import DrawingCNNv2

class TestDrawingCNNv2(unittest.TestCase):
    
    def setUp(self):
        self.model = DrawingCNNv2(num_classes=4)
        self.model.eval()
    
    def test_model_initialization(self):
        """Test model creates without errors"""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.fc2.out_features, 4)
    
    def test_forward_pass(self):
        """Test forward pass with dummy input"""
        x = torch.randn(1, 1, 28, 28)
        output = self.model(x)
        self.assertEqual(output.shape, (1, 4))
    
    def test_inference_shape(self):
        """Test inference output shape"""
        x = torch.randn(5, 1, 28, 28)
        output = self.model(x)
        self.assertEqual(output.shape, (5, 4))
    
    def test_confidence_scores(self):
        """Test confidence scores are in [0, 1]"""
        x = torch.randn(1, 1, 28, 28)
        output = self.model(x)
        probs = torch.softmax(output, dim=1)
        self.assertTrue(torch.all(probs >= 0) and torch.all(probs <= 1))

if __name__ == '__main__':
    unittest.main()
```

### Phase 5: Sketch Control Tests

**File**: `test_sketch_move.py` (NEW)

```python
import unittest
import time
from modules.sketch_position_control import (
    GestureActivator, ShapeTracker, MovementController,
    BoundaryManager, create_shape_data
)

class TestGestureActivator(unittest.TestCase):
    
    def setUp(self):
        self.activator = GestureActivator(hold_duration_sec=1.0)
    
    def test_initial_state(self):
        """Test initial state is inactive"""
        self.assertFalse(self.activator.is_confirmed)
        self.assertEqual(self.activator.get_hold_progress(), 0.0)
    
    def test_activation_on_hold(self):
        """Test gesture activates after hold time"""
        start_time = time.time()
        
        # Simulate holding fist for 1+ seconds
        for i in range(100):
            current_time = start_time + i * 0.01
            result = self.activator.update("closed_fist", True, current_time)
            
            if current_time - start_time >= 1.0:
                if i >= 2:  # Account for confirmation frames
                    self.assertTrue(result)
                    break
    
    def test_reset_on_release(self):
        """Test state resets when gesture released"""
        self.activator.update("closed_fist", True)
        self.activator.reset()
        self.assertFalse(self.activator.is_confirmed)

class TestShapeTracker(unittest.TestCase):
    
    def setUp(self):
        self.tracker = ShapeTracker()
    
    def test_add_shape(self):
        """Test adding shape to tracker"""
        shape_data = create_shape_data('circle', 100, 100, (50, 50), (255, 255, 255))
        shape_id = self.tracker.add_shape(shape_data)
        self.assertIsNotNone(shape_id)
        self.assertEqual(len(self.tracker.shapes), 1)
    
    def test_get_most_recent(self):
        """Test retrieving most recent shape"""
        shape1 = create_shape_data('circle', 100, 100, (50, 50), (255, 255, 255))
        shape2 = create_shape_data('square', 200, 200, (40, 40), (255, 255, 255))
        
        self.tracker.add_shape(shape1)
        self.tracker.add_shape(shape2)
        
        recent = self.tracker.get_most_recent()
        self.assertEqual(recent['type'], 'square')

class TestMovementController(unittest.TestCase):
    
    def setUp(self):
        self.controller = MovementController((800, 600))
    
    def test_start_move(self):
        """Test movement start"""
        self.controller.start_move('shape1', 100, 100, (200, 200))
        self.assertTrue(self.controller.is_moving())
    
    def test_relative_movement(self):
        """Test relative hand motion calculation"""
        self.controller.start_move('shape1', 100, 100, (200, 200))
        
        # Hand moves 50px right
        new_pos = self.controller.calculate_new_position(150, 100)
        
        # Shape should move 50px right from original
        self.assertEqual(new_pos[0], 250)
        self.assertEqual(new_pos[1], 200)

class TestBoundaryManager(unittest.TestCase):
    
    def setUp(self):
        self.boundary = BoundaryManager(800, 600, ui_height=160)
    
    def test_clamp_left_boundary(self):
        """Test clamping at left boundary"""
        shape = create_shape_data('circle', -100, 300, (50, 50), (255, 255, 255))
        clamped = self.boundary.clamp_position(shape, -100, 300)
        self.assertGreaterEqual(clamped[0], 25)
    
    def test_clamp_ui_area(self):
        """Test clamping respects UI area"""
        shape = create_shape_data('circle', 400, 100, (50, 50), (255, 255, 255))
        clamped = self.boundary.clamp_position(shape, 400, 100)
        self.assertGreater(clamped[1], 160)  # Below UI

if __name__ == '__main__':
    unittest.main()
```

**Run Tests**:
```bash
python -m pytest test_shape_mapping_v2.py -v
python -m pytest test_sketch_move.py -v
```

## Integration Testing

### Full Pipeline Test

1. **Start application**
   ```bash
   python main.py
   ```

2. **Test Phase 3 (ML)**
   - Draw shape (circle)
   - Verify shape detected with high confidence (> 90%)
   - Repeat for square, triangle, line
   - Check console output for confidence scores

3. **Test Phase 5 (Sketch Control)**
   - Draw a shape
   - Make closed fist → see activation ring
   - Hold fist for 2-3 seconds → shape highlights
   - Move hand right → shape moves right (relative)
   - Open hand → shape finalizes
   - Repeat for different positions

4. **Boundary Testing**
   - Try to drag shape off-screen
   - Verify clamped at boundary
   - Try to drag into UI area
   - Verify clamped below UI

---

# TROUBLESHOOTING

## Phase 3 Issues

### Model Training Fails

**Problem**: RuntimeError during training
**Solution**:
1. Check GPU memory (use `nvidia-smi`)
2. Reduce batch size: `BATCH_SIZE = 32`
3. Run on CPU: Remove GPU check, use `DEVICE = 'cpu'`

### Accuracy Too Low (< 85%)

**Problem**: Validation accuracy not reaching target
**Solutions**:
1. Increase epochs: `EPOCHS = 200`
2. Add more data augmentation
3. Check dataset generation (verify 20k samples exist)
4. Lower learning rate: `LEARNING_RATE = 0.0005`

### Model Won't Load

**Problem**: FileNotFoundError when loading model
**Solution**:
1. Verify file exists: `ls -la ml/drawing_cnn_v2.pkl`
2. Check path is correct
3. Retrain model if corrupted

---

## Phase 5 Issues

### Gesture Won't Activate

**Problem**: Closed fist not triggering movement
**Debug**:
1. Check hand quality: Face camera directly
2. Print gesture label: `print(f"Gesture: {gesture_this_frame}")`
3. Verify config: `SKETCH_MOVE_ENABLED = True`

### Shape Moves Erratically

**Problem**: Shape jumps around instead of smooth motion
**Solutions**:
1. Add hand position smoothing:
   ```python
   hand_pos_smooth = smooth_buffer.push(hand_x, hand_y)
   ```
2. Reduce gesture confirmation frames
3. Check baseline hand jitter

### Shape Goes Off-Screen

**Problem**: Boundary clamping not working
**Debug**:
1. Verify BoundaryManager initialized with correct canvas size
2. Print clamped position: `print(f"Clamped: {final_pos}")`
3. Check shape size in data (might be 0)

### FPS Drops During Movement

**Problem**: Frame rate degrades when moving shapes
**Solutions**:
1. Profile with cProfile: `python -m cProfile main.py`
2. Optimize shape tracker lookup
3. Reduce visual indicator draw complexity
4. Profile canvas redraw operations

---

## Common Integration Errors

### ImportError: Cannot import sketch_position_control

**Solution**:
- Verify file path: `modules/sketch_position_control.py` exists
- Check file has no syntax errors: `python -m py_compile modules/sketch_position_control.py`

### AttributeError: DrawState has no attribute 'gesture_activator'

**Solution**:
- Verify initialization in `DrawState.__init__()` completed
- Check indentation of initialization code
- Verify file saved properly

### ValueError: Shape data missing required field

**Solution**:
- Verify `create_shape_data()` called correctly
- Check all required fields in shape dict
- Review shape data format in module docstring

---

## Performance Profiling

### Measure Inference Time

```python
import time
from ml.drawing_cnn_v2 import DrawingClassifierV2

clf = DrawingClassifierV2()
image = torch.randn(1, 1, 28, 28)

start = time.time()
for _ in range(100):
    prediction, confidence = clf.predict(image)
elapsed = (time.time() - start) / 100

print(f"Avg inference time: {elapsed*1000:.2f}ms")
```

### Measure Movement Overhead

```python
import cProfile
import pstats

def run_drawing_loop():
    # Run main loop for 10 seconds
    pass

profiler = cProfile.Profile()
profiler.enable()
run_drawing_loop()
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

---

## Validation Checklist

### Phase 3 Completion

- [ ] Dataset generation: 20,000 samples created
- [ ] CNN v2 model trained: 150 epochs
- [ ] Validation accuracy: > 92%
- [ ] Test accuracy: > 90%
- [ ] Inference time: < 50ms per sample
- [ ] Model size: < 10MB
- [ ] Fallback chain working
- [ ] Geometric verification functional
- [ ] No memory leaks during inference
- [ ] Works on CPU (tested)

### Phase 5 Completion

- [ ] GestureActivator operational (2.5s activation)
- [ ] MovementController calculates correct deltas
- [ ] BoundaryManager prevents out-of-bounds
- [ ] VisualIndicators render feedback
- [ ] Shape registration working
- [ ] Shape tracker maintains state
- [ ] Relative motion intuitive (tested by hand)
- [ ] No FPS degradation: still >25 FPS
- [ ] Timeout protection: 3s auto-release
- [ ] All edge cases handled

---

## Next Steps After Implementation

1. **User Testing**
   - Get feedback on gesture feel
   - Measure activation time perception
   - Test with different hand shapes

2. **Performance Optimization**
   - Profile with real user patterns
   - Optimize hotspots
   - Cache frequently accessed data

3. **Feature Extensions**
   - Multi-shape movement (future)
   - Shape rotation gesture (future)
   - Copy while moving (future)

4. **Documentation**
   - Create user guides
   - Record demo video
   - Write API documentation

---

**Document Complete**  
**Ready for Implementation**  
**Target Completion**: April 9, 2026
