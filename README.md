
# XightMD - Chest X-Ray Multi-Label Classification

## Model Performance

- **Current F1 Score**: 0.22 (epoch 45, still training)
- **Baseline**: Random = 0.067, so 3.27x improvement
- **Architecture**: EfficientNet-B0 with 2-layer classifier
- **Dataset**: NIH Chest X-ray 14, ~10k samples

## Architecture

### OptimizedLungClassifier (`lung_classifier.py`)
```python
# EfficientNet-B0 backbone
# Custom classifier: 1280 -> 512 -> 15 outputs
# No sigmoid (BCEWithLogitsLoss handles it)
# Dropout: 0.3, 0.2
```

### SimpleLungModel (`balanced_lung_trainer.py`)
```python
# ResNet18 backbone  
# Direct FC: 512 -> 15 outputs
# Used for balanced training experiments
```

## Multi-Label Classification

**15 Classes**:
```
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, 
Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, 
Pleural Thickening, Pneumonia, Pneumothorax, No Finding
```

**Problem**: Severe class imbalance
- "No Finding": ~60% of samples
- "Hernia": ~0.2% of samples

**Solution**: Balanced sampling in `BalancedNIHDataset`
- Limits "No Finding" to 25% of training data
- Ensures minimum samples per pathology class

## Training Configuration

### Data Processing
```python
# Input: 224x224 RGB (grayscale X-rays converted)
# Augmentation: RandomCrop, HorizontalFlip, Rotation(5°), ColorJitter
# Normalization: ImageNet stats [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
```

### Training Parameters
```python
# Loss: BCEWithLogitsLoss (multi-label)
# Optimizer: AdamW, lr=0.001, weight_decay=1e-4
# Scheduler: ReduceLROnPlateau(patience=3, factor=0.5)
# Batch size: 16
# Gradient clipping: max_norm=1.0
```

## Thresholds

Per-condition optimized thresholds (not standard 0.5):
```python
'Pneumothorax': 0.25,     # Critical condition
'Mass': 0.20,             # Cancer screening
'Pneumonia': 0.22,        
'Atelectasis': 0.18,
'Hernia': 0.50,           # Rare condition
'No Finding': 0.60        # High threshold to reduce false normals
```

## Implementation Details

### Prediction Pipeline
```python
def predict(self, image_path: str) -> Dict[str, float]:
    # Load image -> RGB conversion -> resize(224,224)
    # Forward pass -> sigmoid(logits) -> numpy
    # Return dict of condition:probability
```

### Class Imbalance Handling
```python
class BalancedNIHDataset:
    # Separate "No Finding" from pathology samples
    # Limit pathology samples per condition
    # Shuffle and create balanced final dataset
```

## File Structure

```
backend/utils/lung_classifier.py     # Main model definition and inference
balanced_lung_trainer.py            # Balanced training pipeline
train_optimized.py                  # Full training with metrics tracking
```

## Current Results (Epoch 45)

**Macro F1**: 0.22
- Better performing classes: Cardiomegaly (~0.35), Pneumonia (~0.28)
- Challenging classes: Hernia, Fibrosis (limited training data)

**Training stability**: Loss decreasing, F1 improving consistently

## Technical Challenges Solved

1. **Double sigmoid issue**: Fixed BCEWithLogitsLoss + model architecture mismatch
2. **Class imbalance**: Implemented balanced sampling strategy  
3. **14 vs 15 class mismatch**: Unified architecture to handle all LABELS
4. **Dataset inconsistency**: Standardized on NIH Chest X-ray 14

## Benchmarking

```
NIH Chest X-ray 14 Literature:
- Basic CNN: F1 = 0.15-0.20
- ResNet/DenseNet: F1 = 0.20-0.30  ← Current range
- Ensemble methods: F1 = 0.30-0.40
- SOTA research: F1 = 0.40+
```

## Dependencies

```
torch==2.7.1
torchvision==0.22.1
datasets==3.6.0
scikit-learn==1.7.0
Pillow==11.2.1
```

## Usage

```python
# Load model
classifier = LungClassifierTrainer('path/to/model.pth')

# Inference
predictions = classifier.predict('xray.jpg')
# Returns: {'Pneumonia': 0.78, 'Effusion': 0.23, ...}

# Apply thresholds
significance = classifier.get_statistical_significance(predictions)
```

## Training Commands

```bash
# Balanced training (handles class imbalance)
python balanced_lung_trainer.py

# Full training pipeline  
python train_optimized.py --epochs 50 --batch-size 16
```

## Model Files

- Best model: `models/lung_classifier_BEST.pth`
- Training history: `models/training_history_optimized.json`
- Balanced model: `balanced_lung_model.pth`