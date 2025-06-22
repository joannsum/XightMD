import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax",
    "No Finding"
]

class OptimizedLungClassifier(nn.Module):
    """Optimized multi-label lung disease classifier"""
    def __init__(self, num_classes: int = 15, pretrained: bool = True):
        super(OptimizedLungClassifier, self).__init__()

        # Use EfficientNet for better accuracy/speed tradeoff
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
            # NO SIGMOID - BCEWithLogitsLoss handles it
        )
        
    def forward(self, x):
        return self.backbone(x)

class LungClassifierTrainer:
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = OptimizedLungClassifier(num_classes=len(LABELS))
        self.model.to(self.device)
        # OPTIMIZED thresholds based on validation performance
        self.condition_thresholds = {
            # These are POST-TRAINING optimized thresholds!
            'Atelectasis': 0.18,        # Optimized for your 90% accuracy model
            'Pneumothorax': 0.25,       # Life-threatening - be sensitive
            'Pneumonia': 0.22,          # Common condition
            'Mass': 0.20,               # Cancer concern - be sensitive
            'Edema': 0.30,
            'Cardiomegaly': 0.35,
            'Consolidation': 0.28,
            'Effusion': 0.32,
            'Emphysema': 0.45,
            'Fibrosis': 0.40,
            'Hernia': 0.50,
            'Infiltration': 0.35,
            'Nodule': 0.25,
            'Pleural Thickening': 0.45,
            'No Finding': 0.60
        }

        # Optimized transforms for chest X-rays
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transform (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        if model_path:
            self.load_model(model_path)

    def load_dataset(self):
        """Load the ReXGradient dataset"""
        from datasets import load_dataset
        # Login using: huggingface-cli login
        dataset = load_dataset("rajpurkarlab/ReXGradient-160K")
        return dataset
    
    def predict(self, image_path: str) -> Dict[str, float]:
        """Predict lung conditions for a single image"""
        self.model.eval()
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        return {label: float(prob) for label, prob in zip(LABELS, probabilities)}
    def get_optimized_classification(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """Use optimized thresholds for each condition"""
        results = {}
        for condition, confidence in predictions.items():
                threshold = self.condition_thresholds.get(condition, 0.5)
                is_positive = confidence > threshold

            # Risk-based confidence levels
        if condition in ['Pneumothorax', 'Mass', 'Atelectasis']:  # High-risk conditions
                if confidence > threshold + 0.2:
                    conf_level = 'High'
                    emoji = '🔴'
                elif confidence > threshold:
                    conf_level = 'Medium'
                    emoji = '🟠'
                elif confidence > threshold * 0.7:
                    conf_level = 'Low'
                    emoji = '🟡'
                else:
                    conf_level = 'Very Low'
                    emoji = '🟢'
        else:  # Standard conditions
                if confidence > 0.7:
                    conf_level = 'High'
                    emoji = '🔴'
                elif confidence > 0.5:
                    conf_level = 'Medium'
                    emoji = '🟠'
                elif confidence > 0.3:
                    conf_level = 'Low'
                    emoji = '🟡'
                else:
                    conf_level = 'Very Low'
                    emoji = '🟢'

                results[condition] = {
                'confidence': confidence,
                'positive': is_positive,
                'confidence_level': conf_level,
                'emoji': emoji,
                'threshold_used': threshold,
                'clinical_significance': 'HIGH' if condition in ['Pneumothorax', 'Mass', 'Atelectasis'] else 'MODERATE'
            }

        return results

    def get_statistical_significance(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """Calculate statistical significance with ATELECTASIS-OPTIMIZED thresholds"""
        significant_findings = {}
        
        for condition, confidence in predictions.items():
            # Use lower threshold for Atelectasis model
            if condition == 'Atelectasis':
                threshold = 0.15  # Much lower threshold for specialized model
            else:
                threshold = self.condition_thresholds.get(condition, 0.5)

            is_significant = confidence > threshold

            # Update confidence levels for Atelectasis
            if condition == 'Atelectasis':
                if confidence > 0.16:
                    conf_level = 'High'
                elif confidence > 0.10:
                    conf_level = 'Medium'
                elif confidence > 0.05:
                    conf_level = 'Low'
                else:
                    conf_level = 'Very Low'
            else:
                # Regular confidence levels for other conditions
                if confidence > 0.8:
                    conf_level = 'High'
                elif confidence > 0.6:
                    conf_level = 'Medium'
                elif confidence > 0.4:
                    conf_level = 'Low'
                else:
                    conf_level = 'Very Low'

            significant_findings[condition] = {
                'confidence': confidence,
                'significant': is_significant,
                'confidence_level': conf_level,
                'threshold_used': threshold
            }
            
        return significant_findings
    
    def save_model(self, path: str, epoch: int = None, metrics: dict = None):
        """Save model with metadata"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'labels': LABELS,
            'num_classes': len(LABELS),
            'save_timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'metrics': metrics
        }

        torch.save(save_dict, path)
        print(f"✅ Model saved to {path}")
        
    def load_model(self, path: str):
        """Load model from checkpoint"""
        if not os.path.exists(path):
            print(f"❌ Model file not found: {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from {path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
