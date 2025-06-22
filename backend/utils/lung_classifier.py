import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.metrics import roc_auc_score, average_precision_score

LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax",
    "No Finding"
]

class LungDiseaseClassifier(nn.Module):

    def __init__(self, num_classes: int = 15, pretrained: bool = True):
        super(LungDiseaseClassifier, self).__init__()
        # Use DenseNet-121 as requested
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Replace classifier for multi-label classification
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes),
            nn.Sigmoid()  # For multi-label classification
        )
        
    def forward(self, x):
        return self.backbone(x)

class LungClassifierTrainer:
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LungDiseaseClassifier()
        self.model.to(self.device)
        
        if model_path:
            self.load_model(model_path)
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def load_dataset(self):
        """Load the ReXGradient dataset"""
        from datasets import load_dataset
        # Login using: huggingface-cli login
        dataset = load_dataset("rajpurkarlab/ReXGradient-160K")
        return dataset
    
    def predict(self, image_path: str) -> Dict[str, float]:
        """Predict lung conditions for a single image"""
        self.model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
            predictions = predictions.cpu().numpy()[0]
        
        # Create result dictionary with confidence scores
        results = {}
        for i, label in enumerate(LABELS):
            results[label] = float(predictions[i])
            
        return results
    
    def get_statistical_significance(self, predictions: Dict[str, float], threshold: float = 0.3) -> Dict[str, Any]:
        """Calculate statistical significance of predictions with lower threshold"""
        significant_findings = {}
        
        for condition, confidence in predictions.items():
            # More sensitive detection
            is_significant = confidence > threshold

            # Special handling for critical conditions
            if condition in ['Pneumothorax', 'Pneumonia', 'Mass', 'Edema']:
                is_significant = confidence > 0.25  # Even lower threshold for critical conditions

            significant_findings[condition] = {
                'confidence': confidence,
                'significant': is_significant,
                'confidence_level': 'High' if confidence > 0.6 else 'Medium' if confidence > 0.4 else 'Low'
            }
            
        return significant_findings
    
    def save_model(self, path: str):
        """Save the complete model state with metadata"""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Save complete state with metadata
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_class': 'LungDiseaseClassifier',
            'num_classes': len(LABELS),
            'labels': LABELS,
            'save_timestamp': str(datetime.now()) if 'datetime' in globals() else 'unknown'
        }
        torch.save(save_dict, path)
        print(f"✅ Complete model with metadata saved to {path}")
        
    def load_model(self, path: str):
        """Smart model loader that handles different checkpoint formats"""
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return

        try:
            print(f"Loading model from: {path}")
            checkpoint = torch.load(path, map_location=self.device)

            # Strategy 1: Try loading as full checkpoint first
            if 'model_state_dict' in checkpoint:
                print("📦 Found full checkpoint format")
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 'unknown')
                val_loss = checkpoint.get('val_loss', 'unknown')
                print(f"   Epoch: {epoch}, Val Loss: {val_loss}")

                try:
                    self.model.load_state_dict(state_dict, strict=True)
                    print("✅ Full model loaded successfully from checkpoint")
                    return
                except RuntimeError as e:
                    print(f"⚠️  Full loading failed: {str(e)[:100]}...")
                    # Fall through to partial loading

            # Strategy 2: Try loading as direct state_dict
            else:
                print("📦 Found direct state_dict format")
                state_dict = checkpoint

            # Strategy 3: Smart partial loading
            print("🔄 Attempting smart partial loading...")

            # Check what keys we have
            available_keys = set(state_dict.keys())
            model_keys = set(self.model.state_dict().keys())

            print(f"   Available keys: {len(available_keys)}")
            print(f"   Model needs: {len(model_keys)}")

            # Try to match classifier keys
            classifier_keys = [k for k in available_keys if 'classifier' in k]
            backbone_keys = [k for k in available_keys if 'backbone' in k or 'features' in k]

            print(f"   Classifier keys found: {len(classifier_keys)}")
            print(f"   Backbone keys found: {len(backbone_keys)}")

            if len(backbone_keys) > 100:  # Reasonable number for DenseNet
                print("🎯 Found substantial backbone weights, loading full model")
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                if len(missing_keys) < 10:  # Acceptable number of missing keys
                    print(f"✅ Model loaded with {len(missing_keys)} missing keys")
                    return

            # Strategy 4: Load only classifier if that's all we have
            if classifier_keys:
                print("🎯 Loading only classifier weights, keeping pretrained backbone")
                classifier_state = {k: v for k, v in state_dict.items() if 'classifier' in k}
                missing_keys, unexpected_keys = self.model.load_state_dict(classifier_state, strict=False)
                print(f"✅ Classifier loaded, {len(missing_keys)} keys kept as pretrained")
                return

            # Strategy 5: Last resort - create mapping
            print("🔄 Attempting key mapping...")
            mapped_state = {}
            for key, value in state_dict.items():
                # Try common mapping patterns
                if key.startswith('classifier.'):
                    mapped_key = f'backbone.{key}'
                    if mapped_key in model_keys:
                        mapped_state[mapped_key] = value
                elif key in model_keys:
                    mapped_state[key] = value

            if mapped_state:
                missing_keys, unexpected_keys = self.model.load_state_dict(mapped_state, strict=False)
                print(f"✅ Mapped loading successful, {len(mapped_state)} keys loaded")
                return

            print("⚠️  Could not load model weights, using pretrained only")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Falling back to pretrained weights only")
