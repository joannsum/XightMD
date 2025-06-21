import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import roc_auc_score, average_precision_score

LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax"
]

class LungDiseaseClassifier(nn.Module):
    def __init__(self, num_classes: int = 14, pretrained: bool = True):
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
        """Save trained model"""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str):
        """Load trained model"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))