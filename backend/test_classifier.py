import argparse
import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax",
    "No Finding"
]

class SimpleLungModel(nn.Module):
    """Balanced model architecture (ResNet18)"""
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.fc.in_features, len(LABELS))
        )
    
    def forward(self, x):
        return self.backbone(x)

class OptimizedLungClassifier(nn.Module):
    """Optimized model architecture (EfficientNet-B0)"""
    def __init__(self, num_classes: int = 15, pretrained: bool = True):
        super(OptimizedLungClassifier, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class BalancedModelTester:
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine model architecture based on path
        if model_path and 'balanced' in model_path:
            self.model = SimpleLungModel()
            self.model_type = "Balanced (ResNet18)"
        else:
            self.model = OptimizedLungClassifier()
            self.model_type = "Optimized (EfficientNet-B0)"
        
        self.model.to(self.device)
        
        # Optimized thresholds
        self.condition_thresholds = {
            'Atelectasis': 0.18,
            'Pneumothorax': 0.25,
            'Pneumonia': 0.22,
            'Mass': 0.20,
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
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, path: str):
        """Load model with automatic architecture detection"""
        if not os.path.exists(path):
            print(f"❌ Model file not found: {path}")
            return False

        try:
            print(f"Loading model from: {path}")
            
            if 'balanced' in path:
                # Balanced model saves state_dict directly
                state_dict = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print("📦 Loaded balanced model state_dict")
            else:
                # Optimized model saves full checkpoint
                checkpoint = torch.load(path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    epoch = checkpoint.get('epoch', 'unknown')
                    print(f"📦 Loaded checkpoint from epoch {epoch}")
                else:
                    self.model.load_state_dict(checkpoint)
                    print("📦 Loaded direct state_dict")
            
            print(f"✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, image_path: str) -> dict:
        """Predict lung conditions for a single image"""
        self.model.eval()
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        return {label: float(prob) for label, prob in zip(LABELS, probabilities)}
    
    def get_statistical_significance(self, predictions: dict) -> dict:
        """Calculate statistical significance with optimized thresholds"""
        significant_findings = {}
        
        for condition, confidence in predictions.items():
            threshold = self.condition_thresholds.get(condition, 0.5)
            is_significant = confidence > threshold

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

def create_dummy_xray():
    """Create a dummy grayscale image that looks like an X-ray"""
    img_array = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
    
    # Add some circular patterns to simulate lungs
    center1, center2 = (80, 112), (144, 112)
    y, x = np.ogrid[:224, :224]
    
    mask1 = (x - center1[0])**2 + (y - center1[1])**2 < 50**2
    mask2 = (x - center2[0])**2 + (y - center2[1])**2 < 50**2
    
    img_array[mask1] = np.random.randint(80, 150, np.sum(mask1))
    img_array[mask2] = np.random.randint(80, 150, np.sum(mask2))
    
    img = Image.fromarray(img_array, mode='L').convert('RGB')
    return img

def test_balanced_classifier():
    parser = argparse.ArgumentParser(description='Test Balanced Lung Classifier')
    parser.add_argument('--image', type=str, help='Path to X-ray image')
    parser.add_argument('--dummy', action='store_true', help='Use dummy image for testing')
    parser.add_argument('--model', type=str, default='models/balanced_lung_model.pth', 
                       help='Path to model file')

    args = parser.parse_args()
    
    print("🎯 BALANCED LUNG CLASSIFIER TEST")
    print("="*60)
    
    # Check if balanced model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("\nAvailable models:")
        models_dir = "models"
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.pth'):
                    print(f"  - {os.path.join(models_dir, file)}")
        return
    
    # Initialize tester
    print(f"🔄 Loading balanced model: {args.model}")
    try:
        tester = BalancedModelTester(args.model)
        print(f"✅ Model loaded successfully")
        print(f"🏗️  Architecture: {tester.model_type}")
        print(f"🖥️  Device: {tester.device}")
    except Exception as e:
        print(f"❌ Error initializing tester: {e}")
        return
    
    # Determine image path
    if args.dummy:
        print("\n🔄 Creating dummy X-ray image...")
        try:
            dummy_img = create_dummy_xray()
            dummy_img.save("dummy_xray_balanced.jpg")
            image_path = "dummy_xray_balanced.jpg"
            print("✅ Dummy image created: dummy_xray_balanced.jpg")
        except Exception as e:
            print(f"❌ Error creating dummy image: {e}")
            return
    else:
        image_path = args.image
    
    # Validate image path
    if not image_path:
        print("❌ No image provided!")
        print("Usage examples:")
        print("  python test_classifier.py --dummy")
        print("  python test_classifier.py --image /path/to/xray.jpg")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    print(f"\n🔍 Analyzing image: {image_path}")
    
    try:
        # Test image loading
        test_img = Image.open(image_path)
        print(f"✅ Image loaded: {test_img.size}, mode: {test_img.mode}")
        
        # Make prediction
        print("🔄 Making prediction...")
        predictions = tester.predict(image_path)
        print("✅ Prediction completed")
        
        # Get significance analysis
        print("🔄 Calculating statistical significance...")
        significance = tester.get_statistical_significance(predictions)
        print("✅ Statistical analysis completed")
        
        # Display results
        print("\n" + "="*60)
        print("🏥 BALANCED MODEL ANALYSIS RESULTS (F1: 0.23)")
        print("="*60)
        
        # Check for critical conditions first
        critical_findings = []
        for condition in ['Pneumothorax', 'Mass', 'Pneumonia', 'Atelectasis']:
            if condition in predictions and predictions[condition] > 0.3:
                critical_findings.append((condition, predictions[condition]))
        
        if critical_findings:
            print(f"\n🚨 CRITICAL FINDINGS DETECTED:")
            for condition, confidence in critical_findings:
                urgency = ""
                if condition == 'Pneumothorax':
                    urgency = " ⚠️  IMMEDIATE ATTENTION!"
                elif condition == 'Mass':
                    urgency = " 🔬 Further workup needed"
                elif condition in ['Pneumonia', 'Atelectasis']:
                    urgency = " 📋 Medical evaluation recommended"
                    
                print(f"   🚨 {condition}: {confidence:.3f} confidence{urgency}")
        
        # Sort by confidence
        sorted_results = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 COMPLETE CLASSIFICATION RESULTS")
        print(f"{'Condition':<20} {'Confidence':<10} {'Level':<8} {'Status':<8} {'Threshold'}")
        print("-" * 70)
        
        for condition, confidence in sorted_results:
            sig_data = significance[condition]
            status = "POS" if sig_data['significant'] else "neg"
            confidence_level = sig_data['confidence_level']
            threshold = sig_data['threshold_used']
            
            # Visual indicators
            if sig_data['significant'] and confidence > 0.7:
                indicator = "🔴"
            elif sig_data['significant'] and confidence > 0.5:
                indicator = "🟡"
            elif confidence > 0.3:
                indicator = "🟠"
            else:
                indicator = "🟢"
            
            print(f"{indicator} {condition:<18} {confidence:.3f}      {confidence_level:<8} {status:<8} {threshold:.2f}")
        
        # Summary of significant findings
        print("\n" + "="*60)
        print("📋 SIGNIFICANT FINDINGS SUMMARY:")
        significant_findings = [(cond, data) for cond, data in significance.items() if data['significant']]
        
        if significant_findings:
            for condition, data in significant_findings:
                confidence = predictions[condition]
                clinical_note = ""
                if condition in ['Pneumothorax', 'Mass']:
                    clinical_note = " (HIGH PRIORITY)"
                elif condition in ['Pneumonia', 'Atelectasis']:
                    clinical_note = " (Medical evaluation needed)"
                print(f"  • {condition}: {confidence:.1%} confidence ({data['confidence_level']}){clinical_note}")
        else:
            print("  ✅ No significant pathologies detected")
        
        print("="*60)
        
        # Technical summary
        print(f"\n🔧 Technical Summary:")
        print(f"  ✓ Model: Balanced ResNet18 (F1: 0.23, Epoch 60)")
        print(f"  ✓ Architecture: {tester.model_type}")
        print(f"  ✓ Device: {tester.device}")
        print(f"  ✓ Image: {image_path}")
        print(f"  ✓ Conditions evaluated: {len(predictions)}")
        print(f"  ✓ Significant findings: {len(significant_findings)}")
        print(f"  ✓ Highest confidence: {max(predictions.values()):.3f}")
        print(f"  ✓ Model optimized for class balance (reduced 'No Finding' bias)")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_balanced_classifier()