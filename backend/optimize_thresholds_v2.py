import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from tqdm import tqdm
import json
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax",
    "No Finding"
]

class SimpleLungModel(nn.Module):
    """Balanced model architecture"""
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
    """Optimized model architecture"""
    def __init__(self, num_classes: int = 15):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
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

class ValidationDataset(Dataset):
    """Dataset for threshold optimization"""
    
    def __init__(self, dataset_split, max_samples=2000):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"🔄 Loading validation data (max {max_samples} samples)...")
        self.data = []
        count = 0
        
        for item in dataset_split:
            try:
                if 'image' not in item or 'label' not in item:
                    continue
                    
                self.data.append(item)
                count += 1
                
                if count >= max_samples:
                    break
                    
                if count % 500 == 0:
                    print(f"   Loaded {count} samples...")
                    
            except Exception:
                continue
        
        print(f"✅ Loaded {len(self.data)} validation samples")
        self._print_label_distribution()
    
    def _print_label_distribution(self):
        """Print distribution of labels in validation set"""
        label_counts = defaultdict(int)
        total_samples = len(self.data)
        
        for item in self.data:
            labels = item.get('label', [])
            if isinstance(labels, list):
                for label in labels:
                    if label.strip() in LABELS:
                        label_counts[label.strip()] += 1
        
        print(f"\n📊 Validation Label Distribution:")
        for label in LABELS:
            count = label_counts[label]
            percentage = (count / total_samples) * 100
            print(f"   {label:<20}: {count:>4} ({percentage:>5.1f}%)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            
            image = item['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = self.transform(image)
            
            # Create multi-hot labels
            labels = torch.zeros(len(LABELS), dtype=torch.float32)
            label_list = item.get('label', [])
            
            if isinstance(label_list, list):
                for disease_name in label_list:
                    disease_name = disease_name.strip()
                    if disease_name in LABELS:
                        idx = LABELS.index(disease_name)
                        labels[idx] = 1.0
                    elif disease_name == 'Pleural_Thickening':
                        idx = LABELS.index('Pleural Thickening')
                        labels[idx] = 1.0
            
            return image, labels
            
        except Exception:
            dummy_image = torch.zeros(3, 224, 224)
            dummy_labels = torch.zeros(len(LABELS), dtype=torch.float32)
            return dummy_image, dummy_labels

class ThresholdOptimizer:
    """Advanced threshold optimization with multiple metrics"""
    
    def __init__(self, model_path: str, validation_samples: int = 2000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.validation_samples = validation_samples
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize results storage
        self.optimal_thresholds = {}
        self.metrics_per_threshold = {}
        self.best_metrics = {}
        
    def _load_model(self, model_path: str):
        """Load model with automatic architecture detection"""
        print(f"🔄 Loading model from {model_path}...")
        
        if 'balanced' in model_path.lower():
            model = SimpleLungModel()
            architecture = "Balanced ResNet18"
        else:
            model = OptimizedLungClassifier()
            architecture = "Optimized EfficientNet-B0"
        
        model.to(self.device)
        
        try:
            if 'balanced' in model_path.lower():
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
            else:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            
            print(f"✅ Loaded {architecture}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_validation_data(self):
        """Load validation dataset"""
        print("📥 Loading NIH Chest X-ray dataset...")
        dataset = load_dataset("BahaaEldin0/NIH-Chest-Xray-14", trust_remote_code=True)
        
        val_dataset = ValidationDataset(dataset['train'], max_samples=self.validation_samples)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        return val_loader
    
    def get_model_predictions(self, val_loader):
        """Get model predictions on validation set"""
        print("🔄 Getting model predictions...")
        
        all_predictions = []
        all_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Predicting"):
                images = images.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.sigmoid(outputs)
                
                all_predictions.append(probabilities.cpu().numpy())
                all_targets.append(labels.numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        print(f"✅ Got predictions for {len(predictions)} samples")
        return predictions, targets
    
    def calculate_metrics_for_threshold(self, y_true, y_pred, threshold):
        """Calculate comprehensive metrics for a given threshold"""
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Handle case where no predictions are made
        if np.sum(y_pred_binary) == 0:
            return {
                'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                'specificity': 1.0, 'balanced_accuracy': 0.5
            }
        
        # Handle case where no positive samples exist
        if np.sum(y_true) == 0:
            return {
                'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                'specificity': 1.0, 'balanced_accuracy': 0.5
            }
        
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        
        # Calculate specificity
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        
        # Balanced accuracy
        balanced_accuracy = (recall + specificity) / 2
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'balanced_accuracy': balanced_accuracy
        }
    
    def optimize_threshold_single_condition(self, condition_idx, condition_name, predictions, targets, metric='f1'):
        """Optimize threshold for a single condition"""
        print(f"\n🎯 Optimizing {condition_name}...")
        
        y_true = targets[:, condition_idx]
        y_pred = predictions[:, condition_idx]
        
        # Check if we have positive samples
        positive_samples = int(np.sum(y_true))
        total_samples = len(y_true)
        
        print(f"   Positive samples: {positive_samples}/{total_samples} ({positive_samples/total_samples*100:.1f}%)")
        
        if positive_samples == 0:
            print(f"   ⚠️  No positive samples for {condition_name}")
            return 0.5, {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Test thresholds from 0.01 to 0.99
        thresholds = np.arange(0.01, 1.0, 0.01)
        best_threshold = 0.5
        best_score = 0.0
        threshold_metrics = {}
        
        for threshold in thresholds:
            metrics = self.calculate_metrics_for_threshold(y_true, y_pred, threshold)
            threshold_metrics[threshold] = metrics
            
            score = metrics[metric]
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        best_metrics = threshold_metrics[best_threshold]
        
        print(f"   ✅ Best threshold: {best_threshold:.3f}")
        print(f"   📊 {metric.upper()}: {best_score:.3f}")
        print(f"   📊 Precision: {best_metrics['precision']:.3f}")
        print(f"   📊 Recall: {best_metrics['recall']:.3f}")
        print(f"   📊 Specificity: {best_metrics['specificity']:.3f}")
        
        return best_threshold, best_metrics, threshold_metrics
    
    def optimize_all_thresholds(self, metric='f1'):
        """Optimize thresholds for all conditions"""
        print(f"🚀 THRESHOLD OPTIMIZATION")
        print(f"Metric: {metric.upper()}")
        print("="*60)
        
        # Load validation data
        val_loader = self.load_validation_data()
        
        # Get predictions
        predictions, targets = self.get_model_predictions(val_loader)
        
        # Optimize each condition
        results = {}
        all_threshold_data = {}
        
        for i, condition in enumerate(LABELS):
            threshold, metrics, threshold_curves = self.optimize_threshold_single_condition(
                i, condition, predictions, targets, metric
            )
            
            results[condition] = {
                'optimal_threshold': threshold,
                'metrics': metrics
            }
            all_threshold_data[condition] = threshold_curves
        
        # Store results
        self.optimal_thresholds = {k: v['optimal_threshold'] for k, v in results.items()}
        self.best_metrics = {k: v['metrics'] for k, v in results.items()}
        self.metrics_per_threshold = all_threshold_data
        
        return results
    
    def evaluate_with_optimal_thresholds(self, predictions, targets):
        """Evaluate model performance using optimal thresholds"""
        print(f"\n📊 EVALUATING WITH OPTIMAL THRESHOLDS")
        print("="*60)
        
        # Calculate per-class performance
        per_class_results = {}
        overall_predictions = []
        overall_targets = []
        
        for i, condition in enumerate(LABELS):
            threshold = self.optimal_thresholds[condition]
            y_true = targets[:, i]
            y_pred = predictions[:, i]
            y_pred_binary = (y_pred > threshold).astype(int)
            
            metrics = self.calculate_metrics_for_threshold(y_true, y_pred, threshold)
            per_class_results[condition] = {
                'threshold': threshold,
                'positive_samples': int(np.sum(y_true)),
                **metrics
            }
            
            overall_predictions.append(y_pred_binary)
            overall_targets.append(y_true)
        
        # Calculate overall metrics
        overall_pred = np.column_stack(overall_predictions)
        overall_true = np.column_stack(overall_targets)
        
        macro_f1 = f1_score(overall_true, overall_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(overall_true, overall_pred, average='micro', zero_division=0)
        
        print(f"🎯 Overall Performance:")
        print(f"   Macro F1: {macro_f1:.4f}")
        print(f"   Micro F1: {micro_f1:.4f}")
        
        return per_class_results, {'macro_f1': macro_f1, 'micro_f1': micro_f1}
    
    def save_results(self, output_dir='models/threshold_optimization'):
        """Save optimization results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save optimal thresholds
        with open(f"{output_dir}/optimal_thresholds.json", 'w') as f:
            json.dump(self.optimal_thresholds, f, indent=2)
        
        # Save detailed metrics
        with open(f"{output_dir}/detailed_metrics.json", 'w') as f:
            json.dump(self.best_metrics, f, indent=2)
        
        # Create CSV summary
        df_data = []
        for condition in LABELS:
            threshold = self.optimal_thresholds[condition]
            metrics = self.best_metrics[condition]
            
            df_data.append({
                'Condition': condition,
                'Optimal_Threshold': threshold,
                'F1_Score': metrics['f1'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Specificity': metrics['specificity'],
                'Balanced_Accuracy': metrics['balanced_accuracy']
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(f"{output_dir}/threshold_optimization_results.csv", index=False)
        
        print(f"\n💾 Results saved to {output_dir}/")
        print(f"   - optimal_thresholds.json")
        print(f"   - detailed_metrics.json") 
        print(f"   - threshold_optimization_results.csv")
    
    def generate_code_update(self, output_dir='models/threshold_optimization'):
        """Generate code with optimal thresholds"""
        code_template = '''# Optimized thresholds based on validation data
# Copy this dictionary to your lung_classifier.py file

self.condition_thresholds = {
'''
        
        for condition, threshold in self.optimal_thresholds.items():
            code_template += f"    '{condition}': {threshold:.3f},\n"
        
        code_template += '''}

# Performance Summary:
'''
        
        for condition in LABELS:
            metrics = self.best_metrics[condition]
            code_template += f"# {condition:<20}: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}\n"
        
        code_template += '''
# Usage Instructions:
# 1. Copy the condition_thresholds dictionary above
# 2. Replace the existing thresholds in your LungClassifierTrainer.__init__() method
# 3. These thresholds are optimized for maximum F1 score performance
'''
        
        with open(f"{output_dir}/updated_thresholds.py", 'w') as f:
            f.write(code_template)
        
        print(f"   - updated_thresholds.py")
    
    def print_summary(self):
        """Print optimization summary"""
        print(f"\n🏆 OPTIMIZATION SUMMARY")
        print("="*60)
        
        # Sort by F1 score
        sorted_conditions = sorted(
            LABELS, 
            key=lambda x: self.best_metrics[x]['f1'], 
            reverse=True
        )
        
        print(f"{'Condition':<20} {'Threshold':<10} {'F1':<6} {'Prec':<6} {'Recall':<6}")
        print("-" * 60)
        
        for condition in sorted_conditions:
            threshold = self.optimal_thresholds[condition]
            metrics = self.best_metrics[condition]
            
            print(f"{condition:<20} {threshold:<10.3f} {metrics['f1']:<6.3f} "
                  f"{metrics['precision']:<6.3f} {metrics['recall']:<6.3f}")
        
        # Overall statistics
        avg_f1 = np.mean([metrics['f1'] for metrics in self.best_metrics.values()])
        avg_threshold = np.mean(list(self.optimal_thresholds.values()))
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Average F1: {avg_f1:.3f}")
        print(f"   Average Threshold: {avg_threshold:.3f}")
        print(f"   Conditions with F1 > 0.3: {sum(1 for m in self.best_metrics.values() if m['f1'] > 0.3)}/15")

def main():
    parser = argparse.ArgumentParser(description='Optimize classification thresholds')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--samples', type=int, default=2000, help='Number of validation samples')
    parser.add_argument('--metric', type=str, default='f1', choices=['f1', 'precision', 'recall', 'balanced_accuracy'], 
                       help='Metric to optimize')
    parser.add_argument('--output-dir', type=str, default='models/threshold_optimization', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    # Initialize optimizer
    optimizer = ThresholdOptimizer(args.model, args.samples)
    
    # Run optimization
    results = optimizer.optimize_all_thresholds(metric=args.metric)
    
    # Print summary
    optimizer.print_summary()
    
    # Save results
    optimizer.save_results(args.output_dir)
    optimizer.generate_code_update(args.output_dir)
    
    print(f"\n✅ Threshold optimization complete!")
    print(f"📁 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()