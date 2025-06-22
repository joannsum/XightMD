#!/usr/bin/env python3
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve, auc, accuracy_score
from utils.lung_classifier import LungClassifierTrainer, LABELS
from datasets import load_dataset
import json
import os
from tqdm import tqdm
import argparse

class ThresholdOptimizer:
    def __init__(self, model_path=None):
        self.classifier = LungClassifierTrainer(model_path=model_path)
        self.optimal_thresholds = {}
        self.metrics_history = {}
        
    def load_validation_data(self, num_samples=500):
        """Load validation dataset for threshold optimization"""
        print(f"📥 Loading {num_samples} validation samples...")
        
        try:
            # Load dataset
            dataset = load_dataset("BahaaEldin0/NIH-Chest-Xray-14", streaming=True, trust_remote_code=True)
            val_iter = iter(dataset['train'])  # Using train as validation for now
            
            images = []
            true_labels = []
            predictions = []
            
            processed = 0
            progress_bar = tqdm(total=num_samples, desc="Processing samples")
            
            while processed < num_samples:
                try:
                    item = next(val_iter)
                    
                    # Create true label vector
                    true_label_vector = np.zeros(len(LABELS), dtype=np.float32)
                    label_list = item.get('label', [])
                    
                    for disease_name in label_list:
                        disease_name = disease_name.strip()
                        if disease_name in LABELS:
                            label_idx = LABELS.index(disease_name)
                            true_label_vector[label_idx] = 1.0
                    
                    # Save image temporarily and get predictions
                    image = item['image']
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    temp_path = f"temp_val_{processed}.jpg"
                    image.save(temp_path)
                    
                    # Get model predictions
                    pred_dict = self.classifier.predict(temp_path)
                    pred_vector = np.array([pred_dict[label] for label in LABELS])
                    
                    # Store results
                    true_labels.append(true_label_vector)
                    predictions.append(pred_vector)
                    
                    # Cleanup
                    os.remove(temp_path)
                    
                    processed += 1
                    progress_bar.update(1)
                    
                except StopIteration:
                    val_iter = iter(dataset['train'])
                except Exception as e:
                    print(f"⚠️  Error processing sample {processed}: {e}")
                    continue
            
            progress_bar.close()
            
            self.true_labels = np.array(true_labels)
            self.predictions = np.array(predictions)
            
            print(f"✅ Loaded {processed} samples")
            print(f"   True labels shape: {self.true_labels.shape}")
            print(f"   Predictions shape: {self.predictions.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading validation data: {e}")
            return False
    
    def find_optimal_threshold_single_condition(self, condition_idx, condition_name, metric='f1'):
        """Find optimal threshold for a single condition"""
        
        y_true = self.true_labels[:, condition_idx]
        y_scores = self.predictions[:, condition_idx]
        
        # Skip if no positive samples
        if np.sum(y_true) == 0:
            print(f"⚠️  No positive samples for {condition_name}, using default threshold 0.5")
            return 0.5, {}
        
        # Test thresholds from 0.01 to 0.99
        thresholds = np.arange(0.01, 0.99, 0.01)
        metrics = {
            'thresholds': thresholds,
            'f1_scores': [],
            'precision': [],
            'recall': [],
            'accuracy': [],
            'specificity': []
        }
        
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            # Calculate metrics
            if len(np.unique(y_pred)) > 1:  # Avoid division by zero
                f1 = f1_score(y_true, y_pred, zero_division=0)
                accuracy = accuracy_score(y_true, y_pred)
                
                # Manual precision/recall calculation
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                f1 = accuracy = precision = recall = specificity = 0
            
            metrics['f1_scores'].append(f1)
            metrics['accuracy'].append(accuracy)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['specificity'].append(specificity)
            
            # Update best threshold based on chosen metric
            if metric == 'f1' and f1 > best_score:
                best_score = f1
                best_threshold = threshold
            elif metric == 'accuracy' and accuracy > best_score:
                best_score = accuracy
                best_threshold = threshold
        
        return best_threshold, metrics
    
    def optimize_all_thresholds(self, metric='f1'):
        """Find optimal thresholds for all conditions"""
        
        print(f"\n🎯 Optimizing thresholds using {metric.upper()} metric...")
        print("="*60)
        
        self.optimal_thresholds = {}
        self.metrics_history = {}
        
        for i, condition in enumerate(LABELS):
            print(f"\n🔍 Optimizing {condition}...")
            
            threshold, metrics = self.find_optimal_threshold_single_condition(i, condition, metric)
            
            self.optimal_thresholds[condition] = threshold
            self.metrics_history[condition] = metrics
            
            # Get current performance at optimal threshold
            y_true = self.true_labels[:, i]
            y_scores = self.predictions[:, i]
            y_pred = (y_scores >= threshold).astype(int)
            
            if np.sum(y_true) > 0:
                final_f1 = f1_score(y_true, y_pred, zero_division=0)
                final_accuracy = accuracy_score(y_true, y_pred)
                
                print(f"   ✅ Optimal threshold: {threshold:.3f}")
                print(f"   📊 F1 Score: {final_f1:.3f}")
                print(f"   📊 Accuracy: {final_accuracy:.3f}")
                print(f"   📊 Positive samples: {int(np.sum(y_true))}/{len(y_true)}")
            else:
                print(f"   ⚠️  No positive samples, using default: {threshold:.3f}")
        
        # Save results
        self.save_results()
        
        # Generate text-based analysis instead of plots
        self.create_text_analysis()
        
        return self.optimal_thresholds
    
    def save_results(self):
        """Save optimization results"""
        os.makedirs('models/threshold_optimization', exist_ok=True)
        
        # Save optimal thresholds
        with open('models/threshold_optimization/optimal_thresholds.json', 'w') as f:
            json.dump(self.optimal_thresholds, f, indent=2)
        
        # Save detailed metrics
        with open('models/threshold_optimization/optimization_metrics.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {}
            for condition, metrics in self.metrics_history.items():
                serializable_metrics[condition] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in metrics.items()
                }
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"\n💾 Results saved to models/threshold_optimization/")
    
    def create_text_analysis(self):
        """Create text-based analysis instead of plots"""
        print("\n📊 Creating text-based analysis...")
        
        os.makedirs('models/threshold_optimization', exist_ok=True)
        
        analysis_text = "THRESHOLD OPTIMIZATION ANALYSIS\n"
        analysis_text += "=" * 60 + "\n\n"
        
        for condition in LABELS:
            if condition not in self.metrics_history:
                continue
                
            metrics = self.metrics_history[condition]
            if not metrics['f1_scores']:
                continue
            
            analysis_text += f"CONDITION: {condition}\n"
            analysis_text += "-" * 40 + "\n"
            
            optimal_threshold = self.optimal_thresholds[condition]
            optimal_idx = np.argmin(np.abs(np.array(metrics['thresholds']) - optimal_threshold))
            
            analysis_text += f"Optimal Threshold: {optimal_threshold:.3f}\n"
            analysis_text += f"F1 Score at Optimal: {metrics['f1_scores'][optimal_idx]:.3f}\n"
            analysis_text += f"Precision at Optimal: {metrics['precision'][optimal_idx]:.3f}\n"
            analysis_text += f"Recall at Optimal: {metrics['recall'][optimal_idx]:.3f}\n"
            analysis_text += f"Accuracy at Optimal: {metrics['accuracy'][optimal_idx]:.3f}\n"
            analysis_text += f"Specificity at Optimal: {metrics['specificity'][optimal_idx]:.3f}\n"
            
            # Find best and worst performing thresholds
            best_f1_idx = np.argmax(metrics['f1_scores'])
            worst_f1_idx = np.argmin(metrics['f1_scores'])
            
            analysis_text += f"\nBest F1 Performance:\n"
            analysis_text += f"  Threshold: {metrics['thresholds'][best_f1_idx]:.3f}\n"
            analysis_text += f"  F1 Score: {metrics['f1_scores'][best_f1_idx]:.3f}\n"
            
            # Threshold range analysis
            good_thresholds = []
            for i, f1 in enumerate(metrics['f1_scores']):
                if f1 >= metrics['f1_scores'][optimal_idx] * 0.95:  # Within 95% of optimal
                    good_thresholds.append(metrics['thresholds'][i])
            
            if good_thresholds:
                analysis_text += f"\nGood Threshold Range (≥95% of optimal F1):\n"
                analysis_text += f"  Range: {min(good_thresholds):.3f} - {max(good_thresholds):.3f}\n"
                analysis_text += f"  Count: {len(good_thresholds)} thresholds\n"
            
            analysis_text += "\n" + "=" * 60 + "\n\n"
        
        # Save analysis
        with open('models/threshold_optimization/analysis.txt', 'w') as f:
            f.write(analysis_text)
        
        print(f"✅ Text analysis saved to models/threshold_optimization/analysis.txt")
    
    def create_csv_summary(self):
        """Create CSV summary of results"""
        
        csv_content = "Condition,Optimal_Threshold,F1_Score,Precision,Recall,Accuracy,Specificity\n"
        
        for condition in LABELS:
            if condition not in self.optimal_thresholds:
                continue
                
            threshold = self.optimal_thresholds[condition]
            
            if condition in self.metrics_history and self.metrics_history[condition]['f1_scores']:
                metrics = self.metrics_history[condition]
                optimal_idx = np.argmin(np.abs(np.array(metrics['thresholds']) - threshold))
                
                f1 = metrics['f1_scores'][optimal_idx]
                precision = metrics['precision'][optimal_idx]
                recall = metrics['recall'][optimal_idx]
                accuracy = metrics['accuracy'][optimal_idx]
                specificity = metrics['specificity'][optimal_idx]
                
                csv_content += f"{condition},{threshold:.3f},{f1:.3f},{precision:.3f},{recall:.3f},{accuracy:.3f},{specificity:.3f}\n"
            else:
                csv_content += f"{condition},{threshold:.3f},0.000,0.000,0.000,0.000,0.000\n"
        
        with open('models/threshold_optimization/summary.csv', 'w') as f:
            f.write(csv_content)
        
        print(f"✅ CSV summary saved to models/threshold_optimization/summary.csv")
    
    def generate_updated_code(self):
        """Generate updated code with optimal thresholds"""
        
        print("\n🔧 Generating updated classifier code...")
        
        code_template = '''# Updated thresholds based on optimization results:
# Copy this dictionary to your lung_classifier.py file!

self.condition_thresholds = {
'''
        
        for condition, threshold in self.optimal_thresholds.items():
            code_template += f"    '{condition}': {threshold:.3f},\n"
        
        code_template += '''}

# Usage Instructions:
# 1. Copy the dictionary above
# 2. Replace the condition_thresholds in your LungClassifierTrainer.__init__() method
# 3. These thresholds are optimized for maximum F1 score performance
'''
        
        with open('models/threshold_optimization/updated_thresholds.py', 'w') as f:
            f.write(code_template)
        
        print("✅ Updated code saved to models/threshold_optimization/updated_thresholds.py")
        
        # Create CSV summary
        self.create_csv_summary()
        
        print("\n📋 Summary of optimal thresholds:")
        print("="*50)
        
        for condition, threshold in sorted(self.optimal_thresholds.items()):
            print(f"{condition:<20}: {threshold:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Optimize classification thresholds')
    parser.add_argument('--model', type=str, help='Model path to optimize')
    parser.add_argument('--samples', type=int, default=500, help='Number of validation samples')
    parser.add_argument('--metric', type=str, default='f1', choices=['f1', 'accuracy'], help='Optimization metric')
    parser.add_argument('--atelectasis', action='store_true', help='Use Atelectasis model')
    
    args = parser.parse_args()
    
    # Determine model path
    if args.atelectasis:
        model_path = "models/checkpoints/Atelectasis/best_model_Atelectasis.pth"
    elif args.model:
        model_path = args.model
    else:
        model_path = None  # Use pretrained
    
    print(f"🎯 THRESHOLD OPTIMIZATION")
    print(f"Model: {model_path or 'pretrained'}")
    print(f"Samples: {args.samples}")
    print(f"Metric: {args.metric.upper()}")
    print("="*60)
    
    # Initialize optimizer
    optimizer = ThresholdOptimizer(model_path=model_path)
    
    # Load validation data
    if not optimizer.load_validation_data(num_samples=args.samples):
        print("❌ Failed to load validation data")
        return
    
    # Optimize thresholds
    optimal_thresholds = optimizer.optimize_all_thresholds(metric=args.metric)
    
    # Generate updated code
    optimizer.generate_updated_code()
    
    print(f"\n🎉 Optimization completed!")
    print(f"📁 Results saved to models/threshold_optimization/")
    print(f"📄 Files created:")
    print(f"   - optimal_thresholds.json (machine readable)")
    print(f"   - updated_thresholds.py (code to copy)")
    print(f"   - analysis.txt (detailed text analysis)")
    print(f"   - summary.csv (spreadsheet format)")

if __name__ == "__main__":
    main()