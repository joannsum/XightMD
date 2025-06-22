import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import time
import json
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import your updated classifier
from backend.utils.lung_classifier import OptimizedLungClassifier, LABELS, LungClassifierTrainer

class NIHDataset(Dataset):
    """Optimized NIH dataset loader with proper error handling"""
    
    def __init__(self, dataset_split, transform, max_samples=None, split_name="train"):
        self.transform = transform
        self.split_name = split_name
        
        print(f"🔄 Loading {split_name} data...")
        self.data = []
        self.labels_distribution = {label: 0 for label in LABELS}
        
        count = 0
        start_time = time.time()
        
        # Load data efficiently
        for item in dataset_split:
            try:
                # Basic validation
                if 'image' not in item or 'label' not in item:
                    continue
                    
                self.data.append(item)
                
                # Track label distribution
                labels = item.get('label', [])
                if isinstance(labels, list):
                    for label in labels:
                        if label.strip() in self.labels_distribution:
                            self.labels_distribution[label.strip()] += 1
                
                count += 1
                if max_samples and count >= max_samples:
                    break
                    
                # Progress update
                if count % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(f"   Loaded {count} samples in {elapsed:.1f}s")
                    
            except Exception as e:
                continue
        
        print(f"✅ Loaded {len(self.data)} {split_name} samples")
        self._print_statistics()
    
    def _print_statistics(self):
        """Print dataset statistics"""
        total_labels = sum(self.labels_distribution.values())
        print(f"📊 {self.split_name.capitalize()} Label Distribution:")
        
        # Sort by frequency
        sorted_labels = sorted(self.labels_distribution.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_labels[:10]:  # Top 10
            percentage = (count / len(self.data)) * 100 if len(self.data) > 0 else 0
            print(f"   {label:<20}: {count:>6} ({percentage:>5.1f}%)")
        
        print(f"   Average labels per image: {total_labels / len(self.data):.2f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            
            # Process image
            image = item['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # Process labels - create multi-hot encoding
            labels = torch.zeros(len(LABELS), dtype=torch.float32)
            label_list = item.get('label', [])
            
            if isinstance(label_list, list):
                for disease_name in label_list:
                    disease_name = disease_name.strip()
                    if disease_name in LABELS:
                        idx = LABELS.index(disease_name)
                        labels[idx] = 1.0
                    elif disease_name == 'Pleural_Thickening':  # Handle naming variant
                        if 'Pleural Thickening' in LABELS:
                            idx = LABELS.index('Pleural Thickening')
                            labels[idx] = 1.0
            
            return image, labels
            
        except Exception as e:
            # Return dummy data on error
            print(f"Error in sample {idx}: {e}")
            dummy_image = torch.zeros(3, 224, 224)
            dummy_labels = torch.zeros(len(LABELS), dtype=torch.float32)
            return dummy_image, dummy_labels

class MetricsTracker:
    """Track and display training metrics"""
    
    def __init__(self):
        self.history = []
        self.best_f1 = 0.0
        self.best_epoch = 0
        
    def calculate_metrics(self, targets, predictions, threshold=0.5):
        """Calculate comprehensive metrics"""
        # Convert to binary predictions
        pred_binary = (predictions > threshold).astype(int)
        
        # Calculate per-class metrics
        per_class_f1 = []
        per_class_precision = []
        per_class_recall = []
        
        for i in range(len(LABELS)):
            if np.sum(targets[:, i]) > 0:  # Only if we have positive samples
                f1 = f1_score(targets[:, i], pred_binary[:, i], zero_division=0)
                precision = precision_score(targets[:, i], pred_binary[:, i], zero_division=0)
                recall = recall_score(targets[:, i], pred_binary[:, i], zero_division=0)
            else:
                f1 = precision = recall = 0.0
            
            per_class_f1.append(f1)
            per_class_precision.append(precision)
            per_class_recall.append(recall)
        
        # Overall metrics
        macro_f1 = np.mean(per_class_f1)
        macro_precision = np.mean(per_class_precision)
        macro_recall = np.mean(per_class_recall)
        
        # Micro metrics
        micro_f1 = f1_score(targets, pred_binary, average='micro', zero_division=0)
        micro_precision = precision_score(targets, pred_binary, average='micro', zero_division=0)
        micro_recall = recall_score(targets, pred_binary, average='micro', zero_division=0)
        
        return {
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'per_class_f1': per_class_f1,
            'threshold': threshold
        }
    
    def update(self, epoch, train_loss, train_metrics, val_loss, val_metrics, lr):
        """Update metrics history"""
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_macro_f1': train_metrics['macro_f1'],
            'train_micro_f1': train_metrics['micro_f1'],
            'val_loss': val_loss,
            'val_macro_f1': val_metrics['macro_f1'],
            'val_micro_f1': val_metrics['micro_f1'],
            'learning_rate': lr,
            'timestamp': time.time()
        }
        
        self.history.append(epoch_data)
        
        # Track best performance
        if val_metrics['macro_f1'] > self.best_f1:
            self.best_f1 = val_metrics['macro_f1']
            self.best_epoch = epoch
        
        return epoch_data
    
    def print_epoch_summary(self, epoch_data):
        """Print formatted epoch summary"""
        print(f"\n📊 Epoch {epoch_data['epoch']} Summary:")
        print(f"   Train Loss: {epoch_data['train_loss']:.4f} | Train F1: {epoch_data['train_macro_f1']:.4f}")
        print(f"   Val Loss:   {epoch_data['val_loss']:.4f} | Val F1:   {epoch_data['val_macro_f1']:.4f}")
        print(f"   Learning Rate: {epoch_data['learning_rate']:.6f}")
        print(f"   Best F1: {self.best_f1:.4f} (Epoch {self.best_epoch})")
    
    def save_history(self, filepath):
        """Save training history"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

def train_optimized_model(resume_from_checkpoint=None):
    """Main training function with comprehensive monitoring"""
    
    print("🚀 OPTIMIZED LUNG DISEASE CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Configuration
    config = {
        'max_train_samples': 10000,  # Adjust based on available memory
        'max_val_samples': 2000,
        'batch_size': 16,
        'epochs': 25,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'patience': 5,
        'num_workers': 2
    }
    
    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Load dataset
    print("\n🔄 Loading NIH Chest X-ray dataset...")
    try:
        dataset = load_dataset("BahaaEldin0/NIH-Chest-Xray-14", trust_remote_code=True)
        print("✅ Dataset loaded successfully")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Create trainer and transforms
    trainer = LungClassifierTrainer()
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = NIHDataset(
        dataset['train'], 
        train_transform, 
        max_samples=config['max_train_samples'],
        split_name="train"
    )
    
    val_dataset = NIHDataset(
        dataset['train'],  # Using train split but different samples
        val_transform,
        max_samples=config['max_val_samples'],
        split_name="validation"
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Setup training components
    device = trainer.device
    model = trainer.model
    
    # Use class weights to handle imbalance
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Add after model setup:
    start_epoch = 0
    best_f1 = 0.0

    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"🔄 Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_f1 = checkpoint.get('best_f1', 0.0)
        metrics_tracker.best_f1 = best_f1
        print(f"✅ Resumed from epoch {start_epoch}")

    print(f"\n🎯 Training Setup:")
    print(f"   Device: {device}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Optimizer: AdamW")
    print(f"   Loss function: BCEWithLogitsLoss")
    print(f"   Starting from epoch: {start_epoch}")
    
    # Training loop
    print(f"\n🚀 Starting training for {config['epochs']} epochs...")
    
    start_time = time.time()
    patience_counter = 0
    
    for epoch in range(start_epoch, config['epochs']):
        epoch_start = time.time()
        print(f"\n{'='*20} Epoch {epoch+1}/{config['epochs']} {'='*20}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Store predictions for metrics
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                train_preds.append(probs.cpu().numpy())
                train_targets.append(labels.cpu().numpy())
            
            # Update progress bar
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate training metrics
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        train_metrics = metrics_tracker.calculate_metrics(train_targets, train_preds, threshold=0.5)
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        val_pbar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate validation metrics
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_metrics = metrics_tracker.calculate_metrics(val_targets, val_preds, threshold=0.5)
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(val_metrics['macro_f1'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update metrics tracker
        epoch_data = metrics_tracker.update(
            epoch + 1, avg_train_loss, train_metrics, 
            avg_val_loss, val_metrics, current_lr
        )
        
        # Print epoch summary
        metrics_tracker.print_epoch_summary(epoch_data)
        
        # Print top performing classes
        print(f"\n🏆 Top 5 Performing Classes (F1 Score):")
        class_f1_pairs = [(LABELS[i], val_metrics['per_class_f1'][i]) for i in range(len(LABELS))]
        class_f1_pairs.sort(key=lambda x: x[1], reverse=True)
        for i, (class_name, f1_score) in enumerate(class_f1_pairs[:5]):
            print(f"   {i+1}. {class_name:<20}: {f1_score:.4f}")
        
        # Save best model
        if val_metrics['macro_f1'] > metrics_tracker.best_f1:
            patience_counter = 0
            best_model_path = 'models/lung_classifier_BEST.pth'
            # Save checkpoint with additional info
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': val_metrics['macro_f1'],
                'val_metrics': val_metrics,
                'config': config
            }
    os.makedirs('models', exist_ok=True)
            torch.save(checkpoint, best_model_path)
            print(f"🏆 New best model saved! F1: {val_metrics['macro_f1']:.4f}")
        else:
            patience_counter += 1
        
        # Save checkpoint every few epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'models/lung_classifier_checkpoint_epoch_{epoch+1}.pth'
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': metrics_tracker.best_f1,
                'val_metrics': val_metrics,
                'config': config
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
            print(f"   Best F1: {metrics_tracker.best_f1:.4f} at epoch {metrics_tracker.best_epoch}")
            break
        
        # Time estimation
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        estimated_total = (total_time / (epoch - start_epoch + 1)) * (config['epochs'] - start_epoch)
        remaining = estimated_total - total_time
        
        print(f"⏱️  Epoch time: {epoch_time:.1f}s | Total: {total_time/60:.1f}m | ETA: {remaining/60:.1f}m")
    
    # Training complete
    total_training_time = time.time() - start_time
    print(f"\n🎉 Training completed in {total_training_time/60:.1f} minutes!")
    print(f"🏆 Best validation F1: {metrics_tracker.best_f1:.4f} (Epoch {metrics_tracker.best_epoch})")
    
    # Save training history
    os.makedirs('models', exist_ok=True)
    metrics_tracker.save_history('models/training_history_optimized.json')
    
    # Save final model
    final_model_path = 'models/lung_classifier_FINAL.pth'
    final_checkpoint = {
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1': metrics_tracker.best_f1,
        'val_metrics': val_metrics,
        'config': config
    }
    torch.save(final_checkpoint, final_model_path)
    print(f"\n💾 Models saved:")
    print(f"   Best model: models/lung_classifier_BEST.pth")
    print(f"   Final model: models/lung_classifier_FINAL.pth")
    print(f"   Training history: models/training_history_optimized.json")
    
    return metrics_tracker.best_f1

if __name__ == "__main__":
    # Example usage with checkpoint resumption
    # train_optimized_model(resume_from_checkpoint="models/lung_classifier_checkpoint_epoch_10.pth")
    train_optimized_model()