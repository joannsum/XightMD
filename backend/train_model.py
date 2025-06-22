import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from utils.lung_classifier import LungClassifierTrainer, LungDiseaseClassifier, LABELS
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, hamming_loss, f1_score
import pandas as pd
from tqdm import tqdm
import json
import re

class NIHChestXrayDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, split='train', max_samples=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.split = split
        self.max_samples = max_samples
        
        # FIXED: Load data into memory to avoid streaming issues
        print(f"🔄 Loading {split} data into memory...")
        self.data = []
        count = 0
        for item in hf_dataset:
            self.data.append(item)
            count += 1
            if max_samples and count >= max_samples:
                break
        print(f"✅ Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            image = item['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
                    
            if self.transform:
                image = self.transform(image)
                
            # FIXED: Proper multi-label creation
            labels = torch.zeros(len(LABELS), dtype=torch.float32)
            label_list = item.get('label', [])
            
            if isinstance(label_list, list):
                for disease_name in label_list:
                    disease_name = disease_name.strip()
                    # FIXED: Proper indentation and condition checking
                    if disease_name in LABELS:
                        label_idx = LABELS.index(disease_name)
                        labels[label_idx] = 1.0
                    elif disease_name == 'Pleural_Thickening' and 'Pleural Thickening' in LABELS:
                        label_idx = LABELS.index('Pleural Thickening')
                        labels[label_idx] = 1.0
            
            return image, labels
            
        except Exception as e:
            print(f"⚠️  Error loading sample {idx}: {e}")
            dummy_image = torch.zeros(3, 224, 224)
            dummy_labels = torch.zeros(len(LABELS), dtype=torch.float32)
            return dummy_image, dummy_labels

def calculate_multilabel_metrics(targets, predictions, threshold=0.5):
    """Calculate proper multi-label metrics"""
    predictions_binary = (predictions > threshold).astype(int)
    
    metrics = {}
    
    # Overall metrics
    metrics['hamming_loss'] = hamming_loss(targets, predictions_binary)
    metrics['avg_precision_macro'] = average_precision_score(targets, predictions, average='macro')
    metrics['avg_precision_micro'] = average_precision_score(targets, predictions, average='micro')
    metrics['f1_macro'] = f1_score(targets, predictions_binary, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(targets, predictions_binary, average='micro', zero_division=0)
    
    # Per-condition metrics
    condition_metrics = {}
    for i, condition in enumerate(LABELS):
        if np.sum(targets[:, i]) > 0:  # Only calculate if we have positive samples
            try:
                condition_ap = average_precision_score(targets[:, i], predictions[:, i])
                condition_f1 = f1_score(targets[:, i], predictions_binary[:, i], zero_division=0)
                condition_metrics[condition] = {
                    'avg_precision': condition_ap,
                    'f1': condition_f1,
                    'positive_samples': int(np.sum(targets[:, i]))
                }
            except:
                condition_metrics[condition] = {
                    'avg_precision': 0.0,
                    'f1': 0.0,
                    'positive_samples': int(np.sum(targets[:, i]))
                }
    
    metrics['per_condition'] = condition_metrics
    return metrics

def validate_model(model, val_loader, criterion, device):
    """Proper validation with multi-label metrics"""
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics
    metrics = calculate_multilabel_metrics(all_targets, all_predictions)
    
    return val_loss / len(val_loader), metrics

def train_model(args):
    print("🏥 Loading NIH Chest X-ray 14 dataset...")
    try:
        dataset = load_dataset("BahaaEldin0/NIH-Chest-Xray-14", trust_remote_code=True)
        print("✅ Dataset loaded!")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # FIXED: Proper train/val split with adequate data
    trainer = LungClassifierTrainer()
    
    # Create datasets with proper sizes
    train_dataset = NIHChestXrayDataset(
        dataset['train'], 
        transform=trainer.transform, 
        max_samples=args.max_train_samples
    )
    
    # Use a portion for validation
    val_size = min(1000, len(train_dataset) // 10)
    train_size = len(train_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Initialize model and training components
    device = trainer.device
    model = trainer.model
    
    # FIXED: Use BCEWithLogitsLoss for better numerical stability
    criterion = nn.BCEWithLogitsLoss()
    
    # Update model to not use sigmoid in forward pass (BCEWithLogitsLoss handles it)
    model.backbone.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.backbone.classifier[1].in_features, len(LABELS))
        # Removed sigmoid - BCEWithLogitsLoss will handle it
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)
    
    # Training loop with proper validation
    best_f1 = 0.0
    patience_counter = 0
    training_history = []
    
    print(f"🚀 Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\n🔄 Epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Apply sigmoid for metrics calculation (since we removed it from model)
            outputs_sigmoid = torch.sigmoid(outputs)
            train_predictions.append(outputs_sigmoid.detach().cpu().numpy())
            train_targets.append(labels.cpu().numpy())
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate training metrics
        train_predictions = np.vstack(train_predictions)
        train_targets = np.vstack(train_targets)
        train_metrics = calculate_multilabel_metrics(train_targets, train_predictions)
        
        # Validation phase
        val_loss, val_metrics = validate_model(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['f1_macro'])
        
        # Print epoch results
        print(f"📈 Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"📈 Train F1 (macro): {train_metrics['f1_macro']:.4f}")
        print(f"📉 Val Loss: {val_loss:.4f}")
        print(f"📉 Val F1 (macro): {val_metrics['f1_macro']:.4f}")
        
        # Save best model
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            patience_counter = 0
            
            best_model_path = 'models/lung_classifier_FIXED_best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'training_history': training_history
            }, best_model_path)
            print(f"✅ Saved best model with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Save training history
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_f1_macro': train_metrics['f1_macro'],
            'val_loss': val_loss,
            'val_f1_macro': val_metrics['f1_macro'],
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        training_history.append(epoch_stats)
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation F1: {best_f1:.4f}")
    
    # Save final training history
    with open('models/training_history_FIXED.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    return best_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XightMD Lung Classifier - PROPERLY FIXED')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--max-train-samples', type=int, default=10000, help='Maximum training samples')
    parser.add_argument('--early-stopping-patience', type=int, default=5, help='Early stopping patience')
    
    args = parser.parse_args()
    train_model(args)