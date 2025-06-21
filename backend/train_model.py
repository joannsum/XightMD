import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from utils.lung_classifier import LungClassifierTrainer, LungDiseaseClassifier, LABELS
import argparse
import os
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from tqdm import tqdm
import json

class NIHChestXrayDataset(Dataset):
    """Custom dataset for NIH Chest X-ray 14 with streaming"""
    
    def __init__(self, hf_dataset, transform=None, split='train', max_samples=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.split = split
        self.max_samples = max_samples
        
        # Create label mapping from NIH to our LABELS
        self.label_mapping = {
            'Atelectasis': 'Atelectasis',
            'Cardiomegaly': 'Cardiomegaly', 
            'Consolidation': 'Consolidation',
            'Edema': 'Edema',
            'Effusion': 'Effusion',
            'Emphysema': 'Emphysema',
            'Fibrosis': 'Fibrosis',
            'Hernia': 'Hernia',
            'Infiltration': 'Infiltration',
            'Mass': 'Mass',
            'Nodule': 'Nodule',
            'Pleural_Thickening': 'Pleural Thickening',
            'Pneumonia': 'Pneumonia',
            'Pneumothorax': 'Pneumothorax'
        }
        
        # Limit dataset size if specified
        if self.max_samples:
            print(f"🔄 Using only {self.max_samples} samples for {split}")
        
    def __len__(self):
        if self.max_samples:
            return min(self.max_samples, len(self.dataset))
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Stream individual samples on-demand
        try:
            item = self.dataset[idx]
            
            # Get image - this streams from HuggingFace
            image = item['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Create multi-label target
            labels = torch.zeros(len(LABELS), dtype=torch.float32)
            
            # Map NIH labels to our label indices
            for nih_label, our_label in self.label_mapping.items():
                if item.get(nih_label, 0) == 1:  # If condition is present
                    if our_label in LABELS:
                        label_idx = LABELS.index(our_label)
                        labels[label_idx] = 1.0
            
            return image, labels
            
        except Exception as e:
            print(f"⚠️  Error loading sample {idx}: {e}")
            # Return a dummy sample if loading fails
            dummy_image = torch.zeros(3, 224, 224)
            dummy_labels = torch.zeros(len(LABELS), dtype=torch.float32)
            return dummy_image, dummy_labels

def train_model(args):
    """Train the lung disease classifier on NIH Chest X-ray dataset"""
    
    print("🏥 Loading NIH Chest X-ray 14 dataset (streaming mode)...")
    try:
        # Load dataset in streaming mode - NO LOCAL DOWNLOAD!
        dataset = load_dataset(
            "BahaaEldin0/NIH-Chest-Xray-14", 
            streaming=True,  # This prevents local download!
            trust_remote_code=True
        )
        
        print(f"✅ Dataset loaded in streaming mode!")
        print(f"🌐 Data will be streamed from HuggingFace servers")
        
        # Convert to iterable datasets for streaming
        train_dataset_stream = dataset['train']
        
        # For validation, we'll use a subset of train since streaming
        # doesn't easily allow for train/val splits
        print(f"📊 Using streaming mode with limited samples per epoch")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 Make sure you have internet connection and HuggingFace access")
        return
    
    # Initialize trainer
    trainer = LungClassifierTrainer()
    
    # Setup training parameters for streaming
    samples_per_epoch = args.samples_per_epoch
    val_samples = args.val_samples
    
    print(f"🔄 Training with {samples_per_epoch} samples per epoch")
    print(f"🔄 Validation with {val_samples} samples")
    
    # Setup training
    device = trainer.device
    model = trainer.model
    
    # Loss function for multi-label classification
    criterion = nn.BCELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    print(f"🚀 Starting training for {args.epochs} epochs...")
    print(f"🔧 Device: {device}")
    print(f"🔧 Batch size: {args.batch_size}")
    print(f"🔧 Learning rate: {args.learning_rate}")
    print(f"🌐 Streaming mode: No local storage used!")
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(args.epochs):
        print(f"\n🔄 Epoch {epoch+1}/{args.epochs}")
        
        # Training phase with streaming
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        # Create iterator for this epoch
        train_iter = iter(train_dataset_stream)
        processed_samples = 0
        batch_images = []
        batch_labels = []
        
        progress_bar = tqdm(total=samples_per_epoch, desc=f"Training Epoch {epoch+1}")
        
        while processed_samples < samples_per_epoch:
            try:
                # Get next sample from stream
                item = next(train_iter)
                
                # Process image
                image = item['image']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image_tensor = trainer.transform(image)
                
                # Create labels
                labels = torch.zeros(len(LABELS), dtype=torch.float32)
                label_mapping = {
                    'Atelectasis': 'Atelectasis', 'Cardiomegaly': 'Cardiomegaly', 
                    'Consolidation': 'Consolidation', 'Edema': 'Edema',
                    'Effusion': 'Effusion', 'Emphysema': 'Emphysema',
                    'Fibrosis': 'Fibrosis', 'Hernia': 'Hernia',
                    'Infiltration': 'Infiltration', 'Mass': 'Mass',
                    'Nodule': 'Nodule', 'Pleural_Thickening': 'Pleural Thickening',
                    'Pneumonia': 'Pneumonia', 'Pneumothorax': 'Pneumothorax'
                }
                
                for nih_label, our_label in label_mapping.items():
                    if item.get(nih_label, 0) == 1:
                        if our_label in LABELS:
                            label_idx = LABELS.index(our_label)
                            labels[label_idx] = 1.0
                
                # Add to batch
                batch_images.append(image_tensor)
                batch_labels.append(labels)
                processed_samples += 1
                
                # Process batch when full
                if len(batch_images) == args.batch_size:
                    # Stack batch
                    images_batch = torch.stack(batch_images).to(device)
                    labels_batch = torch.stack(batch_labels).to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(images_batch)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_preds.append(outputs.detach().cpu().numpy())
                    train_targets.append(labels_batch.cpu().numpy())
                    
                    # Clear batch
                    batch_images = []
                    batch_labels = []
                    
                    # Update progress
                    progress_bar.update(args.batch_size)
                    progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
            except StopIteration:
                # Restart iterator if we run out of data
                train_iter = iter(train_dataset_stream)
            except Exception as e:
                print(f"⚠️  Skipping sample due to error: {e}")
                continue
        
        progress_bar.close()
        
        # Process any remaining samples in batch
        if batch_images:
            images_batch = torch.stack(batch_images).to(device)
            labels_batch = torch.stack(batch_labels).to(device)
            
            optimizer.zero_grad()
            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(labels_batch.cpu().numpy())
        
        # Quick validation with streaming
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        val_iter = iter(train_dataset_stream)  # Use train stream for validation
        val_processed = 0
        
        with torch.no_grad():
            while val_processed < val_samples:
                try:
                    item = next(val_iter)
                    
                    # Process validation sample (same as training)
                    image = item['image'].convert('RGB')
                    image_tensor = trainer.transform(image).unsqueeze(0).to(device)
                    
                    labels = torch.zeros(1, len(LABELS), dtype=torch.float32).to(device)
                    for nih_label, our_label in label_mapping.items():
                        if item.get(nih_label, 0) == 1 and our_label in LABELS:
                            label_idx = LABELS.index(our_label)
                            labels[0, label_idx] = 1.0
                    
                    outputs = model(image_tensor)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    val_preds.append(outputs.cpu().numpy())
                    val_targets.append(labels.cpu().numpy())
                    val_processed += 1
                    
                except (StopIteration, Exception):
                    val_iter = iter(train_dataset_stream)
                    continue
        
        # Calculate metrics
        num_batches = len(train_preds)
        train_loss /= max(num_batches, 1)
        val_loss /= max(val_processed, 1)
        
        # Calculate AUC if we have predictions
        try:
            if train_preds and val_preds:
                train_preds_np = np.concatenate(train_preds)
                train_targets_np = np.concatenate(train_targets)
                val_preds_np = np.concatenate(val_preds)
                val_targets_np = np.concatenate(val_targets)
                
                train_auc = roc_auc_score(train_targets_np, train_preds_np, average='macro')
                val_auc = roc_auc_score(val_targets_np, val_preds_np, average='macro')
            else:
                train_auc = val_auc = 0.0
        except:
            train_auc = val_auc = 0.0
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save training history
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'samples_processed': processed_samples
        }
        training_history.append(epoch_stats)
        
        print(f"📈 Epoch {epoch+1}/{args.epochs} Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"   Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        print(f"   Samples: {processed_samples} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"💾 Saving best model (Val Loss: {val_loss:.4f})")
            
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_auc': val_auc,
                'training_history': training_history
            }, 'models/lung_classifier_best.pth')
    
    print(f"\n🎉 Training completed!")
    print(f"💯 Best validation loss: {best_val_loss:.4f}")
    print(f"🌐 No local storage used - everything streamed!")
    
    # Save final model
    trainer.save_model("models/lung_classifier_final.pth")
    
    # Save training history
    with open('models/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"📁 Models saved to models/ directory")
    return training_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XightMD Lung Classifier (Streaming)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--samples-per-epoch', type=int, default=1000, help='Samples per epoch (streaming)')
    parser.add_argument('--val-samples', type=int, default=200, help='Validation samples (streaming)')
    
    args = parser.parse_args()
    
    print(f"🌐 STREAMING MODE: No local download!")
    print(f"📊 Will process {args.samples_per_epoch} samples per epoch")
    print(f"⚡ Fast training without storage requirements")
    
    train_model(args)