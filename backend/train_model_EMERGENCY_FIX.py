import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from utils.lung_classifier import LungDiseaseClassifier, LABELS
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, hamming_loss
from tqdm import tqdm
import json

class FixedNIHDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, max_samples=None):
        self.transform = transform
        
        # Load data properly
        print(f"🔄 Loading data...")
        self.data = []
        count = 0
        
        for item in hf_dataset:
            self.data.append(item)
            count += 1
            if max_samples and count >= max_samples:
                break
                
        print(f"✅ Loaded {len(self.data)} samples")
        
        # Check label distribution
        label_counts = {}
        for item in self.data:
            labels = item.get('label', [])
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        
        print("📊 Label distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")
    
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
            
            # Create proper multi-label tensor
            labels = torch.zeros(len(LABELS), dtype=torch.float32)
            label_list = item.get('label', [])
            
            if isinstance(label_list, list):
                for disease_name in label_list:
                    disease_name = disease_name.strip()
                    if disease_name in LABELS:
                        label_idx = LABELS.index(disease_name)
                        labels[label_idx] = 1.0
                    elif disease_name == 'Pleural_Thickening' and 'Pleural Thickening' in LABELS:
                        label_idx = LABELS.index('Pleural Thickening')
                        labels[label_idx] = 1.0
            
            return image, labels
            
        except Exception as e:
            print(f"Error in sample {idx}: {e}")
            # Return zeros on error
            dummy_image = torch.zeros(3, 224, 224)
            dummy_labels = torch.zeros(len(LABELS), dtype=torch.float32)
            return dummy_image, dummy_labels

def train_fixed_model():
    print("🚨 EMERGENCY MODEL TRAINING FIX")
    print("="*50)
    
    # Load dataset
    dataset = load_dataset("BahaaEldin0/NIH-Chest-Xray-14", trust_remote_code=True)
    
    # Create transform
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FixedNIHDataset(dataset['train'], transform=transform, max_samples=5000)
    
    # Split for validation
    val_size = 1000
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create FIXED model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # CRITICAL: Use 15 classes (not 14!)
    model = LungDiseaseClassifier(num_classes=15, pretrained=True)
    model.to(device)
    
    # FIXED loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    print("🚀 Starting FIXED training...")
    
    best_f1 = 0.0
    for epoch in range(10):
        print(f"\n🔄 Epoch {epoch+1}/10")
        
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Convert to probabilities for metrics
            probs = torch.sigmoid(outputs)
            train_preds.append(probs.detach().cpu().numpy())
            train_targets.append(labels.cpu().numpy())
        
        # Calculate training metrics
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        
        train_preds_binary = (train_preds > 0.3).astype(int)
        train_f1 = f1_score(train_targets, train_preds_binary, average='macro', zero_division=0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        
        val_preds_binary = (val_preds > 0.3).astype(int)
        val_f1 = f1_score(val_targets, val_preds_binary, average='macro', zero_division=0)
        
        # Print results
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'epoch': epoch + 1
            }, 'models/lung_classifier_EMERGENCY_FIXED.pth')
            print(f"✅ Saved best model with F1: {best_f1:.4f}")
        
        scheduler.step(val_f1)
        
        # Early stopping if F1 is still terrible after 5 epochs
        if epoch >= 4 and val_f1 < 0.1:
            print("🚨 Model still broken after 5 epochs - stopping")
            break
    
    print(f"\n🎉 Training completed! Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    train_fixed_model()