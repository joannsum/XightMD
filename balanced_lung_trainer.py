import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import time
from PIL import Image
import random
from collections import defaultdict

LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration", 
    "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax",
    "No Finding"
]

class SimpleLungModel(nn.Module):
    """Simpler, more focused model"""
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        
        # Use ResNet18 - smaller, faster, easier to train
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.fc.in_features, len(LABELS))
        )
    
    def forward(self, x):
        return self.backbone(x)

class BalancedNIHDataset(Dataset):
    """Dataset with balanced sampling - limits 'No Finding' samples"""
    
    def __init__(self, dataset_split, max_samples_per_class=1000, max_no_finding=500):
        print("🔄 Creating BALANCED dataset...")
        
        # Group samples by labels
        self.label_groups = defaultdict(list)
        self.samples_with_pathology = []
        self.no_finding_samples = []
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("🔍 Analyzing dataset for class balance...")
        count = 0
        
        for item in dataset_split:
            try:
                labels = item.get('label', [])
                
                # Check if it's a "No Finding" case
                if not labels or labels == ['No Finding']:
                    if len(self.no_finding_samples) < max_no_finding:
                        self.no_finding_samples.append(item)
                else:
                    # Has actual pathology
                    self.samples_with_pathology.append(item)
                    
                    # Group by individual conditions
                    for label in labels:
                        if label.strip() in LABELS and label.strip() != 'No Finding':
                            if len(self.label_groups[label.strip()]) < max_samples_per_class:
                                self.label_groups[label.strip()].append(item)
                
                count += 1
                if count % 5000 == 0:
                    print(f"   Processed {count} samples...")
                
                # Stop early if we have enough samples
                total_pathology = len(self.samples_with_pathology)
                if total_pathology >= max_samples_per_class * 10:  # Reasonable limit
                    break
                    
            except Exception as e:
                continue
        
        # Create balanced final dataset
        self.final_dataset = []
        
        # Add pathology samples (limit to avoid imbalance)
        pathology_limit = min(len(self.samples_with_pathology), max_samples_per_class * 5)
        self.final_dataset.extend(self.samples_with_pathology[:pathology_limit])
        
        # Add limited "No Finding" samples
        self.final_dataset.extend(self.no_finding_samples)
        
        # Shuffle
        random.shuffle(self.final_dataset)
        
        print(f"✅ Balanced dataset created:")
        print(f"   Pathology samples: {len(self.samples_with_pathology)}")
        print(f"   No Finding samples: {len(self.no_finding_samples)}")
        print(f"   Total samples: {len(self.final_dataset)}")
        print(f"   No Finding ratio: {len(self.no_finding_samples)/len(self.final_dataset)*100:.1f}%")
        
        # Print per-condition stats
        print(f"\n📊 Per-condition distribution:")
        for condition, samples in self.label_groups.items():
            print(f"   {condition:<20}: {len(samples):>4} samples")
    
    def __len__(self):
        return len(self.final_dataset)
    
    def __getitem__(self, idx):
        try:
            item = self.final_dataset[idx]
            
            # Process image
            image = item['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Check image quality
            if image.size[0] < 100 or image.size[1] < 100:
                # Image too small, return dummy
                raise ValueError("Image too small")
            
            image = self.transform(image)
            
            # Process labels
            labels = torch.zeros(len(LABELS), dtype=torch.float32)
            label_list = item.get('label', [])
            
            if not label_list:
                # No labels = "No Finding"
                labels[LABELS.index('No Finding')] = 1.0
            else:
                for disease_name in label_list:
                    disease_name = disease_name.strip()
                    if disease_name in LABELS:
                        idx = LABELS.index(disease_name)
                        labels[idx] = 1.0
                    elif disease_name == 'Pleural_Thickening':
                        idx = LABELS.index('Pleural Thickening')
                        labels[idx] = 1.0
            
            return image, labels
            
        except Exception as e:
            # Return dummy data
            dummy_image = torch.zeros(3, 224, 224)
            dummy_labels = torch.zeros(len(LABELS), dtype=torch.float32)
            dummy_labels[LABELS.index('No Finding')] = 1.0  # Default to "No Finding"
            return dummy_image, dummy_labels

def test_balanced_training():
    """Test training with balanced dataset"""
    print("🎯 BALANCED LUNG DISEASE TRAINING")
    print("="*50)
    
    # Load dataset
    print("📥 Loading NIH dataset...")
    dataset = load_dataset("BahaaEldin0/NIH-Chest-Xray-14", trust_remote_code=True)
    
    # Create BALANCED dataset
    train_dataset = BalancedNIHDataset(
        dataset['train'], 
        max_samples_per_class=800,  # Limit per pathology
        max_no_finding=400          # Heavily limit "No Finding"
    )
    
    # Create validation split
    val_size = min(500, len(train_dataset) // 4)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"📊 Final dataset:")
    print(f"   Training: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleLungModel().to(device)
    
    # Training setup - AGGRESSIVE for faster learning
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher LR
    
    print(f"\n🎯 Training setup:")
    print(f"   Device: {device}")
    print(f"   Model: ResNet18")
    print(f"   Learning rate: 0.01")
    print(f"   Batch size: 8")
    
    # Test one batch first
    print(f"\n🔍 Testing one batch...")
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        print(f"   Batch shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Positive labels: {labels.sum().item():.1f}")
        print(f"   Labels per sample: {labels.sum().item()/labels.shape[0]:.2f}")
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        print(f"   Loss: {loss.item():.4f}")
        
        # Check predictions
        probs = torch.sigmoid(outputs)
        print(f"   Prediction range: {probs.min().item():.3f} - {probs.max().item():.3f}")
        print(f"   Prediction mean: {probs.mean().item():.3f}")
        
        break
    
    print(f"\n🚀 Starting training...")
    
    best_f1 = 0.0
    
    for epoch in range(10):  # Quick test
        print(f"\n--- Epoch {epoch+1} ---")
        
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            # Store predictions
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                train_preds.append(probs.cpu().numpy())
                train_targets.append(labels.cpu().numpy())
        
        # Calculate metrics with different thresholds
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        
        best_thresh = 0.5
        best_f1_score = 0.0
        
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            pred_binary = (train_preds > thresh).astype(int)
            f1 = f1_score(train_targets, pred_binary, average='macro', zero_division=0)
            if f1 > best_f1_score:
                best_f1_score = f1
                best_thresh = thresh
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Train F1: {best_f1_score:.4f} (threshold: {best_thresh})")
        
        # Quick validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                
                val_preds.append(probs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())
        
        if val_preds:
            val_preds = np.vstack(val_preds)
            val_targets = np.vstack(val_targets)
            
            val_pred_binary = (val_preds > best_thresh).astype(int)
            val_f1 = f1_score(val_targets, val_pred_binary, average='macro', zero_division=0)
            
            print(f"Val F1: {val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), 'balanced_lung_model.pth')
                print(f"✅ New best model saved! F1: {val_f1:.4f}")
        
        # Show detailed per-class results
        if epoch % 2 == 0:  # Every 2 epochs
            print(f"\n📊 Per-class predictions (threshold: {best_thresh}):")
            for i, label in enumerate(LABELS):
                if np.sum(val_targets[:, i]) > 0:
                    class_f1 = f1_score(val_targets[:, i], val_pred_binary[:, i], zero_division=0)
                    true_count = int(np.sum(val_targets[:, i]))
                    pred_count = int(np.sum(val_pred_binary[:, i]))
                    print(f"   {label:<20}: F1={class_f1:.3f} (True:{true_count}, Pred:{pred_count})")
        
        # Early success check
        if val_f1 > 0.3:
            print(f"\n🎉 SUCCESS! F1 > 0.3 achieved at epoch {epoch+1}")
            break
        elif val_f1 > 0.1:
            print(f"💪 Progress! F1 > 0.1 - model is learning!")
    
    print(f"\n🏁 Training complete! Best F1: {best_f1:.4f}")
    
    if best_f1 < 0.1:
        print(f"\n🚨 Model still struggling. Possible issues:")
        print(f"   - Dataset corruption")
        print(f"   - Image preprocessing problems") 
        print(f"   - Label format issues")
        print(f"   - Hardware/memory constraints")
    else:
        print(f"\n✅ Model is learning! You can now scale up the training.")

if __name__ == "__main__":
    test_balanced_training()