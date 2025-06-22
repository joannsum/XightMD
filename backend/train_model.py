import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from PIL import Image

# Simple label list - 14 conditions (no "No Finding" for now)
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
    'Pleural Thickening', 'Pneumonia', 'Pneumothorax'
]

class SimpleLungModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, len(LABELS))
        # NO SIGMOID - BCEWithLogitsLoss handles it
    
    def forward(self, x):
        return self.backbone(x)

class SimpleDataset(Dataset):
    def __init__(self, max_samples=500):
        print("Loading dataset...")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load dataset
        dataset = load_dataset("BahaaEldin0/NIH-Chest-Xray-14", trust_remote_code=True)
        
        self.data = []
        count = 0
        
        for item in dataset['train']:
            self.data.append(item)
            count += 1
            if count >= max_samples:
                break
        
        print(f"Loaded {len(self.data)} samples")
        
        # Check data quality
        self._check_data()
    
    def _check_data(self):
        print("Checking data quality...")
        total_labels = 0
        
        for i, item in enumerate(self.data[:50]):
            labels = item.get('label', [])
            if labels:
                total_labels += len(labels)
                if i < 5:  # Print first 5
                    print(f"Sample {i}: {labels}")
        
        print(f"Average labels per sample: {total_labels/50:.2f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            
            # Process image
            image = item['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = self.transform(image)
            
            # Process labels
            labels = torch.zeros(len(LABELS))
            label_list = item.get('label', [])
            
            for disease in label_list:
                disease = disease.strip()
                if disease in LABELS:
                    idx = LABELS.index(disease)
                    labels[idx] = 1.0
                elif disease == 'Pleural_Thickening':
                    idx = LABELS.index('Pleural Thickening')
                    labels[idx] = 1.0
            
            return image, labels
            
        except Exception as e:
            print(f"Error in sample {idx}: {e}")
            return torch.zeros(3, 224, 224), torch.zeros(len(LABELS))

def test_basic_training():
    print("🚀 TESTING BASIC TRAINING FROM SCRATCH")
    print("="*50)
    
    # Create dataset
    dataset = SimpleDataset(max_samples=200)  # Very small for testing
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleLungModel().to(device)
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Test one batch first
    print("\nTesting one batch...")
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Positive labels: {labels.sum().item()}")
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        print(f"Output shape: {outputs.shape}")
        print(f"Loss: {loss.item():.4f}")
        
        # Check if we can backward
        loss.backward()
        print("✅ Backward pass successful")
        
        break
    
    # Train for a few epochs
    print("\nStarting training...")
    
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}")
        
        # Train
        model.train()
        train_loss = 0
        all_preds = []
        all_targets = []
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Get predictions
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(labels.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # Try different thresholds
        best_f1 = 0
        best_thresh = 0.5
        
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            pred_binary = (all_preds > thresh).astype(int)
            f1 = f1_score(all_targets, pred_binary, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Best F1: {best_f1:.4f} (threshold: {best_thresh})")
        
        # Quick validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                
                val_preds.append(preds.cpu().numpy())
                val_targets.append(labels.cpu().numpy())
        
        if val_preds:
            val_preds = np.vstack(val_preds)
            val_targets = np.vstack(val_targets)
            
            val_pred_binary = (val_preds > best_thresh).astype(int)
            val_f1 = f1_score(val_targets, val_pred_binary, average='macro', zero_division=0)
            
            print(f"Val F1: {val_f1:.4f}")
        
        # Debug info
        print(f"Prediction stats: min={all_preds.min():.3f}, max={all_preds.max():.3f}, mean={all_preds.mean():.3f}")
    
    print("\n✅ Basic training test complete!")
    
    # Save if it worked
    if best_f1 > 0.1:
        torch.save(model.state_dict(), 'simple_lung_model.pth')
        print(f"✅ Saved working model with F1: {best_f1:.4f}")
    else:
        print("❌ Model still not learning properly")

if __name__ == "__main__":
    test_basic_training()