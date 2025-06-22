import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from utils.lung_classifier import LungClassifierTrainer, LungDiseaseClassifier, LABELS
import argparse
import os
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, hamming_loss, f1_score
import pandas as pd
from tqdm import tqdm
import json
import re

def extract_epoch_from_checkpoint(checkpoint_path):
    try:
        match = re.search(r'epoch_(\d+)\.pth$', checkpoint_path)
        if match:
            return int(match.group(1))
        return None
    except Exception:
        return None

class NIHChestXrayDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, split='train', max_samples=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.split = split
        self.max_samples = max_samples

        if self.max_samples:
            print(f"🔄 Using only {self.max_samples} samples for {split}")

    def __len__(self):
        if self.max_samples:
            return min(self.max_samples, len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            image = item['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
                    
            if self.transform:
                image = self.transform(image)
                
            # Handle the actual label format (list of strings)
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
            print(f"⚠️  Error loading sample {idx}: {e}")
            dummy_image = torch.zeros(3, 224, 224)
            dummy_labels = torch.zeros(len(LABELS), dtype=torch.float32)
            return dummy_image, dummy_labels

def calculate_metrics(targets, predictions):
    predictions_binary = (predictions > 0.5).astype(int)
    positive_samples = np.sum(targets, axis=0)

    metrics = {}
    if np.sum(positive_samples) > 0:
        metrics['hamming_loss'] = hamming_loss(targets, predictions_binary)
        try:
            metrics['avg_precision'] = average_precision_score(targets, predictions, average='macro')
        except:
            metrics['avg_precision'] = 0.0
        metrics['f1_macro'] = f1_score(targets, predictions_binary, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(targets, predictions_binary, average='micro', zero_division=0)
    else:
        metrics['hamming_loss'] = 0.0
        metrics['avg_precision'] = 0.0
        metrics['f1_macro'] = 0.0
        metrics['f1_micro'] = 0.0

    return metrics

def train_model(args):
    print("🏥 Loading NIH Chest X-ray 14 dataset...")
    try:
        dataset = load_dataset("BahaaEldin0/NIH-Chest-Xray-14", streaming=True, trust_remote_code=True)
        print("✅ Dataset loaded!")
        train_dataset_stream = dataset['train']
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Quick check of first sample
    first_sample = next(iter(dataset['train']))
    print(f"Sample label format: {first_sample.get('label', [])}")

    checkpoint_dir = 'models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer = LungClassifierTrainer()

    print(f"🔧 Device: {trainer.device}")
    print(f"🔧 Training: {args.samples_per_epoch} samples/epoch, Validation: {args.val_samples} samples")
    device = trainer.device
    model = trainer.model
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    start_epoch = 0
    best_val_loss = float('inf')
    training_history = []
    
    if args.resume_checkpoint:
        print(f"📂 Loading checkpoint: {args.resume_checkpoint}")
        try:
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = extract_epoch_from_checkpoint(args.resume_checkpoint) or checkpoint['epoch']
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            training_history = checkpoint.get('training_history', [])
            print(f"✅ Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return

    print(f"🚀 Starting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"\n🔄 Epoch {epoch+1}")

        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        train_iter = iter(train_dataset_stream)
        processed_samples = 0
        batch_images = []
        batch_labels = []
        positive_labels_count = 0
        progress_bar = tqdm(total=args.samples_per_epoch, desc=f"Training")

        while processed_samples < args.samples_per_epoch:
            try:
                item = next(train_iter)

                image = item['image']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_tensor = trainer.transform(image)

                # Handle labels
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
                batch_images.append(image_tensor)
                batch_labels.append(labels)
                processed_samples += 1
                positive_labels_count += torch.sum(labels).item()

                if len(batch_images) == args.batch_size:
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

                    batch_images = []
                    batch_labels = []
                    progress_bar.update(args.batch_size)
                    progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            except StopIteration:
                train_iter = iter(train_dataset_stream)
            except Exception as e:
                continue

        progress_bar.close()

        # Handle remaining batch
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

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        val_iter = iter(train_dataset_stream)
        val_processed = 0

        with torch.no_grad():
            while val_processed < args.val_samples:
                try:
                    item = next(val_iter)
                    image = item['image'].convert('RGB')
                    image_tensor = trainer.transform(image).unsqueeze(0).to(device)

                    labels = torch.zeros(1, len(LABELS), dtype=torch.float32).to(device)
                    label_list = item.get('label', [])
                    
                    if isinstance(label_list, list):
                        for disease_name in label_list:
                            disease_name = disease_name.strip()
                            if disease_name in LABELS:
                                label_idx = LABELS.index(disease_name)
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

        if train_preds and train_targets:
            train_preds_np = np.concatenate(train_preds)
            train_targets_np = np.concatenate(train_targets)
            train_metrics = calculate_metrics(train_targets_np, train_preds_np)
        else:
            train_metrics = {'hamming_loss': 0.0, 'avg_precision': 0.0, 'f1_macro': 0.0, 'f1_micro': 0.0}

        if val_preds and val_targets:
            val_preds_np = np.concatenate(val_preds)
            val_targets_np = np.concatenate(val_targets)
            val_metrics = calculate_metrics(val_targets_np, val_preds_np)
        else:
            val_metrics = {'hamming_loss': 0.0, 'avg_precision': 0.0, 'f1_macro': 0.0, 'f1_micro': 0.0}

        scheduler.step(val_loss)

        # Log results
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Hamming: {train_metrics['hamming_loss']:.4f} | Val Hamming: {val_metrics['hamming_loss']:.4f}")
        print(f"Positive labels found: {positive_labels_count}")
        if args.verbose:
            print(f"Train F1: {train_metrics['f1_macro']:.4f} | Val F1: {val_metrics['f1_macro']:.4f}")

        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'positive_labels': positive_labels_count
        }
        training_history.append(epoch_stats)

        # Save checkpoints
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'lung_classifier_checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'training_history': training_history
            }, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"💾 New best model (Val Loss: {val_loss:.4f})")
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'training_history': training_history
            }, 'models/lung_classifier_best.pth')

    print(f"\n🎉 Training completed! Best val loss: {best_val_loss:.4f}")
    trainer.save_model("models/lung_classifier_final.pth")

    with open('models/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    return training_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XightMD Lung Classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--samples-per-epoch', type=int, default=100, help='Samples per epoch')
    parser.add_argument('--val-samples', type=int, default=20, help='Validation samples')
    parser.add_argument('--checkpoint-freq', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--resume-checkpoint', type=str, help='Path to checkpoint file to resume training from')
    parser.add_argument('--verbose', action='store_true', help='Print detailed metrics')

    args = parser.parse_args()
    train_model(args)