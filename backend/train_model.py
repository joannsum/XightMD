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

# UPDATED LABELS - Add "No Finding" as the last class
LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax",
    "No Finding"  # Add this!
]

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

        # FIXED label mapping
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
            'Pneumothorax': 'Pneumothorax',
            'No Finding': 'No Finding'  # Fixed mapping
        }

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

            # FIXED: Proper label encoding
                labels = torch.zeros(len(LABELS), dtype=torch.float32)

            # Check if this is a "No Finding" case
            has_finding = False

            # First, check for positive findings
            for nih_label, our_label in self.label_mapping.items():
                if nih_label != 'No Finding' and item.get(nih_label, 0) == 1:
                        if our_label in LABELS:
                            label_idx = LABELS.index(our_label)
                            labels[label_idx] = 1.0
                        has_finding = True
                    
            # If no findings detected, set "No Finding" to 1
            if not has_finding and item.get('No Finding', 0) == 1:
                no_finding_idx = LABELS.index('No Finding')
                labels[no_finding_idx] = 1.0

            # Debug: Print label info for first few samples
            if idx < 5:
                positive_labels = [LABELS[i] for i, val in enumerate(labels) if val == 1.0]
                print(f"Sample {idx} labels: {positive_labels}")

            return image, labels

        except Exception as e:
            print(f"⚠️  Error loading sample {idx}: {e}")
            dummy_image = torch.zeros(3, 224, 224)
            dummy_labels = torch.zeros(len(LABELS), dtype=torch.float32)
            return dummy_image, dummy_labels

def calculate_metrics(targets, predictions):
    metrics = {}

    print("\nDEBUG METRICS INFO:")
    print(f"Targets shape: {targets.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample of targets:\n{targets[:5]}")
    print(f"Sample of predictions:\n{predictions[:5]}")
    print(f"Unique values in targets: {np.unique(targets)}")
    print(f"Unique values in predictions: {np.unique(predictions)}")

    predictions_binary = (predictions > 0.5).astype(int)
    print(f"Unique values after thresholding: {np.unique(predictions_binary)}")

    positive_samples = np.sum(targets, axis=0)
    print(f"Positive samples per class: {positive_samples}")

    metrics['hamming_loss'] = hamming_loss(targets, predictions_binary)
    metrics['avg_precision'] = average_precision_score(targets, predictions, average='macro')
    metrics['f1_macro'] = f1_score(targets, predictions_binary, average='macro')
    metrics['f1_micro'] = f1_score(targets, predictions_binary, average='micro')

    print("\nCalculated metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")

    return metrics

def train_model(args):
    print("🏥 Loading NIH Chest X-ray 14 dataset (streaming mode)...")
    try:
        dataset = load_dataset(
            "BahaaEldin0/NIH-Chest-Xray-14",
            streaming=True,
            trust_remote_code=True
            )

        print(f"✅ Dataset loaded in streaming mode!")
        print(f"🌐 Data will be streamed from HuggingFace servers")

        train_dataset_stream = dataset['train']

        print(f"📊 Using streaming mode with limited samples per epoch")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 Make sure you have internet connection and HuggingFace access")
        return

    # After loading the dataset
    print("\nDEBUG: Checking first sample from dataset")
    first_sample = next(iter(dataset['train']))
    print("Available keys:", first_sample.keys())

    # FIXED: Use the instance's label_mapping
    sample_dataset = NIHChestXrayDataset(None)  # Just to get label_mapping
    print("Label values:", {k: v for k, v in first_sample.items() if k in sample_dataset.label_mapping})

    checkpoint_dir = 'models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_frequency = args.checkpoint_freq
    trainer = LungClassifierTrainer()

    samples_per_epoch = args.samples_per_epoch
    val_samples = args.val_samples

    print(f"🔄 Training with {samples_per_epoch} samples per epoch")
    print(f"🔄 Validation with {val_samples} samples")

    device = trainer.device
    model = trainer.model

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    start_epoch = 0
    best_val_loss = float('inf')
    training_history = []
    if args.resume_checkpoint:
        print(f"📂 Loading checkpoint from: {args.resume_checkpoint}")
        try:
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            filename_epoch = extract_epoch_from_checkpoint(args.resume_checkpoint)
            if filename_epoch is not None:
                start_epoch = filename_epoch
                print(f"✅ Starting from epoch {start_epoch} (based on filename)")
            else:
                start_epoch = checkpoint['epoch']
                print(f"✅ Starting from epoch {start_epoch} (based on checkpoint data)")

            best_val_loss = checkpoint.get('val_loss', float('inf'))
            training_history = checkpoint.get('training_history', [])

            print(f"📈 Previous best val loss: {best_val_loss:.4f}")

            checkpoint_epoch = checkpoint.get('epoch')
            if filename_epoch and checkpoint_epoch and filename_epoch != checkpoint_epoch:
                print(f"⚠️  Warning: Checkpoint filename indicates epoch {filename_epoch} "
                      f"but checkpoint data shows epoch {checkpoint_epoch}")

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return

    total_epochs = args.epochs + start_epoch
    print(f"🎯 Will train until epoch {total_epochs}")

    print(f"🚀 Starting training for {args.epochs} epochs...")
    print(f"🔧 Device: {device}")
    print(f"🔧 Batch size: {args.batch_size}")
    print(f"🔧 Learning rate: {args.learning_rate}")
    print(f"🌐 Streaming mode: No local storage used!")

    for epoch in range(start_epoch, total_epochs):
        print(f"\n🔄 Epoch {epoch+1}/{total_epochs}")

        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        train_iter = iter(train_dataset_stream)
        processed_samples = 0
        batch_images = []
        batch_labels = []

        debug_labels_sum = 0
        debug_predictions_sum = 0

        progress_bar = tqdm(total=samples_per_epoch, desc=f"Training Epoch {epoch+1}")

        while processed_samples < samples_per_epoch:
            try:
                item = next(train_iter)

                if processed_samples < 5:
                    print(f"\nDEBUG Sample {processed_samples} raw labels:")
                    sample_dataset = NIHChestXrayDataset(None)
                    for k, v in item.items():
                        if k in sample_dataset.label_mapping:
                            print(f"{k}: {v}")

                image = item['image']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_tensor = trainer.transform(image)

                # FIXED: Use proper label encoding logic
                labels = torch.zeros(len(LABELS), dtype=torch.float32)
                has_finding = False
                label_assignments = []

                # Check for pathological findings first
                sample_dataset = NIHChestXrayDataset(None)
                for nih_label, our_label in sample_dataset.label_mapping.items():
                    if nih_label != 'No Finding' and item.get(nih_label, 0) == 1:
                        if our_label in LABELS:
                            label_idx = LABELS.index(our_label)
                            labels[label_idx] = 1.0
                            has_finding = True
                            label_assignments.append(f"{nih_label} -> {our_label}")

                # If no pathological findings, check for "No Finding"
                if not has_finding and item.get('No Finding', 0) == 1:
                    no_finding_idx = LABELS.index('No Finding')
                    labels[no_finding_idx] = 1.0
                    label_assignments.append("No Finding -> No Finding")

                if processed_samples < 5:
                    print(f"Label assignments for sample {processed_samples}:")
                    print("\n".join(label_assignments) if label_assignments else "No labels assigned")
                    print(f"Final label tensor: {labels}")
                    print(f"Label sum: {torch.sum(labels)}")

                batch_images.append(image_tensor)
                batch_labels.append(labels)
                processed_samples += 1

                if len(batch_images) == args.batch_size:
                    images_batch = torch.stack(batch_images).to(device)
                    labels_batch = torch.stack(batch_labels).to(device)

                    print(f"\nDEBUG Batch stats:")
                    print(f"Batch labels sum: {torch.sum(labels_batch)}")
                    print(f"Batch labels non-zero: {torch.nonzero(labels_batch).shape}")

                    optimizer.zero_grad()
                    outputs = model(images_batch)

                    print(f"Batch predictions > 0.5: {torch.sum(outputs > 0.5)}")

                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_preds.append(outputs.detach().cpu().numpy())
                    train_targets.append(labels_batch.cpu().numpy())

                    debug_labels_sum += torch.sum(labels_batch).item()
                    debug_predictions_sum += torch.sum(outputs > 0.5).item()

                    batch_images = []
                    batch_labels = []
                    progress_bar.update(args.batch_size)
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Pos Labels': f'{debug_labels_sum}',
                        'Pos Preds': f'{debug_predictions_sum}'
                    })

            except StopIteration:
                train_iter = iter(train_dataset_stream)
            except Exception as e:
                print(f"⚠️  Error in training loop: {str(e)}")
                continue

        progress_bar.close()

        print("\nDEBUG END OF EPOCH STATS:")
        print(f"Total positive labels: {debug_labels_sum}")
        print(f"Total positive predictions: {debug_predictions_sum}")

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

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        val_iter = iter(train_dataset_stream)
        val_processed = 0

        with torch.no_grad():
            while val_processed < val_samples:
                try:
                    item = next(val_iter)

                    image = item['image'].convert('RGB')
                    image_tensor = trainer.transform(image).unsqueeze(0).to(device)

                    labels = torch.zeros(1, len(LABELS), dtype=torch.float32).to(device)
                    has_finding = False

                    sample_dataset = NIHChestXrayDataset(None)
                    for nih_label, our_label in sample_dataset.label_mapping.items():
                        if nih_label != 'No Finding' and item.get(nih_label, 0) == 1 and our_label in LABELS:
                            label_idx = LABELS.index(our_label)
                            labels[0, label_idx] = 1.0
                            has_finding = True

                    if not has_finding and item.get('No Finding', 0) == 1:
                        no_finding_idx = LABELS.index('No Finding')
                        labels[0, no_finding_idx] = 1.0

                    outputs = model(image_tensor)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    val_preds.append(outputs.cpu().numpy())
                    val_targets.append(labels.cpu().numpy())
                    val_processed += 1

                except (StopIteration, Exception):
                    val_iter = iter(train_dataset_stream)
                    continue

        num_batches = len(train_preds)
        train_loss /= max(num_batches, 1)
        val_loss /= max(val_processed, 1)

        train_preds_np = np.concatenate(train_preds)
        train_targets_np = np.concatenate(train_targets)
        val_preds_np = np.concatenate(val_preds)
        val_targets_np = np.concatenate(val_targets)

        train_metrics = calculate_metrics(train_targets_np, train_preds_np)
        val_metrics = calculate_metrics(val_targets_np, val_preds_np)
        scheduler.step(val_loss)

        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'samples_processed': processed_samples
        }
        training_history.append(epoch_stats)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Hamming Loss: {train_metrics['hamming_loss']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Hamming Loss: {val_metrics['hamming_loss']:.4f}")
        print(f"Samples: {processed_samples} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if args.verbose:
            print("\nDetailed Metrics:")
            print("   Train Metrics:")
            print(f"      Avg Precision: {train_metrics['avg_precision']:.4f}")
            print(f"      F1 (macro): {train_metrics['f1_macro']:.4f}")
            print(f"      F1 (micro): {train_metrics['f1_micro']:.4f}")
            print("   Val Metrics:")
            print(f"      Avg Precision: {val_metrics['avg_precision']:.4f}")
            print(f"      F1 (macro): {val_metrics['f1_macro']:.4f}")
            print(f"      F1 (micro): {val_metrics['f1_micro']:.4f}")

        if (epoch + 1) % checkpoint_frequency == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'lung_classifier_checkpoint_epoch_{epoch+1}.pth'
            )
            print(f"💾 Saving periodic checkpoint to {checkpoint_path}")
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
            print(f"💾 Saving best model (Val Loss: {val_loss:.4f})")

            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'training_history': training_history
            }, 'models/lung_classifier_best.pth')

    print(f"\n🎉 Training completed!")
    print(f"💯 Best validation loss: {best_val_loss:.4f}")
    print(f"🌐 No local storage used - everything streamed!")

    trainer.save_model("models/lung_classifier_final.pth")

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
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=1,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--resume-checkpoint',
        type=str,
        help='Path to checkpoint file to resume training from'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed metrics for each epoch'
    )

    args = parser.parse_args()

    print(f"🌐 STREAMING MODE: No local download!")
    if args.resume_checkpoint:
        print(f"📂 Will resume training from: {args.resume_checkpoint}")
    print(f"📊 Will process {args.samples_per_epoch} samples per epoch")
    print(f"⚡ Fast training without storage requirements")
    
    train_model(args)
    print("🚀 Training started...")