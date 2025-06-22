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
import datetime
import time
import random

def get_condition_name(condition_idx):
    if condition_idx is not None and 0 <= condition_idx < len(LABELS):
        return LABELS[condition_idx]
    return "all_conditions"

def extract_epoch_from_checkpoint(checkpoint_path):
    try:
        match = re.search(r'epoch_(\d+)\.pth$', checkpoint_path)
        if match:
            return int(match.group(1))
        return None
    except Exception:
        return None

def get_batch_samples(dataset_stream, condition_idx, batch_size, trainer, current_iter=None):
    batch_images = []
    batch_labels = []

    stream_iter = current_iter if current_iter is not None else iter(dataset_stream)
    while len(batch_images) < batch_size:
        try:
            item = next(stream_iter)
            image = item['image'].convert('RGB')
            image_tensor = trainer.transform(image)

            label_list = item.get('label', [])
            condition_name = LABELS[condition_idx]
            has_condition = condition_name in label_list

            batch_images.append(image_tensor)
            batch_labels.append(float(has_condition))

        except StopIteration:
            stream_iter = iter(dataset_stream)
            continue

    return (torch.stack(batch_images),
            torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(1),
            stream_iter)

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

    current_condition = get_condition_name(args.condition_idx)
    print(f"\n🎯 Training for condition: {current_condition} (Index: {args.condition_idx})")

    condition_checkpoint_dir = os.path.join('models/checkpoints', current_condition)
    os.makedirs(condition_checkpoint_dir, exist_ok=True)
    trainer = LungClassifierTrainer()

    condition_idx = args.condition_idx if hasattr(args, 'condition_idx') else None
    subset_data = args.subset_data if hasattr(args, 'subset_data') else None

    if subset_data:
        dataset = NIHChestXrayDataset(
            subset_data,
            transform=trainer.transform,
            max_samples=args.max_samples if hasattr(args, 'max_samples') else None
        )
    device = trainer.device
    model = trainer.model
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    start_epoch = 0
    best_accuracy = 0.0
    patience_counter = 0
    training_history = []

    if args.resume_checkpoint:
        print(f"📂 Loading checkpoint from {args.resume_checkpoint}")
        try:
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            checkpoint_condition = checkpoint.get('condition', 'unknown')
            checkpoint_accuracy = checkpoint.get('best_accuracy', 0.0)

            print(f"✅ Loaded checkpoint trained on '{checkpoint_condition}' with accuracy: {checkpoint_accuracy:.4f}")
            print(f"🎯 Now training for condition: {current_condition}")

            if checkpoint_condition == current_condition:
                start_epoch = checkpoint['epoch']
                best_accuracy = checkpoint_accuracy
                training_history = checkpoint.get('training_history', [])
                print(f"📈 Continuing from epoch {start_epoch} for same condition")
            else:
                start_epoch = 0
                best_accuracy = 0.0
                training_history = []
                print(f"🔄 Starting fresh training for new condition using weights from {checkpoint_condition}")

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print(f"Starting with fresh model...")
            return None

    print(f"🚀 Starting training for {args.epochs} epochs...")

    target_accuracy = args.target_accuracy if hasattr(args, 'target_accuracy') else 0.8
    print(f"Target accuracy for {current_condition}: {target_accuracy}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"\n🔄 Epoch {epoch+1}")
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        processed_samples = 0
        current_iter = iter(train_dataset_stream)

        progress_bar = tqdm(total=args.samples_per_epoch, desc=f"Training")

        while processed_samples < args.samples_per_epoch:
            try:
                images_batch, labels_batch, current_iter = get_batch_samples(
                    train_dataset_stream,
                    args.condition_idx,
                    args.batch_size,
                    trainer,
                    current_iter
                )

                images_batch = images_batch.to(device)
                labels_batch = labels_batch.to(device)

                optimizer.zero_grad()
                outputs = model(images_batch)

                if condition_idx is not None:
                    outputs = outputs[:, condition_idx].unsqueeze(1)

                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                predictions = (outputs > 0.5).float()
                correct += (predictions == labels_batch).sum().item()
                total += labels_batch.numel()
                train_loss += loss.item()

                processed_samples += args.batch_size
                progress_bar.update(args.batch_size)
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{(correct/total if total > 0 else 0):.4f}'
                })

                if processed_samples % 100 == 0:
                    pos_ratio = labels_batch.mean().item()
                    print(f"\nBatch positive ratio: {pos_ratio:.2f}")

            except Exception as e:
                print(f"\nError in batch processing: {e}")
                continue

        progress_bar.close()

        epoch_accuracy = correct / total if total > 0 else 0
        print(f"\nEpoch {epoch+1} - Loss: {train_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            patience_counter = 0
            best_model_path = os.path.join(condition_checkpoint_dir, f'best_model_{current_condition}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
                'training_history': training_history,
                'condition': current_condition,
                'condition_idx': args.condition_idx
            }, best_model_path)
            print(f"✅ Saved best model for {current_condition} with accuracy: {best_accuracy:.4f}")
        if best_accuracy >= target_accuracy:
                print(f"🎯 Reached target accuracy of {target_accuracy} for {current_condition}")
                break
        else:
            patience_counter += 1

        if hasattr(args, 'early_stopping_patience') and patience_counter >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        scheduler.step(epoch_accuracy)

        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'accuracy': epoch_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        training_history.append(epoch_stats)

        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                condition_checkpoint_dir,
                f'{current_condition}_checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
                'training_history': training_history,
                'condition': current_condition,
                'condition_idx': args.condition_idx
            }, checkpoint_path)
            print(f"📦 Saved checkpoint for {current_condition} at epoch {epoch+1}")

    final_model_path = f"models/lung_classifier_{current_condition}_final.pth"
    trainer.save_model(final_model_path)

    history_path = os.path.join(condition_checkpoint_dir, f'training_history_{current_condition}.json')
    with open(history_path, 'w') as f:
        json.dump({
            'condition': current_condition,
            'condition_idx': args.condition_idx,
            'history': training_history,
            'best_accuracy': best_accuracy
        }, f, indent=2)
    print(f"\n🎉 Training completed for {current_condition}!")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Model saved as: {final_model_path}")
    print(f"Training history saved as: {history_path}")
    return best_accuracy

def train_all_conditions(args):
    target_accuracy = 0.8
    completed_conditions = []
    starting_condition = args.condition_idx if args.condition_idx is not None else 0
    original_checkpoint = args.resume_checkpoint

    print(f"🎯 Target accuracy for each condition: {target_accuracy}")
    print(f"🎯 Starting from condition {starting_condition}: {LABELS[starting_condition]}")

    if original_checkpoint:
        if not os.path.exists(original_checkpoint):
            print(f"❌ Checkpoint file not found: {original_checkpoint}")
            print("Starting with fresh model instead.")
            original_checkpoint = None
        else:
            print(f"📦 Will use weights from checkpoint: {original_checkpoint}")

    progress_file = 'models/training_progress.json'
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            completed_conditions = json.load(f).get('completed_conditions', [])
        print(f"📝 Loaded previous progress. Completed conditions: {completed_conditions}")

    for condition_idx in range(starting_condition, len(LABELS)):
        condition_name = LABELS[condition_idx]

        if condition_idx in completed_conditions:
            print(f"✅ Skipping {condition_name} - already reached target accuracy")
            continue

        print(f"\n{'='*50}")
        print(f"🎯 Starting training for condition {condition_idx}: {condition_name}")
        print(f"{'='*50}")

        args.condition_idx = condition_idx

        condition_checkpoint_dir = os.path.join('models/checkpoints', condition_name)
        condition_checkpoint = None

        if os.path.exists(condition_checkpoint_dir):
            checkpoints = [f for f in os.listdir(condition_checkpoint_dir)
                         if f.endswith('.pth') and 'checkpoint' in f]
            if checkpoints:
                latest_checkpoint = max(checkpoints,
                    key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1))
                    if re.search(r'epoch_(\d+)', x) else 0
                )
                condition_checkpoint = os.path.join(condition_checkpoint_dir, latest_checkpoint)
                print(f"📦 Found existing checkpoint for {condition_name}: {latest_checkpoint}")

        if condition_checkpoint and os.path.exists(condition_checkpoint):
            args.resume_checkpoint = condition_checkpoint
            print(f"↪️ Resuming {condition_name} from its previous checkpoint")
        elif original_checkpoint:
            args.resume_checkpoint = original_checkpoint
            print(f"↪️ Starting {condition_name} using weights from specified checkpoint")
        else:
            args.resume_checkpoint = None
            print(f"↪️ Starting {condition_name} with fresh weights")

        try:
            best_accuracy = train_model(args)
            if best_accuracy is None:
                print(f"⚠️ Training failed for {condition_name}. Moving to next condition.")
                continue

            if best_accuracy >= target_accuracy:
                print(f"🎉 Successfully reached target accuracy for {condition_name}: {best_accuracy:.4f}")
                completed_conditions.append(condition_idx)

                with open(progress_file, 'w') as f:
                    json.dump({
                        'completed_conditions': completed_conditions,
                        'last_updated': str(datetime.datetime.now()),
                        'last_condition_completed': condition_idx
                    }, f, indent=2)
            else:
                print(f"⚠️ Failed to reach target accuracy for {condition_name}. Best: {best_accuracy:.4f}")

        except Exception as e:
            print(f"❌ Error training {condition_name}: {str(e)}")
            print("Moving to next condition...")
            continue

        remaining = len(LABELS) - (condition_idx + 1)
        if remaining > 0:
            print(f"\n📋 Remaining conditions: {remaining}")
            print("Next up:", ', '.join(LABELS[condition_idx + 1:]))

        time.sleep(2)

    print("\n🎉 Training completed for all conditions!")
    print(f"Successfully completed {len(completed_conditions)} conditions")
    return completed_conditions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XightMD Lung Classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--samples-per-epoch', type=int, default=100, help='Samples per epoch')
    parser.add_argument('--val-samples', type=int, default=20, help='Validation samples')
    parser.add_argument('--checkpoint-freq', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--resume-checkpoint', type=str, help='Path to checkpoint file to resume training from')
    parser.add_argument('--verbose', action='store_true', help='Print detailed metrics')
    parser.add_argument('--condition-idx', type=int, help='Index of condition to start from or focus on')
    parser.add_argument('--early-stopping-patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--train-all', action='store_true',
                       help='Train all conditions sequentially starting from specified condition')
    parser.add_argument('--target-accuracy', type=float, default=0.8,
                       help='Target accuracy for each condition')

    args = parser.parse_args()

    if args.train_all:
        if args.condition_idx is not None:
            print(f"Starting sequential training from condition {args.condition_idx}: {LABELS[args.condition_idx]}")
        else:
            print("Starting sequential training from the beginning")
        completed_conditions = train_all_conditions(args)
    else:
        train_model(args)

