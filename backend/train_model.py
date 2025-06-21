import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from utils.lung_classifier import LungClassifierTrainer, LungDiseaseClassifier
import argparse

def train_model():
    """Train the lung disease classifier"""
    print("Loading ReXGradient dataset...")
    # Make sure you've run: huggingface-cli login
    dataset = load_dataset("rajpurkarlab/ReXGradient-160K")
    
    trainer = LungClassifierTrainer()
    
    # Training loop would go here
    # For hackathon, you might want to use a pre-trained model or fine-tune
    
    print("Training completed!")
    trainer.save_model("models/lung_classifier.pth")

if __name__ == "__main__":
    train_model()