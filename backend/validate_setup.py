#!/usr/bin/env python3
import sys
import os

def validate_setup():
    print("🔍 Validating XightMD Setup...")
    print("="*50)
    
    # Check imports
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision import failed: {e}")
        return False
    
    try:
        from utils.lung_classifier import LungClassifierTrainer, LABELS
        print(f"✅ LungClassifier imported successfully")
        print(f"   - {len(LABELS)} disease labels configured")
    except ImportError as e:
        print(f"❌ LungClassifier import failed: {e}")
        return False
    
    # Test model creation
    try:
        classifier = LungClassifierTrainer()
        print(f"✅ Model created successfully")
        print(f"   - Device: {classifier.device}")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Check directories
    if os.path.exists("models"):
        print(f"✅ Models directory exists")
        if os.path.exists("models/lung_classifier.pth"):
            print(f"✅ Trained model found")
        else:
            print(f"⚠️  No trained model found (run train_model.py)")
    else:
        print(f"⚠️  Models directory missing")
    
    print("="*50)
    print("🎉 Setup validation complete!")
    return True

if __name__ == "__main__":
    validate_setup()