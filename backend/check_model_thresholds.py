#!/usr/bin/env python3
import torch

def check_model_info():
    checkpoint_path = "models/checkpoints/Atelectasis/best_model_Atelectasis.pth"
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("🔍 MODEL CHECKPOINT ANALYSIS")
        print("="*50)
        
        for key, value in checkpoint.items():
            if key != 'model_state_dict':
                print(f"{key}: {value}")
        
        # Check if there are validation metrics
        if 'val_metrics' in checkpoint:
            print("\n📊 VALIDATION METRICS:")
            for metric, value in checkpoint['val_metrics'].items():
                print(f"  {metric}: {value}")
        
        if 'training_history' in checkpoint:
            print(f"\n📈 Training epochs: {len(checkpoint['training_history'])}")
            
        print("\n💡 The 90% accuracy might have used a different threshold!")
        print("💡 Try using threshold 0.2-0.3 instead of 0.5")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_model_info()