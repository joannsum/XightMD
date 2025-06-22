#!/usr/bin/env python3
from datasets import load_dataset

def check_dataset_format():
    print("🔍 Checking dataset format...")
    
    dataset = load_dataset("BahaaEldin0/NIH-Chest-Xray-14", streaming=True, trust_remote_code=True)
    
    # Get first sample
    train_iter = iter(dataset['train'])
    sample = next(train_iter)
    
    print("📋 Sample keys:", list(sample.keys()))
    print("\n📋 Sample data:")
    for key, value in sample.items():
        if key != 'image':  # Skip image data
            print(f"  {key}: {value} (type: {type(value)})")
    
    # Check a few more samples
    print("\n📋 Checking 5 more samples for label patterns...")
    for i in range(5):
        try:
            sample = next(train_iter)
            print(f"Sample {i+1} label: {sample['label']}")
        except:
            break

if __name__ == "__main__":
    check_dataset_format()