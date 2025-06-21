#!/usr/bin/env python3
import sys
import os
from utils.lung_classifier import LungClassifierTrainer
from PIL import Image
import numpy as np
import argparse

def create_dummy_xray():
    """Create a dummy grayscale image that looks like an X-ray"""
    # Create a 224x224 grayscale image with some chest-like patterns
    img_array = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
    
    # Add some circular patterns to simulate lungs
    center1, center2 = (80, 112), (144, 112)
    y, x = np.ogrid[:224, :224]
    
    # Create lung-like circular regions
    mask1 = (x - center1[0])**2 + (y - center1[1])**2 < 50**2
    mask2 = (x - center2[0])**2 + (y - center2[1])**2 < 50**2
    
    img_array[mask1] = np.random.randint(80, 150, np.sum(mask1))
    img_array[mask2] = np.random.randint(80, 150, np.sum(mask2))
    
    # Convert to RGB PIL image
    img = Image.fromarray(img_array, mode='L').convert('RGB')
    return img

def test_classifier():
    parser = argparse.ArgumentParser(description='Test lung classifier')
    parser.add_argument('--image', type=str, help='Path to X-ray image')
    parser.add_argument('--model', type=str, default='models/lung_classifier.pth', help='Model path')
    parser.add_argument('--dummy', action='store_true', help='Use dummy image for testing')
    
    args = parser.parse_args()
    
    # Initialize classifier
    print("Loading classifier...")
    model_path = args.model if os.path.exists(args.model) else None
    classifier = LungClassifierTrainer(model_path=model_path)
    
    if args.dummy:
        print("Creating dummy X-ray image for testing...")
        dummy_img = create_dummy_xray()
        dummy_img.save("dummy_xray.jpg")
        image_path = "dummy_xray.jpg"
        print("✓ Dummy image created: dummy_xray.jpg")
    else:
        image_path = args.image
    
    if not image_path or not os.path.exists(image_path):
        print("Please provide a valid image path or use --dummy")
        print("Usage examples:")
        print("  python test_classifier.py --dummy")
        print("  python test_classifier.py --image /path/to/xray.jpg")
        return
    
    print(f"Analyzing image: {image_path}")
    
    try:
        # Make prediction
        predictions = classifier.predict(image_path)
        significance = classifier.get_statistical_significance(predictions)
        
        # Display results
        print("\n" + "="*60)
        print("LUNG DISEASE CLASSIFICATION RESULTS")
        print("="*60)
        
        # Sort by confidence
        sorted_results = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        for condition, confidence in sorted_results:
            sig_data = significance[condition]
            status = "SIGNIFICANT" if sig_data['significant'] else "not significant"
            confidence_level = sig_data['confidence_level']
            
            # Add visual indicator
            indicator = "🔴" if sig_data['significant'] and confidence > 0.7 else "🟡" if sig_data['significant'] else "⚪"
            
            print(f"{indicator} {condition:<20} {confidence:.3f} ({confidence_level:>6}) - {status}")
        
        print("\n" + "="*60)
        print("HIGH CONFIDENCE FINDINGS:")
        high_conf = [cond for cond, data in significance.items() if data['significant'] and data['confidence'] > 0.7]
        if high_conf:
            for finding in high_conf:
                print(f"  🚨 {finding} (confidence: {predictions[finding]:.3f})")
        else:
            print("  ✅ No high-confidence pathologies detected")
        
        print("="*60)
        
        # Test model output
        print(f"\nModel Test Results:")
        print(f"  ✓ Image loaded successfully")
        print(f"  ✓ Model inference completed")
        print(f"  ✓ Statistical analysis completed")
        print(f"  ✓ All {len(predictions)} conditions evaluated")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_classifier()