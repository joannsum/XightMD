#!/usr/bin/env python3
import sys
import os
import argparse
from utils.lung_classifier import LungClassifierTrainer
from PIL import Image
import numpy as np
import torch

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
    parser.add_argument('--model', type=str, help='Specific model path to use')
    parser.add_argument('--dummy', action='store_true', help='Use dummy image for testing')
    parser.add_argument('--atelectasis', action='store_true', help='Use Atelectasis-specific model')
    parser.add_argument('--cardiomegaly', action='store_true', help='Use Cardiomegaly-specific model')

    args = parser.parse_args()
    
    # Determine model path - prioritize Atelectasis model ONLY if requested
    if args.atelectasis:
        model_path = "models/checkpoints/Atelectasis/best_model_Atelectasis.pth"
        if os.path.exists(model_path):
            print(f"🫁 Using ATELECTASIS-SPECIFIC model: {model_path}")
            print("🎯 This model is specialized for Atelectasis detection")
        else:
            print(f"❌ Atelectasis model not found: {model_path}")
            return
    if args.cardiomegaly:
        model_path = "models/checkpoints/Cardiomegaly/best_model_Cardiomegaly.pth"
        if os.path.exists(model_path):
            print(f"🫁 Using CARDIOMEGALY-SPECIFIC model: {model_path}")
            print("🎯 This model is specialized for Cardiomegaly detection")
        else:
            print(f"❌ Cardiomegaly model not found: {model_path}")
            return    

    elif args.model and os.path.exists(args.model):
        model_path = args.model
        print(f"Using specified model: {model_path}")
    else:
        # Default model search order - GENERAL models first
        model_candidates = [
            "models/lung_classifier_best.pth",                           # BEST general model
            "models/lung_classifier_trained.pth", 
            "models/lung_classifier_final.pth",
            "models/checkpoints/Atelectasis/best_model_Atelectasis.pth", # Atelectasis as fallback
            "models/checkpoints/Cardiomegaly/best_model_Cardiomegaly.pth", # Cardiomegaly as fallback
            "models/lung_classifier.pth"
        ]
        
        model_path = None
        for path in model_candidates:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            print(f"Found model: {model_path}")
            if "Atelectasis" in model_path:
                print("🫁 Using ATELECTASIS-SPECIFIC model (no general model found)!")
            if "Cardiomegaly" in model_path:
                print("🫁 Using CARDIOMEGALY-SPECIFIC model (no general model found)!")
            elif "best" in model_path:
                print("🏆 Using BEST GENERAL model from training!")
        else:
            print("⚠️  No trained model found, using pretrained weights only")
    
    # Initialize classifier
    print("Loading classifier...")
    try:
        classifier = LungClassifierTrainer(model_path=model_path)
        print("✅ Classifier loaded successfully")
    except Exception as e:
        print(f"❌ Error loading classifier: {e}")
        return
    
    # Determine image path
    if args.dummy:
        print("Creating dummy X-ray image for testing...")
        try:
            dummy_img = create_dummy_xray()
            dummy_img.save("dummy_xray.jpg")
            image_path = "dummy_xray.jpg"
            print("✅ Dummy image created: dummy_xray.jpg")
        except Exception as e:
            print(f"❌ Error creating dummy image: {e}")
            return
    else:
        image_path = args.image
    
    # Validate image path
    if not image_path:
        print("❌ No image provided!")
        print("Usage examples:")
        print("  python test_classifier.py --dummy")
        print("  python test_classifier.py --atelectasis --dummy")
        print("  python test_classifier.py --image /path/to/xray.jpg")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    print(f"Analyzing image: {image_path}")
    
    try:
        # Test image loading first
        test_img = Image.open(image_path)
        print(f"✅ Image loaded: {test_img.size}, mode: {test_img.mode}")
        
        # Make prediction
        print("Making prediction...")
        predictions = classifier.predict(image_path)
        print("✅ Prediction completed")
        
        # Get significance analysis
        print("Calculating statistical significance...")
        significance = classifier.get_statistical_significance(predictions)
        print("✅ Statistical analysis completed")
        
        # Special handling for Atelectasis model
        if "Atelectasis" in str(model_path):
            print(f"\n🫁 ATELECTASIS-SPECIFIC ANALYSIS")
            print("="*60)
            
            atelectasis_confidence = predictions.get('Atelectasis', 0.0)
            atelectasis_sig = significance.get('Atelectasis', {})
            
            if atelectasis_confidence > 0.16:
                print(f"🔴 ATELECTASIS DETECTED!")
                print(f"   Confidence: {atelectasis_confidence:.3f} ({atelectasis_sig.get('confidence_level', 'Unknown')})")
                print(f"   Status: {'SIGNIFICANT' if atelectasis_sig.get('significant', False) else 'not significant'}")
                print(f"   📋 Atelectasis is partial collapse of lung tissue")
                print(f"   📋 Requires medical evaluation and possible intervention")
            elif atelectasis_confidence > 0.10:
                print(f"🟡 POSSIBLE ATELECTASIS")
                print(f"   Confidence: {atelectasis_confidence:.3f} ({atelectasis_sig.get('confidence_level', 'Unknown')})")
                print(f"   📋 Borderline detection - recommend clinical correlation")
            else:
                print(f"🟢 NO ATELECTASIS DETECTED")
                print(f"   Confidence: {atelectasis_confidence:.3f}")
                print(f"   📋 Lung expansion appears normal")
            
            print("\n" + "="*60)
        
        # Check for critical conditions first
        critical_findings = []
        for condition in ['Pneumothorax', 'Mass', 'Pneumonia', 'Atelectasis']:
            if condition in predictions and predictions[condition] > 0.3:
                critical_findings.append((condition, predictions[condition]))
        
        if critical_findings:
            print(f"\n🚨 NOTABLE FINDINGS:")
            for condition, confidence in critical_findings:
                urgency = ""
                if condition == 'Pneumothorax':
                    urgency = " ⚠️  IMMEDIATE ATTENTION!"
                elif condition == 'Atelectasis':
                    urgency = " 📋 Requires evaluation"
                elif condition in ['Mass', 'Pneumonia']:
                    urgency = " 🔬 Further workup needed"
                    
                print(f"   🚨 {condition}: {confidence:.3f} confidence{urgency}")
        
        # Display detailed results
        print("\n" + "="*60)
        print("COMPLETE CLASSIFICATION RESULTS")
        print("="*60)
        
        # Sort by confidence
        sorted_results = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'Condition':<20} {'Confidence':<10} {'Level':<8} {'Status'}")
        print("-" * 60)
        
        for condition, confidence in sorted_results:
            sig_data = significance[condition]
            status = "POSITIVE" if sig_data['significant'] else "negative"
            confidence_level = sig_data['confidence_level']
            
            # Special highlighting for Atelectasis
            if condition == 'Atelectasis':
                if confidence > 0.16:
                    indicator = "🔴"
                elif confidence > 0.10:
                    indicator = "🟡"
                else:
                    indicator = "🟢"
            else:
                # Visual indicators for other conditions
                if sig_data['significant'] and confidence > 0.7:
                    indicator = "🔴"
                elif sig_data['significant'] and confidence > 0.5:
                    indicator = "🟡"
                elif confidence > 0.3:
                    indicator = "🟠"
                else:
                    indicator = "🟢"
            
            print(f"{indicator} {condition:<18} {confidence:.3f}      {confidence_level:<8} {status}")
        
        # Summary of significant findings
        print("\n" + "="*60)
        print("SIGNIFICANT FINDINGS SUMMARY:")
        significant_findings = [(cond, data) for cond, data in significance.items() if data['significant']]
        
        if significant_findings:
            for condition, data in significant_findings:
                confidence = predictions[condition]
                clinical_note = ""
                if condition == 'Atelectasis':
                    clinical_note = " (Lung collapse/consolidation)"
                print(f"  • {condition}: {confidence:.1%} confidence ({data['confidence_level']}){clinical_note}")
        else:
            print("  ✅ No significant pathologies detected")
        
        print("="*60)
        
        # Technical summary
        model_type = "Atelectasis-Specific" if "Atelectasis" in str(model_path) else "General Multi-Class"
        print(f"\nTechnical Summary:")
        print(f"  ✓ Model Type: {model_type}")
        print(f"  ✓ Model Path: {model_path or 'pretrained only'}")
        print(f"  ✓ Image: {image_path}")
        print(f"  ✓ Conditions evaluated: {len(predictions)}")
        print(f"  ✓ Significant findings: {len(significant_findings)}")
        print(f"  ✓ Highest confidence: {max(predictions.values()):.3f}")
        
        if "Atelectasis" in str(model_path):
            atelectasis_conf = predictions.get('Atelectasis', 0.0)
            print(f"  🫁 Atelectasis confidence: {atelectasis_conf:.3f}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_classifier()
