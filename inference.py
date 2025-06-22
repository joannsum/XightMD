import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import json
import sys
from backend.utils.lung_classifier import LungClassifierTrainer, LABELS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to the model weights')
    parser.add_argument('--image_path', required=True, help='Path to the input image')
    args = parser.parse_args()

    try:
        classifier = LungClassifierTrainer(model_path=args.model_path)
        predictions = classifier.predict(args.image_path)
        significant_findings = classifier.get_statistical_significance(predictions)
        critical_conditions = ['Pneumothorax', 'Pneumonia', 'Mass', 'Edema']

        max_critical_confidence = max(predictions[cond] for cond in critical_conditions)
        urgency_level = 1
        if max_critical_confidence > 0.6:
            urgency_level = 5
        elif max_critical_confidence > 0.4:
            urgency_level = 4
        elif max_critical_confidence > 0.25:
            urgency_level = 3
        elif max_critical_confidence > 0.1:
            urgency_level = 2

        sorted_findings = sorted(
            [(label, data) for label, data in significant_findings.items() if data['significant']],
            key=lambda x: x[1]['confidence'],
            reverse=True
        )
        result = {
            "predictions": predictions,
            "significant_findings": significant_findings,
            "urgency_level": urgency_level,
            "top_findings": [
                {
                    "condition": label,
                    "confidence": data['confidence'],
                    "confidence_level": data['confidence_level']
                }
                for label, data in sorted_findings[:5]
            ],
            "has_critical_findings": any(
                significant_findings[cond]['significant']
                for cond in critical_conditions
            )
        }
        
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({
            "error": f"Inference failed: {str(e)}"
        }), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()