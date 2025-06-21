# ... existing imports ...
from typing import Dict, Any
from datetime import datetime
import os
from PIL import Image
from utils.lung_classifier import LungClassifierTrainer, LABELS

class TriageAgent:
    def __init__(self):
        # ... existing initialization ...
        self.lung_classifier = LungClassifierTrainer()
        
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze X-ray image for lung conditions"""
        try:
            # Verify image exists and is readable
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Basic image validation
            try:
                with Image.open(image_path) as img:
                    img.verify()  # Verify it's a valid image
            except Exception as e:
                raise ValueError(f"Invalid image file: {e}")
            
            # Get predictions directly from your lung classifier
            predictions = self.lung_classifier.predict(image_path)
            
            # Calculate statistical significance
            significance = self.lung_classifier.get_statistical_significance(predictions)
            
            # Determine urgency based on conditions
            urgency_score = self.calculate_urgency(significance)
            
            return {
                'predictions': predictions,
                'statistical_significance': significance,
                'urgency_score': urgency_score,
                'high_confidence_findings': [
                    condition for condition, data in significance.items() 
                    if data['significant'] and data['confidence'] > 0.7
                ],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error in image analysis: {e}")
            return {'error': str(e)}
    
    # ... rest of the methods stay the same ...