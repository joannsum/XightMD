import modal
from modal import Image, App, web_endpoint

# Add FastAPI to the image
image = (
    Image.debian_slim()
    .pip_install("fastapi[standard]", "pillow", "numpy")
)

app = App("xightmd-simple", image=image)

@app.function()
@web_endpoint(method="POST") 
def predict_lung_conditions(request_data: dict):
    """
    Simple prediction endpoint that returns mock data
    This gets your pipeline working while we add the real model later
    """
    try:
        # Mock predictions for all 14 conditions
        mock_predictions = {
            "Atelectasis": 0.15,
            "Cardiomegaly": 0.25, 
            "Consolidation": 0.10,
            "Edema": 0.05,
            "Effusion": 0.12,
            "Emphysema": 0.08,
            "Fibrosis": 0.06,
            "Hernia": 0.02,
            "Infiltration": 0.18,
            "Mass": 0.30,
            "Nodule": 0.22,
            "Pleural_Thickening": 0.09,
            "Pneumonia": 0.35,
            "Pneumothorax": 0.45
        }
        
        # Calculate confidence and urgency
        max_confidence = max(mock_predictions.values())
        urgency = 5 if max_confidence > 0.4 else 3 if max_confidence > 0.2 else 1
        
        return {
            "success": True,
            "predictions": mock_predictions,
            "confidence": max_confidence,
            "urgency": urgency,
            "processing_time_ms": 100,
            "model_type": "mock_classifier_v1"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "predictions": {},
            "confidence": 0.0,
            "urgency": 1
        }

@app.function()
@web_endpoint(method="GET")
def health():
    return {
        "status": "healthy", 
        "service": "xightmd-lung-classifier",
        "version": "1.0",
        "endpoints": ["predict_lung_conditions", "health"]
    }

# Optional: Info endpoint to see what conditions we detect
@app.function()
@web_endpoint(method="GET")
def info():
    return {
        "conditions": [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Effusion", "Emphysema", "Fibrosis", "Hernia", 
            "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
            "Pneumonia", "Pneumothorax"
        ],
        "model_info": "Mock classifier for pipeline testing",
        "input_format": {
            "image_data": "base64 encoded image",
            "image_format": "png/jpg/jpeg"
        }
    }