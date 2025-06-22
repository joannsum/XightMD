from uagents import Agent, Context, Model
from typing import Dict, Any, List
from datetime import datetime
import os
import json
import asyncio
from PIL import Image
import base64
import io

# Import your lung classifier
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.lung_classifier import LungClassifierTrainer, LABELS

# Message models for agent communication
class ImageAnalysisRequest(Model):
    image_data: str  # base64 encoded image
    image_format: str
    request_id: str
    timestamp: str

class ImageAnalysisResponse(Model):
    request_id: str
    predictions: Dict[str, float]
    urgency_score: int
    confidence_score: float
    critical_findings: List[str]
    all_findings: List[Dict[str, Any]]
    statistical_significance: Dict[str, Any]
    processing_time_ms: int
    timestamp: str
    error: str = None

class TriageAgent:
    def __init__(self, agent_address: str = "triage_agent"):
        # REMOVED enable_wallet parameter
        self.agent = Agent(
            name="triage_agent",
            port=8001,
            seed="triage_agent_seed_123",
            endpoint=["http://localhost:8001/submit"]
        )
        
        # Initialize the lung classifier
        print("Initializing lung disease classifier...")
        self.lung_classifier = LungClassifierTrainer()
        
        # Define critical conditions that require immediate attention
        self.critical_conditions = [
            "Pneumothorax",  # Collapsed lung - emergency
            "Pneumonia",     # Infection requiring treatment
            "Pleural Effusion",  # Fluid in lungs
            "Mass",          # Potential tumor
            "Consolidation"  # Lung tissue filling
        ]
        
        # Setup agent handlers
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup agent message handlers"""
        
        @self.agent.on_message(model=ImageAnalysisRequest)
        async def handle_image_analysis(ctx: Context, sender: str, msg: ImageAnalysisRequest):
            """Handle incoming image analysis requests"""
            ctx.logger.info(f"Received image analysis request: {msg.request_id}")
            
            try:
                # Process the image
                result = await self.analyze_image_from_base64(
                    msg.image_data, 
                    msg.image_format,
                    msg.request_id
                )
                
                # Send response back to coordinator
                await ctx.send(sender, result)
                
            except Exception as e:
                error_response = ImageAnalysisResponse(
                    request_id=msg.request_id,
                    predictions={},
                    urgency_score=0,
                    confidence_score=0.0,
                    critical_findings=[],
                    all_findings=[],
                    statistical_significance={},
                    processing_time_ms=0,
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
                await ctx.send(sender, error_response)

    async def analyze_image_from_base64(self, image_data: str, image_format: str, request_id: str) -> ImageAnalysisResponse:
        """Analyze X-ray image from base64 data"""
        start_time = datetime.now()
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Save temporarily for processing
            temp_path = f"/tmp/temp_xray_{request_id}.jpg"
            image.save(temp_path)
            
            try:
                # Get predictions from lung classifier
                predictions = self.lung_classifier.predict(temp_path)
                
                # Calculate statistical significance
                significance = self.lung_classifier.get_statistical_significance(predictions)
                
                # Determine urgency and confidence
                urgency_score = self.calculate_urgency(significance)
                confidence_score = self.calculate_overall_confidence(predictions)
                
                # Identify critical findings
                critical_findings = self.identify_critical_findings(significance)
                
                # Format all findings
                all_findings = self.format_findings(significance)
                
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return ImageAnalysisResponse(
                    request_id=request_id,
                    predictions=predictions,
                    urgency_score=urgency_score,
                    confidence_score=confidence_score,
                    critical_findings=critical_findings,
                    all_findings=all_findings,
                    statistical_significance=significance,
                    processing_time_ms=int(processing_time),
                    timestamp=datetime.now().isoformat()
                )
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return ImageAnalysisResponse(
                request_id=request_id,
                predictions={},
                urgency_score=0,
                confidence_score=0.0,
                critical_findings=[],
                all_findings=[],
                statistical_significance={},
                processing_time_ms=int(processing_time),
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )

    def calculate_urgency(self, significance: Dict[str, Any]) -> int:
        """Calculate urgency score from 1 (normal) to 5 (critical emergency)"""
        max_urgency = 1
        
        for condition, data in significance.items():
            if not data['significant']:
                continue
                
            confidence = data['confidence']
            
            # Critical conditions requiring immediate attention
            if condition in self.critical_conditions:
                if confidence > 0.8:
                    max_urgency = max(max_urgency, 5)  # Critical
                elif confidence > 0.6:
                    max_urgency = max(max_urgency, 4)  # High priority
                else:
                    max_urgency = max(max_urgency, 3)  # Medium priority
            
            # Other significant findings
            elif confidence > 0.7:
                max_urgency = max(max_urgency, 3)  # Medium priority
            elif confidence > 0.5:
                max_urgency = max(max_urgency, 2)  # Low priority
                
        return max_urgency

    def calculate_overall_confidence(self, predictions: Dict[str, float]) -> float:
        """Calculate overall confidence in the analysis"""
        if not predictions:
            return 0.0
            
        # Get the top 3 predictions
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_predictions = sorted_predictions[:3]
        
        # Weight higher predictions more heavily
        weighted_sum = sum(conf * (4-i) for i, (_, conf) in enumerate(top_predictions))
        total_weight = sum(4-i for i in range(len(top_predictions)))
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def identify_critical_findings(self, significance: Dict[str, Any]) -> List[str]:
        """Identify critical findings requiring immediate attention"""
        critical = []
        
        for condition, data in significance.items():
            if (data['significant'] and 
                condition in self.critical_conditions and 
                data['confidence'] > 0.6):
                critical.append(f"{condition} (confidence: {data['confidence']:.1%})")
                
        return critical

    def format_findings(self, significance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format all significant findings for display"""
        findings = []
        
        for condition, data in significance.items():
            if data['significant']:
                findings.append({
                    'condition': condition,
                    'confidence': data['confidence'],
                    'confidence_level': data['confidence_level'],
                    'critical': condition in self.critical_conditions,
                    'description': self.get_condition_description(condition)
                })
        
        # Sort by confidence
        findings.sort(key=lambda x: x['confidence'], reverse=True)
        return findings

    def get_condition_description(self, condition: str) -> str:
        """Get human-readable description of medical condition"""
        descriptions = {
            "Atelectasis": "Collapse or closure of lung tissue",
            "Cardiomegaly": "Enlarged heart",
            "Consolidation": "Lung tissue filled with liquid instead of air",
            "Edema": "Fluid accumulation in lung tissue",
            "Effusion": "Fluid accumulation in pleural space",
            "Emphysema": "Damage to air sacs in lungs",
            "Fibrosis": "Thickening and scarring of lung tissue",
            "Hernia": "Organ displacement through weakness in surrounding tissue",
            "Infiltration": "Substance accumulation in lung tissue",
            "Mass": "Abnormal growth or tumor",
            "Nodule": "Small round growth in lung",
            "Pleural Thickening": "Thickening of lung lining",
            "Pneumonia": "Infection causing inflammation in lungs",
            "Pneumothorax": "Collapsed lung due to air in pleural space"
        }
        return descriptions.get(condition, "Medical condition requiring evaluation")

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'name': 'Triage Agent',
            'status': 'active',
            'address': str(self.agent.address),
            'capabilities': [
                'Chest X-ray analysis',
                'Lung disease detection',
                'Urgency assessment',
                'Statistical significance calculation'
            ],
            'supported_conditions': LABELS,
            'critical_conditions': self.critical_conditions,
            'last_updated': datetime.now().isoformat()
        }

    def run(self):
        """Start the agent"""
        print(f"🔍 Starting Triage Agent...")
        print(f"📍 Agent address: {self.agent.address}")
        print(f"🫁 Supported conditions: {LABELS}")
        print("=" * 50)
        self.agent.run()

if __name__ == "__main__":
    # Initialize and run the triage agent
    triage_agent = TriageAgent()
    triage_agent.run()