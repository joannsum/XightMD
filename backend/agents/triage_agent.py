from uagents import Agent, Context, Model
from typing import Dict, Any, List
from datetime import datetime
import os
import json
import asyncio
import aiohttp
from PIL import Image
import base64
import io

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
        # Updated Modal endpoint - use the actual predict endpoint
        self.modal_endpoint = "https://joannsum--xightmd-simple-predict-lung-conditions.modal.run"
        
        self.agent = Agent(
            name="triage_agent",
            port=8001,
            seed="triage_agent_seed_123",
            endpoint=["http://localhost:8001/submit"]
        )
        
        # Define supported labels (since we removed local classifier)
        self.labels = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Effusion", "Emphysema", "Fibrosis", "Hernia",
            "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
            "Pneumonia", "Pneumothorax"
        ]
        
        # 🔧 IMPROVED DETECTION THRESHOLDS - This is where you make it suck less!
        self.detection_thresholds = {
            # Critical conditions - MUCH lower thresholds for better detection
            "Pneumothorax": 0.10,      # Super sensitive for collapsed lung
            "Pneumonia": 0.15,         # More sensitive for pneumonia
            "Mass": 0.12,              # More sensitive for masses
            "Consolidation": 0.20,     # Lower threshold for consolidation
            "Effusion": 0.18,          # Lower threshold for effusions
            
            # Other important conditions
            "Cardiomegaly": 0.25,      # Heart enlargement
            "Atelectasis": 0.30,       # Lung collapse
            "Edema": 0.25,             # Fluid accumulation
            "Emphysema": 0.30,         # Lung damage
            "Fibrosis": 0.30,          # Scarring
            "Infiltration": 0.28,      # Substance accumulation
            "Nodule": 0.20,            # Small growths
            "Pleural_Thickening": 0.35, # Pleural changes
            "Hernia": 0.40,            # Displacement
            
            # Default threshold for any condition not listed above
            "default": 0.25
        }
        
        # Define critical conditions that require immediate attention
        self.critical_conditions = [
            "Pneumothorax",  # Collapsed lung - emergency
            "Pneumonia",     # Infection requiring treatment
            "Effusion",      # Fluid in lungs (updated from "Pleural Effusion")
            "Mass",          # Potential tumor
            "Consolidation"  # Lung tissue filling
        ]
        
        # 🚨 URGENCY CALCULATION IMPROVEMENTS
        self.urgency_thresholds = {
            "critical": {
                "very_high": 0.6,    # Confidence > 60% = urgency 5
                "high": 0.4,         # Confidence > 40% = urgency 4  
                "moderate": 0.25,    # Confidence > 25% = urgency 3
                "low": 0.15          # Confidence > 15% = urgency 2
            },
            "non_critical": {
                "high": 0.5,         # Confidence > 50% = urgency 3
                "moderate": 0.3,     # Confidence > 30% = urgency 2
                "low": 0.2           # Confidence > 20% = urgency 2
            }
        }
        
        print("🔗 Triage Agent configured to use Modal service")
        print(f"📡 Modal endpoint: {self.modal_endpoint}")
        print(f"🔧 Using improved detection thresholds")
        
        # Setup agent handlers
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup agent message handlers"""
        
        @self.agent.on_message(model=ImageAnalysisRequest)
        async def handle_image_analysis(ctx: Context, sender: str, msg: ImageAnalysisRequest):
            """Handle incoming image analysis requests"""
            ctx.logger.info(f"Received image analysis request: {msg.request_id}")
            
            try:
                # Process the image using Modal service
                result = await self.analyze_image_with_modal(
                    msg.image_data, 
                    msg.image_format,
                    msg.request_id
                )
                
                # Send response back to coordinator
                await ctx.send(sender, result)
                
            except Exception as e:
                ctx.logger.error(f"Error in image analysis: {e}")
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

    async def analyze_image_with_modal(self, image_data: str, image_format: str, request_id: str) -> ImageAnalysisResponse:
        """Analyze X-ray image using Modal service"""
        start_time = datetime.now()
        
        try:
            # Call Modal service
            async with aiohttp.ClientSession() as session:
                payload = {
                    "image_data": image_data,
                    "image_format": image_format
                }
                
                print(f"🚀 Calling Modal service for request {request_id}")
                
                async with session.post(self.modal_endpoint, json=payload, timeout=30) as response:
                    if response.status == 200:
                        modal_result = await response.json()
                        
                        if modal_result.get("success", False):
                            # Extract predictions from Modal response
                            predictions = modal_result["predictions"]
                            
                            # 🔧 USE OUR IMPROVED ANALYSIS instead of Modal's urgency/confidence
                            significance = self.create_improved_significance_from_predictions(predictions)
                            
                            # Calculate improved urgency and confidence
                            urgency_score = self.calculate_improved_urgency(significance)
                            confidence_score = self.calculate_improved_confidence(predictions, significance)
                            
                            # Use our improved logic for critical findings
                            critical_findings = self.identify_improved_critical_findings(significance)
                            all_findings = self.format_improved_findings(significance)
                            
                            processing_time = (datetime.now() - start_time).total_seconds() * 1000
                            
                            print(f"✅ Modal analysis completed for {request_id}")
                            print(f"🔍 Found {len(critical_findings)} critical findings")
                            print(f"📊 Urgency: {urgency_score}, Confidence: {confidence_score:.2f}")
                            
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
                        else:
                            raise Exception(f"Modal service error: {modal_result.get('error', 'Unknown error')}")
                    else:
                        raise Exception(f"Modal service returned status {response.status}")
                        
        except asyncio.TimeoutError:
            raise Exception("Modal service timeout")
        except Exception as e:
            print(f"❌ Modal service call failed: {e}")
            # Return error response
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
                error=f"Modal service error: {str(e)}"
            )

    def create_improved_significance_from_predictions(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """Convert Modal predictions to significance format with improved thresholds"""
        significance = {}
        
        for condition, confidence in predictions.items():
            # Get the threshold for this specific condition
            threshold = self.detection_thresholds.get(condition, self.detection_thresholds["default"])
            
            # 🔧 Use improved threshold logic
            is_significant = confidence > threshold
            
            # More granular confidence levels
            if confidence > 0.7:
                confidence_level = "very high"
            elif confidence > 0.5:
                confidence_level = "high"
            elif confidence > threshold:
                confidence_level = "moderate"
            elif confidence > threshold * 0.7:
                confidence_level = "low"
            else:
                confidence_level = "very low"
            
            significance[condition] = {
                'significant': is_significant,
                'confidence': confidence,
                'confidence_level': confidence_level,
                'threshold_used': threshold,
                'above_threshold_margin': confidence - threshold
            }
        
        return significance

    def calculate_improved_urgency(self, significance: Dict[str, Any]) -> int:
        """Calculate urgency score with improved sensitivity"""
        max_urgency = 1
        
        for condition, data in significance.items():
            if not data['significant']:
                continue
                
            confidence = data['confidence']
            
            # Critical conditions requiring immediate attention
            if condition in self.critical_conditions:
                thresholds = self.urgency_thresholds["critical"]
                
                if confidence > thresholds["very_high"]:
                    max_urgency = max(max_urgency, 5)  # Critical
                elif confidence > thresholds["high"]:
                    max_urgency = max(max_urgency, 4)  # High priority  
                elif confidence > thresholds["moderate"]:
                    max_urgency = max(max_urgency, 3)  # Medium priority
                elif confidence > thresholds["low"]:
                    max_urgency = max(max_urgency, 2)  # Low priority
            
            # Other significant findings
            else:
                thresholds = self.urgency_thresholds["non_critical"]
                
                if confidence > thresholds["high"]:
                    max_urgency = max(max_urgency, 3)  # Medium priority
                elif confidence > thresholds["moderate"]:
                    max_urgency = max(max_urgency, 2)  # Low priority
                elif confidence > thresholds["low"]:
                    max_urgency = max(max_urgency, 2)  # Low priority
                
        return max_urgency

    def calculate_improved_confidence(self, predictions: Dict[str, float], significance: Dict[str, Any]) -> float:
        """Calculate overall confidence with improved logic"""
        if not predictions:
            return 0.0
        
        # Get significant findings
        significant_findings = [
            (condition, data['confidence']) 
            for condition, data in significance.items() 
            if data['significant']
        ]
        
        if not significant_findings:
            # Even if no findings are significant, provide base confidence
            max_prediction = max(predictions.values()) if predictions else 0
            return min(max_prediction * 0.8, 0.7)  # Base confidence
        
        # Weight critical conditions more heavily
        weighted_confidences = []
        for condition, conf in significant_findings:
            weight = 1.2 if condition in self.critical_conditions else 1.0
            
            # Apply high confidence bonus
            if conf > 0.7:
                weight *= 1.15
            
            weighted_confidences.append(conf * weight)
        
        # Calculate weighted average
        avg_confidence = sum(weighted_confidences) / len(weighted_confidences)
        
        # Apply bounds (min 0.5 if we found significant findings, max 0.95)
        final_confidence = max(0.5, min(avg_confidence, 0.95))
        
        return final_confidence

    def identify_improved_critical_findings(self, significance: Dict[str, Any]) -> List[str]:
        """Identify critical findings with improved sensitivity"""
        critical = []
        
        for condition, data in significance.items():
            if (data['significant'] and 
                condition in self.critical_conditions and 
                data['confidence'] > 0.1):  # Very low threshold for critical conditions!
                
                # Add detailed information
                margin = data['above_threshold_margin']
                critical.append(
                    f"{condition} (confidence: {data['confidence']:.1%}, "
                    f"threshold: {data['threshold_used']:.1%}, "
                    f"margin: +{margin:.1%})"
                )
                
        return critical

    def format_improved_findings(self, significance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format all significant findings with additional metadata"""
        findings = []
        
        for condition, data in significance.items():
            if data['significant']:
                findings.append({
                    'condition': condition,
                    'confidence': data['confidence'],
                    'confidence_level': data['confidence_level'],
                    'critical': condition in self.critical_conditions,
                    'threshold_used': data['threshold_used'],
                    'margin_above_threshold': data['above_threshold_margin'],
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
            "Pleural_Thickening": "Thickening of lung lining",
            "Pneumonia": "Infection causing inflammation in lungs",
            "Pneumothorax": "Collapsed lung due to air in pleural space"
        }
        return descriptions.get(condition, "Medical condition requiring evaluation")

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'name': 'Triage Agent (Modal-Enhanced)',
            'status': 'active',
            'address': str(self.agent.address),
            'modal_endpoint': self.modal_endpoint,
            'capabilities': [
                'Chest X-ray analysis via Modal',
                'Improved lung disease detection',
                'Enhanced urgency assessment',
                'Critical findings identification'
            ],
            'supported_conditions': self.labels,
            'critical_conditions': self.critical_conditions,
            'detection_thresholds': self.detection_thresholds,
            'urgency_thresholds': self.urgency_thresholds,
            'last_updated': datetime.now().isoformat()
        }

    def run(self):
        """Start the agent"""
        print(f"🔍 Starting Enhanced Triage Agent with Modal...")
        print(f"📍 Agent address: {self.agent.address}")
        print(f"🔗 Modal endpoint: {self.modal_endpoint}")
        print(f"🫁 Supported conditions: {self.labels}")
        print(f"🔧 Detection thresholds: {self.detection_thresholds}")
        print("=" * 50)
        self.agent.run()

if __name__ == "__main__":
    # Initialize and run the triage agent
    triage_agent = TriageAgent()
    triage_agent.run()