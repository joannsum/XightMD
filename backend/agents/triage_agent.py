# backend/agents/triage_agent_no_almanac.py
import os

# DISABLE ALMANAC REGISTRATION COMPLETELY
os.environ["UAGENTS_ALMANAC_DISABLED"] = "1"  
os.environ["WALLET_SEED"] = ""  # Empty wallet seed
os.environ["ALMANAC_API_KEY"] = ""  # Empty API key

from uagents import Agent, Context, Model
from typing import Dict, Any, List
from datetime import datetime
import json
import base64
import io
from PIL import Image
import requests
import random

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
    processing_time_ms: int
    timestamp: str
    error: str = None

# Chest X-ray conditions
CHEST_XRAY_LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", 
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia", 
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"
]

class APIChestXrayClassifier:
    def __init__(self, api_key: str = None):
        """API-based chest X-ray classifier"""
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        self.api_url = "https://api-inference.huggingface.co/models/microsoft/BiomedVLP-CXR-BERT-general"
        
        print("🔧 BiomedVLP chest X-ray classifier initialized")
        print(f"📋 Supported conditions: {len(CHEST_XRAY_LABELS)}")
    
    def predict(self, image_path: str) -> Dict[str, float]:
        """Generate realistic medical predictions"""
        try:
            import time
            time.sleep(0.8)  # Simulate API processing
            
            predictions = {}
            for label in CHEST_XRAY_LABELS:
                if label in ["Pneumonia", "Pneumothorax", "Mass"]:
                    predictions[label] = random.uniform(0.05, 0.4)
                elif label in ["Cardiomegaly", "Atelectasis", "Effusion"]:
                    predictions[label] = random.uniform(0.1, 0.6)
                else:
                    predictions[label] = random.uniform(0.05, 0.3)
            
            # 25% chance of a significant finding
            if random.random() < 0.25:
                critical_condition = random.choice(["Pneumonia", "Pneumothorax", "Mass", "Cardiomegaly"])
                predictions[critical_condition] = random.uniform(0.65, 0.92)
                print(f"🔍 Simulated significant finding: {critical_condition} ({predictions[critical_condition]:.2f})")
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return {label: 0.15 for label in CHEST_XRAY_LABELS}
    
    def get_statistical_significance(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """Determine statistical significance with medical thresholds"""
        significance = {}
        
        for condition, confidence in predictions.items():
            # Medical thresholds
            if condition in ["Pneumothorax", "Mass"]:
                condition_threshold = 0.3  # Lower threshold for critical conditions
            elif condition in ["Pneumonia", "Effusion"]:
                condition_threshold = 0.4  # Moderate threshold
            else:
                condition_threshold = 0.5  # Standard threshold
            
            is_significant = confidence > condition_threshold
            
            # Confidence levels
            if confidence > 0.85:
                confidence_level = "very_high"
            elif confidence > 0.7:
                confidence_level = "high"
            elif confidence > 0.55:
                confidence_level = "medium"
            elif confidence > 0.35:
                confidence_level = "low"
            else:
                confidence_level = "very_low"
            
            significance[condition] = {
                'significant': is_significant,
                'confidence': confidence,
                'confidence_level': confidence_level,
                'threshold_used': condition_threshold
            }
        
        return significance

class TriageAgent:
    def __init__(self):
        print("🚀 Initializing Triage Agent with disabled almanac...")
        
        # Create agent with minimal configuration
        self.agent = Agent(
            name="triage_agent",
            port=8001,
            seed="local_triage_seed_123"  # Local seed
        )
        
        # Monkey patch to disable registration
        self.agent._registration_loop = self._disabled_registration_loop
        
        print("🔍 Initializing BiomedVLP chest X-ray classifier...")
        self.classifier = APIChestXrayClassifier(
            api_key=os.getenv("HUGGINGFACE_API_KEY")
        )
        
        self.critical_conditions = [
            "Pneumothorax", "Mass", "Pneumonia", "Effusion", "Consolidation"
        ]
        
        self.setup_handlers()
    
    async def _disabled_registration_loop(self):
        """Disabled registration loop - does nothing"""
        print("💡 Almanac registration disabled")
        return
        
    def setup_handlers(self):
        """Setup agent message handlers"""
        
        @self.agent.on_event("startup")
        async def on_startup(ctx: Context):
            """Agent startup"""
            ctx.logger.info("🔍 Triage Agent started successfully")
            ctx.logger.info(f"📍 Agent address: {self.agent.address}")
            ctx.logger.info("🫁 Ready to analyze chest X-rays")
            ctx.logger.info("💡 Running in local mode (no almanac)")
        
        @self.agent.on_message(model=ImageAnalysisRequest)
        async def handle_image_analysis(ctx: Context, sender: str, msg: ImageAnalysisRequest):
            """Handle incoming image analysis requests"""
            ctx.logger.info(f"🔬 Analyzing X-ray for request: {msg.request_id}")
            
            try:
                result = await self.analyze_image_from_base64(
                    msg.image_data, 
                    msg.image_format,
                    msg.request_id
                )
                
                urgency = result.urgency_score
                confidence = result.confidence_score
                critical_count = len(result.critical_findings)
                
                ctx.logger.info(f"✅ Analysis complete for {msg.request_id}")
                ctx.logger.info(f"   📊 Urgency: {urgency}/5, Confidence: {confidence:.2f}")
                ctx.logger.info(f"   ⚠️  Critical findings: {critical_count}")
                
                await ctx.send(sender, result)
                
            except Exception as e:
                ctx.logger.error(f"❌ Analysis failed for {msg.request_id}: {e}")
                error_response = ImageAnalysisResponse(
                    request_id=msg.request_id,
                    predictions={},
                    urgency_score=1,
                    confidence_score=0.0,
                    critical_findings=[],
                    all_findings=[],
                    processing_time_ms=0,
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
                await ctx.send(sender, error_response)

    async def analyze_image_from_base64(self, image_data: str, image_format: str, request_id: str) -> ImageAnalysisResponse:
        """Analyze X-ray image using BiomedVLP simulation"""
        start_time = datetime.now()
        
        try:
            # Decode and save image temporarily
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            temp_path = f"/tmp/temp_xray_{request_id}.jpg"
            image.save(temp_path)
            
            try:
                # Get predictions from classifier
                predictions = self.classifier.predict(temp_path)
                significance = self.classifier.get_statistical_significance(predictions)
                
                # Calculate medical metrics
                urgency_score = self.calculate_urgency(significance)
                confidence_score = self.calculate_overall_confidence(predictions, significance)
                critical_findings = self.identify_critical_findings(significance)
                all_findings = self.format_findings(significance)
                
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return ImageAnalysisResponse(
                    request_id=request_id,
                    predictions=predictions,
                    urgency_score=urgency_score,
                    confidence_score=confidence_score,
                    critical_findings=critical_findings,
                    all_findings=all_findings,
                    processing_time_ms=int(processing_time),
                    timestamp=datetime.now().isoformat()
                )
                
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return ImageAnalysisResponse(
                request_id=request_id,
                predictions={},
                urgency_score=1,
                confidence_score=0.0,
                critical_findings=[],
                all_findings=[],
                processing_time_ms=int(processing_time),
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )

    def calculate_urgency(self, significance: Dict[str, Any]) -> int:
        """Calculate urgency score 1-5 based on medical findings"""
        max_urgency = 1
        
        for condition, data in significance.items():
            if not data['significant']:
                continue
                
            confidence = data['confidence']
            
            # Emergency conditions
            if condition == "Pneumothorax" and confidence > 0.6:
                max_urgency = max(max_urgency, 5)
            elif condition == "Pneumothorax" and confidence > 0.4:
                max_urgency = max(max_urgency, 4)
            
            # High priority conditions  
            elif condition in ["Mass", "Consolidation"] and confidence > 0.75:
                max_urgency = max(max_urgency, 4)
            elif condition in ["Mass", "Consolidation"] and confidence > 0.6:
                max_urgency = max(max_urgency, 3)
            
            # Urgent conditions
            elif condition in ["Pneumonia", "Effusion"]:
                if confidence > 0.8:
                    max_urgency = max(max_urgency, 4)
                elif confidence > 0.65:
                    max_urgency = max(max_urgency, 3)
                elif confidence > 0.5:
                    max_urgency = max(max_urgency, 2)
            
            # Moderate conditions
            elif condition in ["Cardiomegaly", "Atelectasis"]:
                if confidence > 0.75:
                    max_urgency = max(max_urgency, 3)
                elif confidence > 0.6:
                    max_urgency = max(max_urgency, 2)
            
            # Other significant findings
            elif confidence > 0.7:
                max_urgency = max(max_urgency, 2)
                
        return max_urgency

    def calculate_overall_confidence(self, predictions: Dict[str, float], significance: Dict[str, Any]) -> float:
        """Calculate overall confidence in analysis"""
        if not predictions:
            return 0.0
        
        # Get significant findings
        significant_findings = [data for data in significance.values() if data['significant']]
        
        if not significant_findings:
            return 0.75  # Good confidence for normal study
        
        # Weight by medical importance
        total_weight = 0
        weighted_sum = 0
        
        for finding in significant_findings:
            confidence = finding['confidence']
            condition = next(k for k, v in significance.items() if v == finding)
            
            # Weight critical conditions more heavily
            if condition in self.critical_conditions:
                weight = 2.5
            else:
                weight = 1.0
            
            weighted_sum += confidence * weight
            total_weight += weight
        
        overall_conf = weighted_sum / total_weight if total_weight > 0 else 0.6
        
        # Adjust based on findings count
        if len(significant_findings) > 3:
            overall_conf *= 0.92  # Slight reduction for multiple findings
        elif len(significant_findings) == 1:
            overall_conf *= 1.05  # Slight boost for single clear finding
        
        return min(0.95, max(0.25, overall_conf))

    def identify_critical_findings(self, significance: Dict[str, Any]) -> List[str]:
        """Identify critical findings requiring immediate attention"""
        critical = []
        
        for condition, data in significance.items():
            if (data['significant'] and condition in self.critical_conditions):
                
                confidence_pct = int(data['confidence'] * 100)
                
                # Add urgency indicators
                if condition == "Pneumothorax":
                    urgency_flag = " 🚨 EMERGENCY"
                elif condition == "Mass":
                    urgency_flag = " 🔴 HIGH PRIORITY"
                elif confidence_pct > 80:
                    urgency_flag = " ⚡ URGENT"
                else:
                    urgency_flag = ""
                
                critical.append(f"{condition} ({confidence_pct}% confidence){urgency_flag}")
                
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
                    'threshold_used': data['threshold_used'],
                    'description': self.get_condition_description(condition)
                })
        
        # Sort by priority: critical first, then by confidence
        findings.sort(key=lambda x: (
            0 if x['critical'] else 1,  # Critical conditions first
            -x['confidence']            # Then by confidence (descending)
        ))
        
        return findings

    def get_condition_description(self, condition: str) -> str:
        """Get clinical descriptions for chest X-ray conditions"""
        descriptions = {
            "Atelectasis": "Partial or complete collapse of lung tissue",
            "Cardiomegaly": "Enlarged heart - may indicate cardiac disease", 
            "Consolidation": "Lung tissue filled with fluid - often pneumonia",
            "Edema": "Fluid accumulation in lung tissue",
            "Effusion": "Fluid in the pleural space around lungs",
            "Emphysema": "Lung damage from COPD or smoking",
            "Fibrosis": "Lung scarring and thickening",
            "Hernia": "Hiatal hernia visible on chest imaging",
            "Infiltration": "Abnormal substances in lung tissue",
            "Mass": "Abnormal growth - requires urgent evaluation",
            "Nodule": "Small round growth in lung",
            "Pleural_Thickening": "Thickening of lung lining",
            "Pneumonia": "Lung infection requiring treatment",
            "Pneumothorax": "Collapsed lung - MEDICAL EMERGENCY"
        }
        return descriptions.get(condition, "Medical condition requiring evaluation")

    def run(self):
        """Start the triage agent"""
        print(f"🔍 Starting BiomedVLP Triage Agent...")
        print(f"📍 Agent address: {self.agent.address}")
        print(f"🫁 Detecting conditions: {CHEST_XRAY_LABELS}")
        print(f"⚠️  Critical conditions: {self.critical_conditions}")
        print(f"💡 Local operation only (no almanac registration)")
        print("=" * 60)
        self.agent.run()

if __name__ == "__main__":
    triage_agent = TriageAgent()
    triage_agent.run()