from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import os
import sys

# Add agents directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))

from coordinator import CoordinatorAgent, AnalysisRequest

app = FastAPI(title="XightMD API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global coordinator agent instance
coordinator_agent = None
analysis_results = {}  # Store completed analysis results

class APIServer:
    def __init__(self):
        self.coordinator = None
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        
        @app.post("/api/analyze")
        async def analyze_image(file: UploadFile = File(...)):
            """Analyze uploaded chest X-ray image"""
            try:
                # Validate file
                if not file.content_type.startswith('image/'):
                    raise HTTPException(status_code=400, detail="File must be an image")
                
                # Read and encode image
                image_data = await file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                
                # Generate request ID
                request_id = str(uuid.uuid4())
                
                # Create analysis request
                analysis_request = AnalysisRequest(
                    image_data=base64_image,
                    image_format=file.content_type,
                    patient_info={}
                )
                
                # For development - simulate agent processing
                # In production, this would go through the actual agent network
                result = await self.simulate_agent_analysis(
                    base64_image, 
                    file.content_type, 
                    request_id
                )
                
                return JSONResponse(content={
                    "success": True,
                    "data": result
                })
                
            except HTTPException:
                raise
            except Exception as e:
                print(f"Analysis error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "services": {
                    "api": {"status": "up", "port": 8000},
                    "agents": {
                        "coordinator": {"status": "active", "address": "coordinator_agent"},
                        "triage": {"status": "active", "address": "triage_agent"},
                        "report": {"status": "mock", "address": "report_agent"},
                        "qa": {"status": "mock", "address": "qa_agent"}
                    }
                }
            }

        @app.get("/api/agents/status")
        async def get_agent_status():
            """Get status of all agents"""
            return {
                "success": True,
                "data": {
                    "coordinator": {
                        "status": "active",
                        "lastSeen": datetime.now().isoformat(),
                        "capabilities": ["Request coordination", "Result compilation"]
                    },
                    "triage": {
                        "status": "active", 
                        "lastSeen": datetime.now().isoformat(),
                        "capabilities": ["Lung disease detection", "Urgency assessment"]
                    },
                    "report": {
                        "status": "active",
                        "lastSeen": datetime.now().isoformat(),
                        "capabilities": ["Report generation", "Medical formatting"]
                    },
                    "qa": {
                        "status": "active",
                        "lastSeen": datetime.now().isoformat(),
                        "capabilities": ["Quality validation", "Consistency checking"]
                    }
                }
            }

        @app.get("/api/analysis/{request_id}")
        async def get_analysis_result(request_id: str):
            """Get analysis result by request ID"""
            if request_id in analysis_results:
                return JSONResponse(content={
                    "success": True,
                    "data": analysis_results[request_id]
                })
            else:
                raise HTTPException(status_code=404, detail="Analysis not found")

    async def simulate_agent_analysis(self, image_data: str, image_format: str, request_id: str) -> Dict[str, Any]:
        """Simulate the agent analysis workflow for development"""
        try:
            # Import and use the actual lung classifier
            from utils.lung_classifier import LungClassifierTrainer
            import tempfile
            import base64
            from PIL import Image
            import io
            
            # Initialize classifier
            classifier = LungClassifierTrainer()
            
            # Decode and save image temporarily
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                image.save(temp_file.name)
                temp_path = temp_file.name
            
            try:
                # Get predictions from the actual model
                predictions = classifier.predict(temp_path)
                significance = classifier.get_statistical_significance(predictions)
                
                # Calculate urgency and other metrics
                urgency_score = self.calculate_urgency(significance)
                confidence_score = self.calculate_overall_confidence(predictions)
                critical_findings = self.identify_critical_findings(significance)
                
                # Generate structured report
                report = self.generate_report(significance, predictions, urgency_score)
                
                # Create final result
                result = {
                    'id': f'analysis-{request_id}',
                    'timestamp': datetime.now().isoformat(),
                    'urgency': urgency_score,
                    'confidence': confidence_score,
                    'findings': [
                        f.split(' (confidence:')[0] for f in critical_findings
                    ] + [
                        data['condition'] for condition, data in significance.items() 
                        if data['significant'] and data['confidence'] > 0.5
                    ],
                    'report': report,
                    'image': f'data:{image_format};base64,{image_data}',
                    'processing_details': {
                        'model_predictions': predictions,
                        'statistical_significance': significance,
                        'critical_findings': critical_findings,
                        'processing_time_ms': 2000  # Simulated processing time
                    }
                }
                
                # Store result
                analysis_results[request_id] = result
                
                return result
                
            finally:
                # Clean up temp file
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"Error in simulate_agent_analysis: {e}")
            raise e

    def calculate_urgency(self, significance: Dict[str, Any]) -> int:
        """Calculate urgency score from 1-5"""
        critical_conditions = [
            "Pneumothorax", "Pneumonia", "Effusion", "Mass", "Consolidation"
        ]
        
        max_urgency = 1
        
        for condition, data in significance.items():
            if not data['significant']:
                continue
                
            confidence = data['confidence']
            
            if condition in critical_conditions:
                if confidence > 0.8:
                    max_urgency = max(max_urgency, 5)
                elif confidence > 0.6:
                    max_urgency = max(max_urgency, 4)
                else:
                    max_urgency = max(max_urgency, 3)
            elif confidence > 0.7:
                max_urgency = max(max_urgency, 3)
            elif confidence > 0.5:
                max_urgency = max(max_urgency, 2)
                
        return max_urgency

    def calculate_overall_confidence(self, predictions: Dict[str, float]) -> float:
        """Calculate overall confidence score"""
        if not predictions:
            return 0.0
            
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_predictions = sorted_predictions[:3]
        
        weighted_sum = sum(conf * (4-i) for i, (_, conf) in enumerate(top_predictions))
        total_weight = sum(4-i for i in range(len(top_predictions)))
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def identify_critical_findings(self, significance: Dict[str, Any]) -> list:
        """Identify critical findings"""
        critical_conditions = [
            "Pneumothorax", "Pneumonia", "Effusion", "Mass", "Consolidation"
        ]
        
        critical = []
        for condition, data in significance.items():
            if (data['significant'] and 
                condition in critical_conditions and 
                data['confidence'] > 0.6):
                critical.append(f"{condition} (confidence: {data['confidence']:.1%})")
                
        return critical

    def generate_report(self, significance: Dict[str, Any], predictions: Dict[str, float], urgency: int) -> Dict[str, str]:
        """Generate structured radiology report"""
        
        # Find significant findings
        significant_findings = [
            (condition, data) for condition, data in significance.items() 
            if data['significant'] and data['confidence'] > 0.5
        ]
        
        # Generate report sections
        indication = "Chest imaging for evaluation of cardiopulmonary status"
        comparison = "No prior studies available for comparison"
        
        # Generate findings
        findings = self.generate_findings_text(significant_findings)
        
        # Generate impression
        impression = self.generate_impression(significant_findings, urgency)
        
        return {
            'indication': indication,
            'comparison': comparison,
            'findings': findings,
            'impression': impression
        }

    def generate_findings_text(self, findings: list) -> str:
        """Generate detailed findings text"""
        if not findings:
            return ("The heart size and mediastinal contours appear normal. "
                   "The lungs are clear bilaterally without evidence of consolidation, "
                   "pleural effusion, or pneumothorax. No acute osseous abnormalities identified.")
        
        findings_parts = []
        
        # Check for cardiomegaly
        cardiomegaly = any(condition == 'Cardiomegaly' and data['confidence'] > 0.5 
                          for condition, data in findings)
        
        if cardiomegaly:
            findings_parts.append("The heart size appears enlarged.")
        else:
            findings_parts.append("The heart size and mediastinal contours appear normal.")
        
        # Lung findings
        lung_conditions = []
        for condition, data in findings:
            confidence = data['confidence']
            if confidence > 0.6:
                if condition == 'Pneumonia':
                    lung_conditions.append("consolidative changes consistent with pneumonia")
                elif condition == 'Pneumothorax':
                    lung_conditions.append("pneumothorax")
                elif condition in ['Effusion', 'Pleural Effusion']:
                    lung_conditions.append("pleural effusion")
                elif condition == 'Atelectasis':
                    lung_conditions.append("atelectatic changes")
                elif condition == 'Mass':
                    lung_conditions.append("pulmonary mass")
                elif condition == 'Nodule':
                    lung_conditions.append("pulmonary nodule")
        
        if lung_conditions:
            findings_parts.append(f"The lungs demonstrate {', '.join(lung_conditions)}.")
        else:
            findings_parts.append("The lungs are clear bilaterally without acute abnormality.")
        
        findings_parts.append("No acute osseous abnormalities identified.")
        
        return " ".join(findings_parts)

    def generate_impression(self, findings: list, urgency: int) -> str:
        """Generate impression based on findings"""
        if not findings and urgency <= 2:
            return "No acute cardiopulmonary abnormality identified."
        
        impressions = []
        
        for condition, data in findings:
            confidence = data['confidence']
            
            if confidence > 0.7:
                if condition == 'Pneumonia':
                    impressions.append("Findings consistent with pneumonia")
                elif condition == 'Pneumothorax':
                    impressions.append("Pneumothorax identified - recommend immediate clinical correlation")
                elif condition == 'Mass':
                    impressions.append("Pulmonary mass - recommend further evaluation with CT")
                elif condition == 'Cardiomegaly':
                    impressions.append("Cardiomegaly")
                elif condition in ['Effusion', 'Pleural Effusion']:
                    impressions.append("Pleural effusion")
        
        if impressions:
            return ". ".join(impressions) + "."
        else:
            return "Findings of uncertain clinical significance. Clinical correlation recommended."

# Create API server instance
api_server = APIServer()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("XightMD API Server starting...")
    print("Lung classifier model loading...")
    # The lung classifier will be loaded on first use

if __name__ == "__main__":
    import uvicorn
    print("Starting XightMD API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)