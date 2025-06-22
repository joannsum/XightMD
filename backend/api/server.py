# backend/api/server.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import asyncio
import uuid
import httpx
from datetime import datetime
from typing import Dict, Any, Optional
import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()



# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Add the backend directory to the Python path to import utils
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)


app = FastAPI(title="XightMD API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agent endpoint configuration for health checks
AGENT_PORTS = {
    "coordinator": 8000,  # This server
    "triage": 9001,       # Updated ports to match uAgents
    "report": 9002,
    "qa": 9003
}

MODAL_ENDPOINT = "https://joannsum--xightmd-simple-predict-lung-conditions.modal.run"

# Store completed analysis results
analysis_results = {}

class APIServer:
    def __init__(self):
        self.lung_classifier = None
        self.model_loaded = False
        self.setup_routes()
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize Modal connection instead of local model"""
        try:
            logger.info("🤖 Connecting to Modal service...")
            
            # Test Modal connection instead of loading local model
            import asyncio
            asyncio.create_task(self.test_modal_connection())
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Modal: {e}")
            self.model_loaded = False

    async def test_modal_connection(self):
        """Test Modal service availability"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("https://joannsum--xightmd-simple-health.modal.run", timeout=10)
                if response.status_code == 200:
                    self.model_loaded = True
                    logger.info("✅ Modal service connected successfully!")
                else:
                    logger.warning(f"⚠️ Modal service returned status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Modal connection failed: {e}")
            self.model_loaded = False
    
    def setup_routes(self):
        """Setup API routes"""
        
        @app.post("/api/analyze")
        async def analyze_image(file: UploadFile = File(...)):
            """Analyze uploaded chest X-ray image"""
            try:
                # Validate file
                if not file.content_type or not file.content_type.startswith('image/'):
                    raise HTTPException(status_code=400, detail="File must be an image")
                
                # Check file size (15MB limit)
                file_size = 0
                image_data = await file.read()
                file_size = len(image_data)
                
                if file_size > 15 * 1024 * 1024:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds 15MB limit"
                    )
                
                # Encode image
                base64_image = base64.b64encode(image_data).decode('utf-8')
                
                # Generate request ID
                request_id = str(uuid.uuid4())
                
                logger.info(f"🔍 Starting analysis for request {request_id} (file: {file.filename}, size: {file_size / 1024 / 1024:.1f}MB)")
                
                # Analyze with lung classifier
                result = await self.analyze_with_lung_classifier(
                    base64_image, 
                    file.content_type, 
                    request_id,
                    file.filename
                )
                
                return JSONResponse(content={
                    "success": True,
                    "data": result
                })
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ Analysis error: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "service": "xightmd_api",
                "model_loaded": self.model_loaded,
                "lung_classifier": "available" if self.model_loaded else "unavailable"
            }

        # Add this to your server.py - improved agent status detection

    @app.get("/api/agents/status")
    async def get_agent_status():
        """Get real status of all agents using simple port checking"""
        import socket
        
        agent_statuses = {}
        
        agents = {
            "coordinator": {"port": 9000, "address": os.getenv("COORDINATOR_AGENT_ADDRESS", "agent1q...")},
            "triage": {"port": 8001, "address": os.getenv("TRIAGE_AGENT_ADDRESS", "agent1q...")},
            "report": {"port": 8002, "address": os.getenv("REPORT_AGENT_ADDRESS", "agent1q...")},
            "qa": {"port": 8006, "address": os.getenv("QA_AGENT_ADDRESS", "agent1q...")}  # CHANGED: 8003 → 8006
        }
        
        logger.info("🔍 Checking agent network status...")
        
        for agent_name, config in agents.items():
            try:
                # Simple port check
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('localhost', config["port"]))
                sock.close()
                
                port_listening = (result == 0)
                address_valid = not config["address"].endswith("agent1q...")
                
                if port_listening and address_valid:
                    status = "active"
                    last_seen = datetime.now().isoformat()
                    details = {
                        "port": config["port"],
                        "address": config["address"],
                        "type": "uagent",
                        "connection": "listening",
                        "check_result": "port_open_and_configured"
                    }
                elif port_listening and not address_valid:
                    status = "idle"
                    last_seen = datetime.now().isoformat()
                    details = {
                        "port": config["port"],
                        "address": config["address"],
                        "type": "uagent",
                        "connection": "listening",
                        "warning": "Using placeholder address - check .env file",
                        "check_result": "port_open_but_not_configured"
                    }
                else:
                    status = "offline"
                    last_seen = ""
                    details = {
                        "error": "Port not listening - agent not running",
                        "port": config["port"],
                        "address": config["address"],
                        "type": "uagent",
                        "check_result": "port_closed"
                    }
                
                agent_statuses[agent_name] = {
                    "status": status,
                    "lastSeen": last_seen,
                    "details": details
                }
                
                logger.info(f"✅ {agent_name}: {status} (port {config['port']}, listening: {port_listening})")
                
            except Exception as e:
                logger.error(f"❌ Error checking {agent_name}: {e}")
                agent_statuses[agent_name] = {
                    "status": "error",
                    "lastSeen": "",
                    "details": {
                        "error": str(e),
                        "port": config["port"],
                        "type": "uagent"
                    }
                }
        
        # Calculate network health
        active_count = sum(1 for a in agent_statuses.values() if a["status"] == "active")
        total_count = len(agent_statuses)
        
        if active_count == total_count:
            network_health = "optimal"
        elif active_count > 0:
            network_health = "degraded"
        else:
            network_health = "offline"
        
        logger.info(f"🌐 Network health: {network_health} ({active_count}/{total_count} agents active)")
        
        return {
            "success": True,
            "agents": agent_statuses,
            "timestamp": datetime.now().isoformat(),
            "network_health": network_health,
            "summary": {
                "total_agents": total_count,
                "active_agents": active_count,
                "health_percentage": (active_count / total_count) * 100 if total_count > 0 else 0
            }
        }
            
    @app.get("/api/modal-health")
    async def check_modal_health():
        """Check if Modal service is healthy"""
        try:
            modal_health_url = "https://joannsum--xightmd-simple-health.modal.run"
            async with httpx.AsyncClient() as client:
                response = await client.get(modal_health_url, timeout=10)
                if response.status_code == 200:
                    return {"modal_status": "healthy", "modal_response": response.json()}
                else:
                    return {"modal_status": "unhealthy", "status_code": response.status_code}
        except Exception as e:
            return {"modal_status": "error", "error": str(e)}

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

    @app.get("/api/model/status")
    async def get_model_status():
        """Get detailed model status"""
        model_info = {
            "model_loaded": self.model_loaded,
            "model_path": os.path.join(backend_dir, 'models', 'lung_classifier_best.pth'),
            "model_exists": os.path.exists(os.path.join(backend_dir, 'models', 'lung_classifier_best.pth')),
            "backend_dir": backend_dir,
            "python_path": sys.path[:3]  # First 3 entries
        }
        
        if self.lung_classifier:
            try:
                # Get model info if available
                model_info["classifier_type"] = type(self.lung_classifier).__name__
                if hasattr(self.lung_classifier, 'labels'):
                    model_info["supported_labels"] = self.lung_classifier.labels
            except:
                pass
        
        return model_info

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "XightMD Backend API",
            "version": "1.0.0",
            "status": "running",
            "model_status": "loaded" if self.model_loaded else "not_loaded",
            "endpoints": {
                "health": "/api/health",
                "analyze": "/api/analyze",
                "agent_status": "/api/agents/status",
                "model_status": "/api/model/status",
                "docs": "/docs"
            }
        }
        


    async def analyze_with_lung_classifier(self, image_data: str, image_format: str, request_id: str, filename: str = None) -> Dict[str, Any]:
        """Use Modal service for analysis instead of local classifier"""
        start_time = datetime.now()
        
        try:
            logger.info(f"🤖 Running analysis via Modal service...")
            
            # Call Modal instead of local model
            async with httpx.AsyncClient() as client:
                payload = {
                    "image_data": image_data,
                    "image_format": image_format.split('/')[-1]
                }
                
                response = await client.post(MODAL_ENDPOINT, json=payload, timeout=60)
                
                if response.status_code == 200:
                    modal_result = response.json()
                    
                    if modal_result.get("success"):
                        # Convert Modal result to your existing format
                        predictions = modal_result["predictions"]
                        urgency_score = modal_result["urgency"]
                        confidence_score = modal_result["confidence"]
                        
                        # Create significance data for your existing methods
                        significance = self.create_significance_from_modal(predictions)
                        critical_findings = self.identify_critical_findings(significance)
                        
                        # Generate report using your existing logic
                        report = self.generate_report(significance, predictions, urgency_score)
                        
                        processing_time = (datetime.now() - start_time).total_seconds() * 1000
                        
                        logger.info(f"✅ Modal analysis complete - Urgency: {urgency_score}, Confidence: {confidence_score:.2f}")
                        
                        # Create result in your existing format
                        result = {
                            'id': f'analysis-{request_id[:8]}',
                            'timestamp': datetime.now().isoformat(),
                            'urgency': urgency_score,
                            'confidence': confidence_score,
                            'findings': [
                                f.split(' (confidence:')[0] for f in critical_findings
                            ] + [
                                condition for condition, data in significance.items() 
                                if data['significant'] and data['confidence'] > 0.5
                            ][:5],
                            'report': report,
                            'image': f'data:{image_format};base64,{image_data[:100]}...',
                            'processing_details': {
                                'model_predictions': {k: round(v, 3) for k, v in predictions.items()},
                                'statistical_significance': {
                                    k: {
                                        'significant': v['significant'],
                                        'confidence': round(v['confidence'], 3),
                                        'confidence_level': v['confidence_level']
                                    } for k, v in significance.items() if v['significant']
                                },
                                'critical_findings': critical_findings,
                                'processing_time_ms': round(processing_time),
                                'pipeline': ['modal_service', 'report_generator'],
                                'mode': 'modal_analysis',
                                'filename': filename,
                                'model': 'modal_service'
                            }
                        }
                        
                        analysis_results[request_id] = result
                        return result
                    else:
                        raise Exception(f"Modal service error: {modal_result.get('error', 'Unknown error')}")
                else:
                    raise Exception(f"Modal service returned status {response.status_code}")
                            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"❌ Modal service error: {e}")
            
            return self.create_fallback_result(
                image_data, image_format, request_id, 
                f"Modal service error: {str(e)}", processing_time
            )

    def create_significance_from_modal(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """Convert Modal predictions to significance format for existing methods"""
        significance = {}
        
        for condition, confidence in predictions.items():
            is_significant = confidence > 0.1
            
            if confidence > 0.7:
                confidence_level = "high"
            elif confidence > 0.4:
                confidence_level = "medium"
            else:
                confidence_level = "low"
            
            significance[condition] = {
                'significant': is_significant,
                'confidence': confidence,
                'confidence_level': confidence_level
            }
        
        return significance

    def create_fallback_result(self, image_data: str, image_format: str, request_id: str, error_msg: str, processing_time: float = 500) -> Dict[str, Any]:
        """Create a fallback result when the lung classifier isn't available"""
        logger.info(f"🔄 Creating fallback analysis result: {error_msg}")
        
        result = {
            'id': f'analysis-{request_id[:8]}',
            'timestamp': datetime.now().isoformat(),
            'urgency': 2,
            'confidence': 0.5,
            'findings': [
                'Image processed successfully',
                'Automated analysis unavailable',
                'Manual radiologist review recommended'
            ],
            'report': {
                'indication': 'Chest imaging for evaluation of cardiopulmonary status',
                'comparison': 'No prior studies available for comparison',
                'findings': ('The submitted chest radiograph has been received and processed. '
                           'Image quality appears adequate for diagnostic interpretation. '
                           'Automated lung disease classification is currently unavailable. '
                           'Manual radiologist review is recommended for comprehensive evaluation.'),
                'impression': ('Image successfully processed. '
                             'Automated analysis system temporarily unavailable. '
                             'Recommend manual radiologist review for detailed findings and clinical correlation.')
            },
            'image': f'data:{image_format};base64,{image_data[:100]}...',
            'processing_details': {
                'mode': 'fallback_processing',
                'error': error_msg,
                'recommendation': 'Check model file and dependencies',
                'pipeline': ['image_processor', 'basic_validator'],
                'processing_time_ms': processing_time,
                'model': 'unavailable'
            }
        }
        
        # Store result
        analysis_results[request_id] = result
        return result

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

if __name__ == "__main__":
    import uvicorn
    logger.info("🏗️ Starting XightMD API Server...")
    logger.info(f"📁 Backend directory: {backend_dir}")
    logger.info(f"🤖 Model file: {os.path.join(backend_dir, 'models', 'lung_classifier_best.pth')}")
    
    uvicorn.run(
        "server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )