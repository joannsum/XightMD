from uagents import Agent, Context, Model, Protocol
from uagents.setup import fund_agent_if_low
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import json
import uuid
import base64
import os
from dotenv import load_dotenv

load_dotenv()

# Message models for inter-agent communication
class ImageAnalysisRequest(Model):
    image_data: str
    image_format: str
    request_id: str
    timestamp: str
    patient_info: Dict[str, Any] = {}

class ImageAnalysisResponse(Model):
    request_id: str
    predictions: Dict[str, float]
    urgency_score: int
    confidence_score: float
    critical_findings: list[str]
    all_findings: list[Dict[str, Any]]
    processing_time_ms: int
    timestamp: str
    error: Optional[str] = None

class ReportGenerationRequest(Model):
    request_id: str
    triage_results: Dict[str, Any]
    image_data: str
    timestamp: str

class ReportGenerationResponse(Model):
    request_id: str
    report: Dict[str, str]
    timestamp: str
    error: Optional[str] = None

class QARequest(Model):
    request_id: str
    triage_results: Dict[str, Any]
    report_results: Dict[str, Any]
    timestamp: str

class QAResponse(Model):
    request_id: str
    validation_results: Dict[str, Any]
    timestamp: str
    error: Optional[str] = None

# API request model (from FastAPI)
class AnalysisRequest(Model):
    image_data: str
    image_format: str
    patient_info: Dict[str, Any] = {}

class AnalysisResponse(Model):
    request_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Create the coordinator agent
coordinator = Agent(
    name="coordinator", 
    port=9000,
    seed="coordinator_recovery_phrase", 
    endpoint=["http://127.0.0.1:9000/submit"],
)

# Known agent addresses (you'll get these when you run the other agents)
TRIAGE_AGENT_ADDRESS = "agent1q..."  # Replace with actual address from triage agent
REPORT_AGENT_ADDRESS = "agent1q..."  # Replace with actual address from report agent
QA_AGENT_ADDRESS = "agent1q..."      # Replace with actual address from QA agent

# Fund agent if needed
fund_agent_if_low(coordinator.wallet.address())

# Track active analysis requests
active_requests: Dict[str, Dict[str, Any]] = {}

# Create protocols for different types of communication
analysis_protocol = Protocol("Analysis Coordination")

@analysis_protocol.on_message(model=AnalysisRequest)
async def handle_analysis_request(ctx: Context, sender: str, msg: AnalysisRequest):
    """Handle analysis requests from the API server"""
    request_id = str(uuid.uuid4())
    
    ctx.logger.info(f"🚀 Starting analysis request {request_id}")
    
    # Initialize request tracking
    active_requests[request_id] = {
        'status': 'processing',
        'start_time': datetime.now(),
        'stages': {
            'triage': {'complete': False, 'results': None},
            'report': {'complete': False, 'results': None},
            'qa': {'complete': False, 'results': None}
        },
        'api_sender': sender
    }
    
    try:
        # Step 1: Send to Triage Agent
        triage_request = ImageAnalysisRequest(
            image_data=msg.image_data,
            image_format=msg.image_format,
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            patient_info=msg.patient_info
        )
        
        await ctx.send(TRIAGE_AGENT_ADDRESS, triage_request)
        
        # Send initial response to API
        response = AnalysisResponse(
            request_id=request_id,
            status="processing"
        )
        await ctx.send(sender, response)
        
    except Exception as e:
        ctx.logger.error(f"❌ Error in analysis request {request_id}: {e}")
        active_requests[request_id]['status'] = 'error'
        
        error_response = AnalysisResponse(
            request_id=request_id,
            status="error",
            error=str(e)
        )
        await ctx.send(sender, error_response)

@analysis_protocol.on_message(model=ImageAnalysisResponse)
async def handle_triage_response(ctx: Context, sender: str, msg: ImageAnalysisResponse):
    """Handle response from Triage Agent"""
    request_id = msg.request_id
    
    if request_id not in active_requests:
        ctx.logger.warning(f"⚠️ Received response for unknown request: {request_id}")
        return
    
    ctx.logger.info(f"🔍 Received triage results for {request_id}")
    
    # Store triage results
    active_requests[request_id]['stages']['triage'] = {
        'complete': True,
        'results': {
            'predictions': msg.predictions,
            'urgency_score': msg.urgency_score,
            'confidence_score': msg.confidence_score,
            'critical_findings': msg.critical_findings,
            'all_findings': msg.all_findings,
            'processing_time_ms': msg.processing_time_ms
        }
    }
    
    if msg.error:
        ctx.logger.error(f"❌ Triage error for {request_id}: {msg.error}")
        await handle_analysis_error(ctx, request_id, f"Triage failed: {msg.error}")
        return
    
    # Step 2: Send to Report Agent
    try:
        report_request = ReportGenerationRequest(
            request_id=request_id,
            triage_results=active_requests[request_id]['stages']['triage']['results'],
            image_data="",  # Don't send image data again to save bandwidth
            timestamp=datetime.now().isoformat()
        )
        
        await ctx.send(REPORT_AGENT_ADDRESS, report_request)
        
    except Exception as e:
        await handle_analysis_error(ctx, request_id, f"Failed to send to report agent: {str(e)}")

@analysis_protocol.on_message(model=ReportGenerationResponse)
async def handle_report_response(ctx: Context, sender: str, msg: ReportGenerationResponse):
    """Handle response from Report Agent"""
    request_id = msg.request_id
    
    if request_id not in active_requests:
        ctx.logger.warning(f"⚠️ Received report response for unknown request: {request_id}")
        return
    
    ctx.logger.info(f"📄 Received report results for {request_id}")
    
    # Store report results
    active_requests[request_id]['stages']['report'] = {
        'complete': True,
        'results': msg.report
    }
    
    if msg.error:
        ctx.logger.error(f"❌ Report error for {request_id}: {msg.error}")
        await handle_analysis_error(ctx, request_id, f"Report generation failed: {msg.error}")
        return
    
    # Step 3: Send to QA Agent
    try:
        qa_request = QARequest(
            request_id=request_id,
            triage_results=active_requests[request_id]['stages']['triage']['results'],
            report_results=active_requests[request_id]['stages']['report']['results'],
            timestamp=datetime.now().isoformat()
        )
        
        await ctx.send(QA_AGENT_ADDRESS, qa_request)
        
    except Exception as e:
        await handle_analysis_error(ctx, request_id, f"Failed to send to QA agent: {str(e)}")

@analysis_protocol.on_message(model=QAResponse)
async def handle_qa_response(ctx: Context, sender: str, msg: QAResponse):
    """Handle response from QA Agent"""
    request_id = msg.request_id
    
    if request_id not in active_requests:
        ctx.logger.warning(f"⚠️ Received QA response for unknown request: {request_id}")
        return
    
    ctx.logger.info(f"✅ Received QA results for {request_id}")
    
    # Store QA results
    active_requests[request_id]['stages']['qa'] = {
        'complete': True,
        'results': msg.validation_results
    }
    
    if msg.error:
        ctx.logger.error(f"❌ QA error for {request_id}: {msg.error}")
        await handle_analysis_error(ctx, request_id, f"QA validation failed: {msg.error}")
        return
    
    # All stages complete - compile final results
    await finalize_analysis(ctx, request_id)

async def handle_analysis_error(ctx: Context, request_id: str, error_message: str):
    """Handle analysis errors and notify API"""
    if request_id not in active_requests:
        return
    
    active_requests[request_id]['status'] = 'error'
    api_sender = active_requests[request_id]['api_sender']
    
    error_response = AnalysisResponse(
        request_id=request_id,
        status="error",
        error=error_message
    )
    
    try:
        await ctx.send(api_sender, error_response)
    except Exception as e:
        ctx.logger.error(f"Failed to send error response: {e}")

async def finalize_analysis(ctx: Context, request_id: str):
    """Compile final results and send to API"""
    request_data = active_requests[request_id]
    
    # Compile final results
    final_results = {
        'analysis_id': request_id,
        'timestamp': datetime.now().isoformat(),
        'urgency': request_data['stages']['triage']['results']['urgency_score'],
        'confidence': request_data['stages']['triage']['results']['confidence_score'],
        'findings': request_data['stages']['triage']['results']['critical_findings'],
        'report': request_data['stages']['report']['results'],
        'quality_metrics': request_data['stages']['qa']['results'],
        'processing_summary': {
            'total_time_ms': (datetime.now() - request_data['start_time']).total_seconds() * 1000,
            'agents_involved': ['triage', 'report', 'qa']
        }
    }
    
    # Mark as complete
    active_requests[request_id]['status'] = 'completed'
    
    # Send final response to API
    response = AnalysisResponse(
        request_id=request_id,
        status="completed",
        results=final_results
    )
    
    try:
        api_sender = request_data['api_sender']
        await ctx.send(api_sender, response)
        ctx.logger.info(f"🎉 Analysis completed for {request_id}")
        
        # Clean up after some time
        await asyncio.sleep(300)  # Keep for 5 minutes
        if request_id in active_requests:
            del active_requests[request_id]
            
    except Exception as e:
        ctx.logger.error(f"Failed to send final response: {e}")

# Include the protocol in the agent
coordinator.include(analysis_protocol)

@coordinator.on_startup()
async def agent_startup(ctx: Context):
    ctx.logger.info(f"🚀 Coordinator Agent starting up...")
    ctx.logger.info(f"📍 Agent address: {coordinator.address}")
    ctx.logger.info(f"🌐 Agent endpoints: {coordinator.endpoints}")

@coordinator.on_shutdown()
async def agent_shutdown(ctx: Context):
    ctx.logger.info("👋 Coordinator Agent shutting down...")

if __name__ == "__main__":
    print("🏗️ Starting XightMD Coordinator Agent")
    print(f"📍 Address: {coordinator.address}")
    print(f"🔗 Endpoints: {coordinator.endpoints}")
    print("=" * 50)
    coordinator.run()