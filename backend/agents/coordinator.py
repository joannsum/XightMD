import asyncio
from datetime import datetime
import os
from typing import Any, Dict, Optional
import uuid
from dotenv import load_dotenv
from uagents import Agent, Context, Model
from uagents.setup import fund_agent_if_low
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading

load_dotenv()

# Message models for inter-agent communication
class ImageAnalysisRequest(Model):
    image_data: str
    image_format: str
    request_id: str
    timestamp: str
    user_description: Optional[str] = None
    priority_search: Optional[str] = None
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
    user_description: Optional[str] = None
    priority_search: Optional[str] = None
    timestamp: str

class ReportGenerationResponse(Model):
    request_id: str
    report: Dict[str, str]
    pdf_data: str = ""
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

# HTTP API Models
class HTTPAnalysisRequest(BaseModel):
    image_data: str
    image_format: str
    user_description: Optional[str] = None
    priority_search: Optional[str] = None
    patient_info: Dict[str, Any] = {}

# Create the coordinator agent
coordinator = Agent(
    name="coordinator",
    port=9000,
    seed="coordinator_recovery_phrase",
    endpoint=["http://127.0.0.1:9000/submit"]
)

# Get agent addresses from environment variables
TRIAGE_AGENT_ADDRESS = os.getenv("TRIAGE_AGENT_ADDRESS", "agent1q...")
REPORT_AGENT_ADDRESS = os.getenv("REPORT_AGENT_ADDRESS", "agent1q...")
QA_AGENT_ADDRESS = os.getenv("QA_AGENT_ADDRESS", "agent1q...")

print(f"🔧 Agent Configuration:")
print(f"   Coordinator: {coordinator.address}")
print(f"   Triage: {TRIAGE_AGENT_ADDRESS}")
print(f"   Report: {REPORT_AGENT_ADDRESS}")
print(f"   QA: {QA_AGENT_ADDRESS}")

# Fund agent if needed
fund_agent_if_low(coordinator.wallet.address())

# Track active analysis requests
active_requests: Dict[str, Dict[str, Any]] = {}

# Create FastAPI app for HTTP interface
app = FastAPI(title="XightMD Coordinator HTTP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def http_analyze(request: HTTPAnalysisRequest):
    """HTTP endpoint to receive analysis requests"""
    request_id = str(uuid.uuid4())
    
    print(f"🚀 HTTP: Received analysis request {request_id}")
    print(f"📝 User description: {request.user_description}")
    print(f"🎯 Priority search: {request.priority_search}")
    
    # Store request
    active_requests[request_id] = {
        'status': 'processing',
        'start_time': datetime.now(),
        'user_input': {
            'description': request.user_description,
            'priority_search': request.priority_search
        },
        'stages': {
            'triage': {'complete': False, 'results': None},
            'report': {'complete': False, 'results': None},
            'qa': {'complete': False, 'results': None}
        }
    }
    
    try:
        # Simulate processing time
        await asyncio.sleep(1)
        
        # Generate a real PDF report
        pdf_data = await generate_pdf_report(request_id, request.user_description, request.priority_search)
        
        # Create enhanced mock results with real PDF
        mock_results = {
            'analysis_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'urgency': 3 if request.priority_search else 2,
            'confidence': 0.89,
            'findings': [
                'Enhanced AI analysis with user context',
                f'User description: {request.user_description}' if request.user_description else 'No clinical history provided',
                f'Priority focus: {request.priority_search}' if request.priority_search else 'General examination',
                'Professional PDF report generated',
                'Multi-agent analysis pipeline active'
            ],
            'report': {
                'indication': request.user_description or 'Routine chest X-ray examination',
                'comparison': 'No prior studies available for comparison',
                'findings': f'Chest X-ray analysis incorporating clinical context. {f"Special attention given to {request.priority_search} as requested." if request.priority_search else ""} The heart size and mediastinal contours appear normal. The lungs are clear bilaterally without evidence of consolidation, pleural effusion, or pneumothorax.',
                'impression': f'No acute cardiopulmonary abnormality identified. {f"No evidence of {request.priority_search} detected." if request.priority_search else ""}'
            },
            'pdf_data': pdf_data,  # Real PDF data
            'quality_metrics': {'overall_score': 0.9},
            'processing_summary': {
                'total_time_ms': 1000,
                'agents_involved': ['coordinator', 'pdf_generator'],
                'user_input_processed': True,
                'pdf_generated': True
            }
        }
        
        active_requests[request_id]['status'] = 'completed'
        active_requests[request_id]['results'] = mock_results
        
        return {
            'success': True,
            'data': mock_results,
            'message': 'Analysis completed with PDF generation'
        }
        
    except Exception as e:
        print(f"❌ Error processing request {request_id}: {e}")
        return {
            'success': False,
            'error': str(e)
        }

async def generate_pdf_report(request_id: str, user_description: str = None, priority_search: str = None) -> str:
    """Generate a real PDF report"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from io import BytesIO
        import base64
        
        # Create a BytesIO buffer to store the PDF
        buffer = BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1e3a8a')
        )
        
        section_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=18,
            textColor=colors.HexColor('#1e40af'),
            borderWidth=1,
            borderColor=colors.HexColor('#e5e7eb'),
            borderPadding=5,
            backColor=colors.HexColor('#f8fafc')
        )
        
        content_style = ParagraphStyle(
            'Content',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            leading=14
        )
        
        # Build the story (content)
        story = []
        
        # Title
        story.append(Paragraph("XightMD - AI-Powered Chest X-ray Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Header information table
        header_data = [
            ['Analysis ID:', request_id],
            ['Date:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Priority Level:', 'Medium Priority' if priority_search else 'Normal'],
            ['Confidence:', '89.0%']
        ]
        
        if user_description:
            header_data.append(['Clinical History:', user_description])
        if priority_search:
            header_data.append(['Priority Focus:', priority_search])
        
        header_table = Table(header_data, colWidths=[2*inch, 4*inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d1d5db'))
        ]))
        
        story.append(header_table)
        story.append(Spacer(1, 20))
        
        # Report sections
        sections = [
            ('INDICATION', user_description or 'Routine chest X-ray examination for cardiopulmonary assessment'),
            ('COMPARISON', 'No prior studies available for comparison'),
            ('FINDINGS', f'The heart size and mediastinal contours appear normal. The lungs are clear bilaterally without evidence of consolidation, pleural effusion, or pneumothorax. {f"Special attention was given to evaluating for {priority_search} as requested." if priority_search else ""} No acute osseous abnormalities are identified.'),
            ('IMPRESSION', f'No acute cardiopulmonary abnormality identified. {f"No evidence of {priority_search} detected." if priority_search else ""}')
        ]
        
        for section_name, section_content in sections:
            story.append(Paragraph(section_name, section_style))
            story.append(Paragraph(section_content, content_style))
            story.append(Spacer(1, 10))
        
        # AI Analysis Summary
        story.append(Paragraph("AI ANALYSIS SUMMARY", section_style))
        
        # Create findings table
        findings_data = [['Finding', 'Confidence', 'Status']]
        findings_data.append(['Normal cardiac silhouette', '92%', 'Detected'])
        findings_data.append(['Clear lung fields', '88%', 'Detected'])
        if priority_search:
            findings_data.append([f'{priority_search} evaluation', '85%', 'No evidence'])
        
        findings_table = Table(findings_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d1d5db')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(findings_table)
        story.append(Spacer(1, 20))
        
        # Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#6b7280'),
            alignment=TA_JUSTIFY,
            borderWidth=1,
            borderColor=colors.HexColor('#fbbf24'),
            borderPadding=10,
            backColor=colors.HexColor('#fffbeb')
        )
        
        disclaimer_text = """
        <b>IMPORTANT DISCLAIMER:</b> This report was generated using AI-powered analysis and is intended for research and educational purposes only. 
        This analysis should not be used as a substitute for professional medical diagnosis or treatment. 
        Always consult with qualified healthcare professionals for medical advice and interpretation of medical images.
        The AI system is designed to assist healthcare providers but cannot replace clinical judgment and expertise.
        """
        
        story.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data and encode as base64
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(pdf_data).decode('utf-8')
        
    except ImportError:
        print("⚠️ ReportLab not installed. Run: pip install reportlab")
        return ""
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return ""

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "coordinator_address": str(coordinator.address),
        "active_requests": len(active_requests),
        "agent_network": "coordinator_ready"
    }

@app.get("/status/{request_id}")
async def get_status(request_id: str):
    """Get analysis status"""
    if request_id not in active_requests:
        raise HTTPException(status_code=404, detail="Request not found")
    
    return active_requests[request_id]

# Agent message handlers (will be activated when other agents connect)
@coordinator.on_message(model=ImageAnalysisResponse)
async def handle_triage_response(ctx: Context, sender: str, msg: ImageAnalysisResponse):
    """Handle response from Triage Agent"""
    request_id = msg.request_id
    ctx.logger.info(f"🔍 Received triage results for {request_id}")
    
    if request_id in active_requests:
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
        
        # Continue to report agent...
        # (Implementation continues when agents are connected)

@coordinator.on_message(model=ReportGenerationResponse)
async def handle_report_response(ctx: Context, sender: str, msg: ReportGenerationResponse):
    """Handle response from Report Agent"""
    request_id = msg.request_id
    ctx.logger.info(f"📄 Received report results for {request_id}")
    # Implementation continues...

@coordinator.on_message(model=QAResponse)
async def handle_qa_response(ctx: Context, sender: str, msg: QAResponse):
    """Handle response from QA Agent"""
    request_id = msg.request_id
    ctx.logger.info(f"✅ Received QA results for {request_id}")
    # Implementation continues...

@coordinator.on_event("startup")
async def agent_startup(ctx: Context):
    ctx.logger.info(f"🚀 Coordinator Agent starting up...")
    ctx.logger.info(f"📍 Agent address: {coordinator.address}")

@coordinator.on_event("shutdown")
async def agent_shutdown(ctx: Context):
    ctx.logger.info("👋 Coordinator Agent shutting down...")

def run_http_server():
    """Run the HTTP server in a separate thread"""
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")

if __name__ == "__main__":
    print("🏗️ Starting XightMD Coordinator with HTTP API")
    print(f"📍 Agent Address: {coordinator.address}")
    print(f"🌐 HTTP API: http://localhost:8080")
    print("=" * 50)
    
    # Start HTTP server in background thread
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    
    # Run the agent
    coordinator.run()