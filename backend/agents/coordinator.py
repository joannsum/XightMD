from uagents import Agent, Context, Model
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import json
import uuid
import base64
from triage_agent import ImageAnalysisRequest, ImageAnalysisResponse

# Message models
class AnalysisRequest(Model):
    image_data: str
    image_format: str
    patient_info: Dict[str, Any] = {}

class AnalysisResponse(Model):
    request_id: str
    status: str
    triage_results: Optional[Dict[str, Any]] = None
    report_results: Optional[Dict[str, Any]] = None
    qa_results: Optional[Dict[str, Any]] = None
    final_results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class CoordinatorAgent:
    def __init__(self):
        self.agent = Agent(
            name="coordinator_agent",
            port=8000,
            seed="coordinator_seed_123",
            endpoint=["http://localhost:8000/submit"]
        )
        
        # Agent addresses
        self.triage_address = "agent1qw8r4vtmb8s3f6q9r7t4p2x4n8d8z5j6k9h7g2a3s5f8q1w3e4r7t9"  # Triage agent
        self.report_address = "agent1qw8r4vtmb8s3f6q9r7t4p2x4n8d8z5j6k9h7g2a3s5f8q1w3e4r7t8"  # Report agent
        self.qa_address = "agent1qw8r4vtmb8s3f6q9r7t4p2x4n8d8z5j6k9h7g2a3s5f8q1w3e4r7t7"     # QA agent
        
        # Active requests tracking
        self.active_requests = {}
        
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup message handlers"""
        
        @self.agent.on_message(model=AnalysisRequest)
        async def handle_analysis_request(ctx: Context, sender: str, msg: AnalysisRequest):
            """Handle incoming analysis requests from API"""
            request_id = str(uuid.uuid4())
            
            ctx.logger.info(f"Starting analysis request {request_id}")
            
            # Initialize request tracking
            self.active_requests[request_id] = {
                'status': 'processing',
                'triage_complete': False,
                'report_complete': False,
                'qa_complete': False,
                'start_time': datetime.now(),
                'results': {}
            }
            
            try:
                # Step 1: Send to Triage Agent
                triage_request = ImageAnalysisRequest(
                    image_data=msg.image_data,
                    image_format=msg.image_format,
                    request_id=request_id,
                    timestamp=datetime.now().isoformat()
                )
                
                await ctx.send(self.triage_address, triage_request)
                
                # Send initial response
                response = AnalysisResponse(
                    request_id=request_id,
                    status="processing",
                    triage_results=None
                )
                await ctx.send(sender, response)
                
            except Exception as e:
                ctx.logger.error(f"Error in analysis request {request_id}: {e}")
                error_response = AnalysisResponse(
                    request_id=request_id,
                    status="error",
                    error=str(e)
                )
                await ctx.send(sender, error_response)

        @self.agent.on_message(model=ImageAnalysisResponse)
        async def handle_triage_response(ctx: Context, sender: str, msg: ImageAnalysisResponse):
            """Handle response from Triage Agent"""
            request_id = msg.request_id
            
            if request_id not in self.active_requests:
                ctx.logger.warning(f"Received response for unknown request: {request_id}")
                return
            
            ctx.logger.info(f"Received triage results for {request_id}")
            
            # Store triage results
            self.active_requests[request_id]['triage_complete'] = True
            self.active_requests[request_id]['results']['triage'] = {
                'predictions': msg.predictions,
                'urgency_score': msg.urgency_score,
                'confidence_score': msg.confidence_score,
                'critical_findings': msg.critical_findings,
                'all_findings': msg.all_findings,
                'processing_time_ms': msg.processing_time_ms
            }
            
            # For now, create mock report and QA results
            # In a full implementation, you'd send to actual report and QA agents
            await self.generate_mock_report_and_qa(ctx, request_id, msg)

    async def generate_mock_report_and_qa(self, ctx: Context, request_id: str, triage_msg: ImageAnalysisResponse):
        """Generate mock report and QA results (placeholder for actual agents)"""
        try:
            # Mock report generation
            report_results = self.generate_mock_report(triage_msg)
            
            # Mock QA validation
            qa_results = self.generate_mock_qa(triage_msg, report_results)
            
            # Update tracking
            self.active_requests[request_id]['report_complete'] = True
            self.active_requests[request_id]['qa_complete'] = True
            self.active_requests[request_id]['results']['report'] = report_results
            self.active_requests[request_id]['results']['qa'] = qa_results
            
            # Generate final results
            final_results = self.compile_final_results(request_id)
            
            # Mark as complete
            self.active_requests[request_id]['status'] = 'completed'
            self.active_requests[request_id]['results']['final'] = final_results
            
            ctx.logger.info(f"Analysis completed for {request_id}")
            
        except Exception as e:
            ctx.logger.error(f"Error generating report/QA for {request_id}: {e}")
            self.active_requests[request_id]['status'] = 'error'
            self.active_requests[request_id]['error'] = str(e)

    def generate_mock_report(self, triage_msg: ImageAnalysisResponse) -> Dict[str, Any]:
        """Generate a structured radiology report based on triage results"""
        critical_findings = triage_msg.critical_findings
        all_findings = triage_msg.all_findings
        
        # Generate indication
        indication = "Chest pain, shortness of breath, rule out pneumonia"
        
        # Generate comparison
        comparison = "No prior chest radiographs available for comparison"
        
        # Generate findings section
        findings_text = self.generate_findings_text(all_findings, triage_msg.predictions)
        
        # Generate impression
        impression = self.generate_impression(critical_findings, all_findings, triage_msg.urgency_score)
        
        return {
            'indication': indication,
            'comparison': comparison,
            'findings': findings_text,
            'impression': impression,
            'generated_at': datetime.now().isoformat()
        }

    def generate_findings_text(self, findings: List[Dict[str, Any]], predictions: Dict[str, float]) -> str:
        """Generate detailed findings text"""
        if not findings:
            return ("The heart size and mediastinal contours appear normal. "
                   "The lungs are clear bilaterally without evidence of consolidation, "
                   "pleural effusion, or pneumothorax. No acute osseous abnormalities identified.")
        
        findings_parts = []
        
        # Heart and mediastinum
        if any(f['condition'] == 'Cardiomegaly' for f in findings if f['confidence'] > 0.5):
            findings_parts.append("The heart size appears enlarged.")
        else:
            findings_parts.append("The heart size and mediastinal contours appear normal.")
        
        # Lung findings
        lung_findings = []
        for finding in findings:
            condition = finding['condition']
            confidence = finding['confidence']
            
            if confidence > 0.6:
                if condition == 'Pneumonia':
                    lung_findings.append("consolidative changes consistent with pneumonia")
                elif condition == 'Pneumothorax':
                    lung_findings.append("pneumothorax")
                elif condition == 'Pleural Effusion' or condition == 'Effusion':
                    lung_findings.append("pleural effusion")
                elif condition == 'Atelectasis':
                    lung_findings.append("atelectatic changes")
                elif condition == 'Mass':
                    lung_findings.append("pulmonary mass")
                elif condition == 'Nodule':
                    lung_findings.append("pulmonary nodule")
        
        if lung_findings:
            findings_parts.append(f"The lungs demonstrate {', '.join(lung_findings)}.")
        else:
            findings_parts.append("The lungs are clear bilaterally without acute abnormality.")
        
        # Bones
        findings_parts.append("No acute osseous abnormalities identified.")
        
        return " ".join(findings_parts)

    def generate_impression(self, critical_findings: List[str], all_findings: List[Dict[str, Any]], urgency: int) -> str:
        """Generate impression based on findings"""
        if not critical_findings and urgency <= 2:
            return "No acute cardiopulmonary abnormality identified."
        
        impressions = []
        
        for finding in all_findings:
            condition = finding['condition']
            confidence = finding['confidence']
            
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

    def generate_mock_qa(self, triage_msg: ImageAnalysisResponse, report_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate QA validation results"""
        
        # Calculate consistency score
        consistency_score = 0.85 + (triage_msg.confidence_score * 0.15)
        
        # Generate validation flags
        validation_flags = []
        if triage_msg.urgency_score >= 4:
            validation_flags.append("High urgency case - recommend senior radiologist review")
        
        if triage_msg.confidence_score < 0.7:
            validation_flags.append("Lower confidence predictions - consider additional imaging")
        
        # Check for critical findings
        critical_alert = len(triage_msg.critical_findings) > 0
        
        return {
            'consistency_score': consistency_score,
            'validation_flags': validation_flags,
            'critical_alert': critical_alert,
            'review_recommended': triage_msg.urgency_score >= 4 or triage_msg.confidence_score < 0.6,
            'validated_at': datetime.now().isoformat()
        }

    def compile_final_results(self, request_id: str) -> Dict[str, Any]:
        """Compile final analysis results"""
        request_data = self.active_requests[request_id]
        triage = request_data['results']['triage']
        report = request_data['results']['report']
        qa = request_data['results']['qa']
        
        return {
            'analysis_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'urgency': triage['urgency_score'],
            'confidence': triage['confidence_score'],
            'findings': [f.split(' (confidence:')[0] for f in triage['critical_findings']] + 
                       [f['condition'] for f in triage['all_findings'] if f['confidence'] > 0.5],
            'report': {
                'indication': report['indication'],
                'comparison': report['comparison'],
                'findings': report['findings'],
                'impression': report['impression']
            },
            'quality_metrics': {
                'consistency_score': qa['consistency_score'],
                'review_recommended': qa['review_recommended'],
                'critical_alert': qa['critical_alert']
            },
            'processing_summary': {
                'total_time_ms': (datetime.now() - request_data['start_time']).total_seconds() * 1000,
                'triage_time_ms': triage['processing_time_ms'],
                'agents_involved': ['triage', 'report', 'qa']
            }
        }

    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific request"""
        if request_id not in self.active_requests:
            return None
        
        request_data = self.active_requests[request_id]
        return {
            'request_id': request_id,
            'status': request_data['status'],
            'progress': {
                'triage_complete': request_data['triage_complete'],
                'report_complete': request_data['report_complete'],
                'qa_complete': request_data['qa_complete']
            },
            'results': request_data['results'] if request_data['status'] == 'completed' else None
        }

    def run(self):
        """Start the coordinator agent"""
        print(f"Starting Coordinator Agent...")
        print(f"Agent address: {self.agent.address}")
        self.agent.run()

if __name__ == "__main__":
    coordinator = CoordinatorAgent()
    coordinator.run()