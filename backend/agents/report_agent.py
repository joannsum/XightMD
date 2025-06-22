# backend/agents/report_agent.py
from uagents import Agent, Context, Model
from typing import Dict, Any, Optional
from datetime import datetime
import os
import json
import sys

# Message models for agent communication
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

class ReportAgent:
    def __init__(self):
        self.agent = Agent(
            name="report_agent",
            port=8002,
            seed="report_agent_seed_456",
            endpoint=["http://localhost:8002/submit"]
        )
        
        # Setup agent handlers
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup agent message handlers"""
        
        @self.agent.on_message(model=ReportGenerationRequest)
        async def handle_report_generation(ctx: Context, sender: str, msg: ReportGenerationRequest):
            """Handle incoming report generation requests"""
            ctx.logger.info(f"📄 Received report generation request: {msg.request_id}")
            
            try:
                # Generate structured report
                report = await self.generate_radiology_report(
                    msg.triage_results,
                    msg.request_id
                )
                
                # Send response back to coordinator
                response = ReportGenerationResponse(
                    request_id=msg.request_id,
                    report=report,
                    timestamp=datetime.now().isoformat()
                )
                
                await ctx.send(sender, response)
                ctx.logger.info(f"✅ Report generated for {msg.request_id}")
                
            except Exception as e:
                ctx.logger.error(f"❌ Report generation error for {msg.request_id}: {e}")
                error_response = ReportGenerationResponse(
                    request_id=msg.request_id,
                    report={},
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
                await ctx.send(sender, error_response)

    async def generate_radiology_report(self, triage_results: Dict[str, Any], request_id: str) -> Dict[str, str]:
        """Generate structured radiology report based on triage results"""
        
        # Extract key information from triage results
        predictions = triage_results.get('predictions', {})
        urgency_score = triage_results.get('urgency_score', 1)
        confidence_score = triage_results.get('confidence_score', 0.0)
        critical_findings = triage_results.get('critical_findings', [])
        all_findings = triage_results.get('all_findings', [])
        
        # Generate indication
        indication = self.generate_indication(urgency_score, critical_findings)
        
        # Generate comparison
        comparison = "No prior studies available for comparison"
        
        # Generate findings section
        findings = self.generate_findings_section(all_findings, predictions)
        
        # Generate impression
        impression = self.generate_impression(critical_findings, all_findings, urgency_score)
        
        return {
            'indication': indication,
            'comparison': comparison,
            'findings': findings,
            'impression': impression
        }

    def generate_indication(self, urgency_score: int, critical_findings: list) -> str:
        """Generate indication based on urgency and findings"""
        if urgency_score >= 4 or critical_findings:
            return "Acute chest symptoms, evaluation for emergency conditions"
        elif urgency_score >= 3:
            return "Chest symptoms, rule out acute pathology"
        else:
            return "Routine chest imaging for cardiopulmonary evaluation"

    def generate_findings_section(self, findings: list, predictions: Dict[str, float]) -> str:
        """Generate detailed findings section"""
        findings_parts = []
        
        # Check for cardiomegaly
        cardiomegaly_confidence = predictions.get('Cardiomegaly', 0.0)
        if cardiomegaly_confidence > 0.5:
            if cardiomegaly_confidence > 0.7:
                findings_parts.append("The heart size is enlarged.")
            else:
                findings_parts.append("The heart size is at the upper limits of normal.")
        else:
            findings_parts.append("The heart size and mediastinal contours appear normal.")
        
        # Process lung findings
        lung_conditions = []
        significant_findings = [f for f in findings if f.get('confidence', 0) > 0.6]
        
        for finding in significant_findings:
            condition = finding.get('condition', '')
            confidence = finding.get('confidence', 0)
            
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
            elif condition == 'Consolidation':
                lung_conditions.append("consolidation")
            elif condition == 'Edema':
                lung_conditions.append("pulmonary edema")
            elif condition == 'Emphysema':
                lung_conditions.append("emphysematous changes")
            elif condition == 'Fibrosis':
                lung_conditions.append("fibrotic changes")
        
        # Generate lung findings text
        if lung_conditions:
            if len(lung_conditions) == 1:
                findings_parts.append(f"The lungs demonstrate {lung_conditions[0]}.")
            else:
                findings_parts.append(f"The lungs demonstrate {', '.join(lung_conditions[:-1])}, and {lung_conditions[-1]}.")
        else:
            findings_parts.append("The lungs are clear bilaterally without evidence of consolidation, pleural effusion, or pneumothorax.")
        
        # Add osseous findings
        findings_parts.append("No acute osseous abnormalities are identified.")
        
        return " ".join(findings_parts)

    def generate_impression(self, critical_findings: list, all_findings: list, urgency_score: int) -> str:
        """Generate impression based on findings"""
        if not critical_findings and urgency_score <= 2:
            return "No acute cardiopulmonary abnormality identified."
        
        impressions = []
        
        # Process critical findings first
        for critical_finding in critical_findings:
            condition = critical_finding.split(' (confidence:')[0]
            
            if condition == 'Pneumonia':
                impressions.append("Findings consistent with pneumonia")
            elif condition == 'Pneumothorax':
                impressions.append("Pneumothorax identified - recommend immediate clinical evaluation")
            elif condition == 'Mass':
                impressions.append("Pulmonary mass - recommend further evaluation with CT chest")
            elif condition in ['Effusion', 'Pleural Effusion']:
                impressions.append("Pleural effusion")
            elif condition == 'Consolidation':
                impressions.append("Pulmonary consolidation")
        
        # Add other significant findings
        significant_findings = [f for f in all_findings if f.get('confidence', 0) > 0.7 and f.get('critical', False) == False]
        
        for finding in significant_findings:
            condition = finding.get('condition', '')
            if condition == 'Cardiomegaly':
                impressions.append("Cardiomegaly")
            elif condition == 'Atelectasis':
                impressions.append("Atelectasis")
            elif condition == 'Emphysema':
                impressions.append("Emphysematous changes")
            elif condition == 'Fibrosis':
                impressions.append("Pulmonary fibrosis")
        
        # Add recommendations based on urgency
        if urgency_score >= 4:
            impressions.append("Recommend immediate clinical correlation")
        elif urgency_score >= 3:
            impressions.append("Recommend clinical correlation")
        
        if impressions:
            return ". ".join(impressions) + "."
        else:
            return "Findings of uncertain clinical significance. Clinical correlation recommended."

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'name': 'Report Agent',
            'status': 'active',
            'address': str(self.agent.address),
            'capabilities': [
                'Structured report generation',
                'Medical terminology formatting',
                'ReXGradient-160K report structure',
                'Clinical findings interpretation'
            ],
            'report_sections': ['indication', 'comparison', 'findings', 'impression'],
            'last_updated': datetime.now().isoformat()
        }

    def run(self):
        """Start the agent"""
        print(f"📄 Starting Report Agent...")
        print(f"📍 Agent address: {self.agent.address}")
        print(f"🔗 Agent endpoints: {self.agent.endpoints}")
        self.agent.run()

if __name__ == "__main__":
    # Initialize and run the report agent
    report_agent = ReportAgent()
    report_agent.run()