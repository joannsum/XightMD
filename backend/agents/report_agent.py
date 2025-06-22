# backend/agents/report_agent.py
from uagents import Agent, Context, Model
from typing import Dict, Any, Optional
from datetime import datetime
import os

# Message models
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
        
        # Medical report templates and guidelines
        self.setup_medical_knowledge()
        self.setup_handlers()
        
    def setup_medical_knowledge(self):
        """Initialize medical knowledge base for reports"""
        
        # Standard indications based on findings
        self.indication_templates = {
            'high_urgency': "Acute chest symptoms, evaluation for emergency pathology",
            'pneumonia': "Suspected pneumonia, fever and respiratory symptoms",
            'trauma': "Chest trauma, rule out pneumothorax and fractures",
            'routine': "Routine chest imaging for cardiopulmonary assessment",
            'followup': "Follow-up chest imaging, monitoring known condition"
        }
        
        # Medical terminology for findings
        self.medical_findings = {
            "Pneumothorax": {
                "description": "pneumothorax",
                "severity": "CRITICAL",
                "recommendation": "immediate clinical correlation and urgent intervention"
            },
            "Mass": {
                "description": "pulmonary mass lesion",
                "severity": "HIGH",
                "recommendation": "further evaluation with contrast CT and tissue sampling"
            },
            "Pneumonia": {
                "description": "consolidative changes consistent with pneumonia",
                "severity": "HIGH",
                "recommendation": "antibiotic therapy and clinical correlation"
            },
            "Cardiomegaly": {
                "description": "cardiomegaly",
                "severity": "MODERATE",
                "recommendation": "echocardiogram and cardiology consultation"
            },
            "Effusion": {
                "description": "pleural effusion",
                "severity": "MODERATE",
                "recommendation": "thoracentesis if symptomatic"
            },
            "Atelectasis": {
                "description": "atelectatic changes",
                "severity": "MILD",
                "recommendation": "pulmonary hygiene and incentive spirometry"
            },
            "Consolidation": {
                "description": "pulmonary consolidation",
                "severity": "HIGH",
                "recommendation": "antibiotic therapy consideration"
            },
            "Nodule": {
                "description": "pulmonary nodule",
                "severity": "MODERATE",
                "recommendation": "follow-up imaging per Fleischner guidelines"
            },
            "Edema": {
                "description": "pulmonary edema",
                "severity": "HIGH",
                "recommendation": "diuretic therapy and cardiac evaluation"
            }
        }
        
    def setup_handlers(self):
        """Setup agent message handlers"""
        
        @self.agent.on_message(model=ReportGenerationRequest)
        async def handle_report_generation(ctx: Context, sender: str, msg: ReportGenerationRequest):
            """Generate professional radiology report"""
            ctx.logger.info(f"📄 Generating report for {msg.request_id}")
            
            try:
                report = await self.generate_professional_report(
                    msg.triage_results,
                    msg.request_id
                )
                
                response = ReportGenerationResponse(
                    request_id=msg.request_id,
                    report=report,
                    timestamp=datetime.now().isoformat()
                )
                
                ctx.logger.info(f"✅ Report generated for {msg.request_id}")
                await ctx.send(sender, response)
                
            except Exception as e:
                ctx.logger.error(f"❌ Report generation failed for {msg.request_id}: {e}")
                error_response = ReportGenerationResponse(
                    request_id=msg.request_id,
                    report={},
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
                await ctx.send(sender, error_response)

    async def generate_professional_report(self, triage_results: Dict[str, Any], request_id: str) -> Dict[str, str]:
        """Generate medical-grade radiology report"""
        
        urgency_score = triage_results.get('urgency_score', 1)
        confidence_score = triage_results.get('confidence_score', 0.0)
        critical_findings = triage_results.get('critical_findings', [])
        all_findings = triage_results.get('all_findings', [])
        
        # Generate each section
        indication = self.generate_indication(urgency_score, critical_findings)
        comparison = self.generate_comparison()
        findings = self.generate_findings_section(all_findings, critical_findings)
        impression = self.generate_impression(critical_findings, all_findings, urgency_score)
        
        return {
            'indication': indication,
            'comparison': comparison,
            'findings': findings,
            'impression': impression
        }

    def generate_indication(self, urgency_score: int, critical_findings: list) -> str:
        """Generate appropriate clinical indication"""
        
        if urgency_score >= 5:
            return self.indication_templates['high_urgency']
        elif any('Pneumonia' in finding for finding in critical_findings):
            return self.indication_templates['pneumonia']
        elif any('Pneumothorax' in finding for finding in critical_findings):
            return self.indication_templates['trauma']
        elif urgency_score >= 3:
            return "Chest symptoms requiring radiological evaluation"
        else:
            return self.indication_templates['routine']

    def generate_comparison(self) -> str:
        """Generate comparison section"""
        return "No prior studies available for comparison at this time."

    def generate_findings_section(self, all_findings: list, critical_findings: list) -> str:
        """Generate detailed medical findings"""
        findings_parts = []
        
        # Start with heart assessment
        heart_findings = [f for f in all_findings if f.get('condition') == 'Cardiomegaly']
        if heart_findings and heart_findings[0]['confidence'] > 0.5:
            if heart_findings[0]['confidence'] > 0.7:
                findings_parts.append("The cardiac silhouette is enlarged.")
            else:
                findings_parts.append("The cardiac silhouette is at the upper limits of normal.")
        else:
            findings_parts.append("The cardiac silhouette and mediastinal contours are within normal limits.")
        
        # Process lung findings by severity
        critical_lung_findings = []
        moderate_lung_findings = []
        mild_lung_findings = []
        
        for finding in all_findings:
            if finding['condition'] == 'Cardiomegaly':
                continue  # Already handled
                
            condition = finding['condition']
            confidence = finding['confidence']
            
            if condition in self.medical_findings:
                medical_info = self.medical_findings[condition]
                
                if medical_info['severity'] == 'CRITICAL' and confidence > 0.6:
                    critical_lung_findings.append(medical_info['description'])
                elif medical_info['severity'] == 'HIGH' and confidence > 0.6:
                    moderate_lung_findings.append(medical_info['description'])
                elif confidence > 0.5:
                    mild_lung_findings.append(medical_info['description'])
        
        # Generate lung findings text
        if critical_lung_findings:
            findings_parts.append(f"CRITICAL: The lungs demonstrate {', '.join(critical_lung_findings)}.")
        
        if moderate_lung_findings:
            findings_parts.append(f"The lungs show {', '.join(moderate_lung_findings)}.")
        
        if mild_lung_findings:
            findings_parts.append(f"Mild {', '.join(mild_lung_findings)} is noted.")
        
        if not critical_lung_findings and not moderate_lung_findings and not mild_lung_findings:
            findings_parts.append("The lungs are clear bilaterally without evidence of consolidation, pleural effusion, or pneumothorax.")
        
        # Add standard assessments
        findings_parts.append("The osseous structures appear intact without acute fracture.")
        
        return " ".join(findings_parts)

    def generate_impression(self, critical_findings: list, all_findings: list, urgency_score: int) -> str:
        """Generate clinical impression with recommendations"""
        
        if not critical_findings and urgency_score <= 2:
            return "No acute cardiopulmonary abnormality identified."
        
        impression_parts = []
        recommendations = []
        
        # Process critical findings first
        for critical_finding in critical_findings:
            condition_name = critical_finding.split(' (')[0]
            
            if condition_name in self.medical_findings:
                medical_info = self.medical_findings[condition_name]
                
                # Add finding to impression
                if medical_info['severity'] == 'CRITICAL':
                    impression_parts.append(f"{medical_info['description'].upper()}")
                else:
                    impression_parts.append(medical_info['description'].title())
                
                # Add recommendation
                recommendations.append(medical_info['recommendation'])
        
        # Process other significant findings
        significant_findings = [f for f in all_findings 
                             if f.get('confidence', 0) > 0.6 and not f.get('critical', False)]
        
        for finding in significant_findings:
            condition = finding['condition']
            if condition in self.medical_findings:
                medical_info = self.medical_findings[condition]
                impression_parts.append(medical_info['description'])
                if medical_info['severity'] in ['HIGH', 'MODERATE']:
                    recommendations.append(medical_info['recommendation'])
        
        # Combine impression and recommendations
        final_impression = []
        
        if impression_parts:
            if len(impression_parts) == 1:
                final_impression.append(f"{impression_parts[0].title()}.")
            else:
                final_impression.append(f"{', '.join(impression_parts[:-1])}, and {impression_parts[-1]}.")
        
        # Add urgency-based recommendations
        if urgency_score >= 5:
            recommendations.insert(0, "STAT clinical correlation and immediate intervention")
        elif urgency_score >= 4:
            recommendations.insert(0, "urgent clinical correlation")
        elif urgency_score >= 3:
            recommendations.insert(0, "clinical correlation recommended")
        
        # Add recommendations to impression
        if recommendations:
            if len(recommendations) == 1:
                final_impression.append(f"Recommend {recommendations[0]}.")
            else:
                final_impression.append(f"Recommend {recommendations[0]} and {recommendations[1]}.")
        
        if final_impression:
            return " ".join(final_impression)
        else:
            return "Findings of uncertain clinical significance. Clinical correlation recommended."

    def run(self):
        """Start the report agent"""
        print(f"📄 Starting Report Agent...")
        print(f"📍 Agent address: {self.agent.address}")
        print(f"🏥 Medical-grade reporting enabled")
        print("=" * 50)
        self.agent.run()

if __name__ == "__main__":
    report_agent = ReportAgent()
    report_agent.run()