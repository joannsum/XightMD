# backend/agents/report_agent.py
from uagents import Agent, Context, Model
from typing import Dict, Any, Optional
from datetime import datetime
import os
import json
import sys
import base64
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import anthropic

# Message models for agent communication
class ReportGenerationRequest(Model):
    request_id: str
    triage_results: Dict[str, Any]
    image_data: str  # base64 encoded image
    user_description: Optional[str] = None  # User's description of symptoms/concerns
    priority_search: Optional[str] = None   # What to specifically look for
    timestamp: str

class ReportGenerationResponse(Model):
    request_id: str
    report: Dict[str, str]
    pdf_data: str  # base64 encoded PDF
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
        
        # Initialize Claude client
        self.claude_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
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
                # Generate structured report using Claude
                report = await self.generate_radiology_report_with_claude(
                    msg.triage_results,
                    msg.image_data,
                    msg.user_description,
                    msg.priority_search,
                    msg.request_id
                )
                
                # Generate PDF
                pdf_data = await self.generate_pdf_report(
                    report,
                    msg.triage_results,
                    msg.image_data,
                    msg.request_id
                )
                
                # Send response back to coordinator
                response = ReportGenerationResponse(
                    request_id=msg.request_id,
                    report=report,
                    pdf_data=pdf_data,
                    timestamp=datetime.now().isoformat()
                )
                
                await ctx.send(sender, response)
                ctx.logger.info(f"✅ Report and PDF generated for {msg.request_id}")
                
            except Exception as e:
                ctx.logger.error(f"❌ Report generation error for {msg.request_id}: {e}")
                error_response = ReportGenerationResponse(
                    request_id=msg.request_id,
                    report={},
                    pdf_data="",
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
                await ctx.send(sender, error_response)

    async def generate_radiology_report_with_claude(
        self, 
        triage_results: Dict[str, Any], 
        image_data: str,
        user_description: Optional[str],
        priority_search: Optional[str],
        request_id: str
    ) -> Dict[str, str]:
        """Generate structured radiology report using Claude's vision capabilities"""
        
        # Prepare context for Claude
        context_parts = []
        
        # Add user description if provided
        if user_description:
            context_parts.append(f"Patient/User Description: {user_description}")
        
        # Add priority search if provided
        if priority_search:
            context_parts.append(f"Specific Areas of Concern: {priority_search}")
        
        # Add triage results
        predictions = triage_results.get('predictions', {})
        urgency_score = triage_results.get('urgency_score', 1)
        confidence_score = triage_results.get('confidence_score', 0.0)
        critical_findings = triage_results.get('critical_findings', [])
        all_findings = triage_results.get('all_findings', [])
        
        # Format findings for Claude
        findings_text = ""
        if all_findings:
            findings_text = "AI Model Detected Conditions:\n"
            for finding in all_findings[:5]:  # Top 5 findings
                condition = finding.get('condition', 'Unknown')
                confidence = finding.get('confidence', 0)
                findings_text += f"- {condition}: {confidence:.1%} confidence\n"
        
        # Create prompt for Claude
        prompt = f"""You are an expert radiologist reviewing a chest X-ray. Please generate a professional radiology report based on the image and the following information:

{chr(10).join(context_parts) if context_parts else ""}

AI Analysis Results:
- Urgency Score: {urgency_score}/5
- Overall Confidence: {confidence_score:.1%}
- Critical Findings: {', '.join(critical_findings) if critical_findings else 'None'}

{findings_text}

Please generate a complete radiology report with the following sections:

1. INDICATION: Reason for examination
2. COMPARISON: Prior studies reference
3. FINDINGS: Detailed radiological observations
4. IMPRESSION: Summary and clinical recommendations

The report should be professional, accurate, and consider both the AI analysis and the image. If there are discrepancies between the AI analysis and what you observe, please note them. Remember this is for educational/research purposes and should not replace clinical judgment."""

        try:
            # Call Claude API with image and text
            message = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            # Parse Claude's response
            claude_response = message.content[0].text
            
            # Extract sections from Claude's response
            report = self.parse_claude_response(claude_response, triage_results)
            
            return report
            
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            # Fallback to rule-based generation
            return self.generate_fallback_report(triage_results, user_description, priority_search)

    def parse_claude_response(self, claude_response: str, triage_results: Dict[str, Any]) -> Dict[str, str]:
        """Parse Claude's response into structured sections"""
        
        sections = {
            'indication': '',
            'comparison': '',
            'findings': '',
            'impression': ''
        }
        
        # Try to extract sections from Claude's response
        lines = claude_response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Identify section headers
            if any(keyword in line.upper() for keyword in ['INDICATION:', 'INDICATION']):
                current_section = 'indication'
                # Extract content after colon if present
                if ':' in line:
                    sections[current_section] = line.split(':', 1)[1].strip()
                continue
            elif any(keyword in line.upper() for keyword in ['COMPARISON:', 'COMPARISON']):
                current_section = 'comparison'
                if ':' in line:
                    sections[current_section] = line.split(':', 1)[1].strip()
                continue
            elif any(keyword in line.upper() for keyword in ['FINDINGS:', 'FINDINGS']):
                current_section = 'findings'
                if ':' in line:
                    sections[current_section] = line.split(':', 1)[1].strip()
                continue
            elif any(keyword in line.upper() for keyword in ['IMPRESSION:', 'IMPRESSION']):
                current_section = 'impression'
                if ':' in line:
                    sections[current_section] = line.split(':', 1)[1].strip()
                continue
            
            # Add content to current section
            if current_section and line:
                if sections[current_section]:
                    sections[current_section] += " " + line
                else:
                    sections[current_section] = line
        
        # Ensure all sections have content
        for section in sections:
            if not sections[section]:
                sections[section] = self.get_fallback_section_content(section, triage_results)
        
        return sections

    def get_fallback_section_content(self, section: str, triage_results: Dict[str, Any]) -> str:
        """Generate fallback content for missing sections"""
        urgency_score = triage_results.get('urgency_score', 1)
        
        fallback_content = {
            'indication': "Chest X-ray examination for evaluation of cardiopulmonary status",
            'comparison': "No prior studies available for comparison",
            'findings': "Chest X-ray findings are being analyzed. Please refer to the detailed AI analysis results.",
            'impression': "Analysis in progress. Clinical correlation recommended." if urgency_score <= 2 else "Findings require clinical attention and correlation."
        }
        
        return fallback_content.get(section, "Content not available")

    def generate_fallback_report(self, triage_results: Dict[str, Any], user_description: Optional[str], priority_search: Optional[str]) -> Dict[str, str]:
        """Generate fallback report when Claude API fails"""
        
        predictions = triage_results.get('predictions', {})
        urgency_score = triage_results.get('urgency_score', 1)
        confidence_score = triage_results.get('confidence_score', 0.0)
        critical_findings = triage_results.get('critical_findings', [])
        all_findings = triage_results.get('all_findings', [])
        
        # Generate indication
        indication = "Chest X-ray examination"
        if user_description:
            indication += f" for {user_description.lower()}"
        if priority_search:
            indication += f", with focus on {priority_search.lower()}"
        
        # Generate findings
        findings_parts = ["The chest X-ray demonstrates:"]
        
        if all_findings:
            for finding in all_findings[:3]:  # Top 3 findings
                condition = finding.get('condition', '')
                confidence = finding.get('confidence', 0)
                if confidence > 0.5:
                    findings_parts.append(f"- {condition} with {confidence:.0%} confidence")
        
        if not any(findings_parts[1:]):  # If no significant findings
            findings_parts.append("- No acute abnormalities detected by automated analysis")
        
        # Generate impression
        if critical_findings:
            impression = f"Findings suggest: {', '.join([f.split(' (confidence:')[0] for f in critical_findings])}. "
            impression += "Recommend immediate clinical correlation."
        elif urgency_score >= 3:
            impression = "Findings requiring clinical attention detected. Recommend prompt evaluation."
        else:
            impression = "No acute abnormalities identified by automated analysis. Clinical correlation recommended."
        
        return {
            'indication': indication,
            'comparison': "No prior studies available for comparison",
            'findings': " ".join(findings_parts),
            'impression': impression
        }

    async def generate_pdf_report(
        self, 
        report: Dict[str, str], 
        triage_results: Dict[str, Any], 
        image_data: str, 
        request_id: str
    ) -> str:
        """Generate a professional PDF report"""
        
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
            ['Urgency Level:', self.get_urgency_text(triage_results.get('urgency_score', 1))],
            ['Confidence:', f"{triage_results.get('confidence_score', 0):.1%}"]
        ]
        
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
        
        # X-ray image (if available)
        if image_data:
            try:
                # Decode base64 image
                img_data = base64.b64decode(image_data)
                img = Image(BytesIO(img_data))
                
                # Resize image to fit page
                img.drawHeight = 3*inch
                img.drawWidth = 4*inch
                
                story.append(Paragraph("Chest X-ray Image", section_style))
                story.append(img)
                story.append(Spacer(1, 20))
            except Exception as e:
                print(f"Error adding image to PDF: {e}")
        
        # Report sections
        sections = [
            ('INDICATION', report.get('indication', '')),
            ('COMPARISON', report.get('comparison', '')),
            ('FINDINGS', report.get('findings', '')),
            ('IMPRESSION', report.get('impression', ''))
        ]
        
        for section_name, section_content in sections:
            story.append(Paragraph(section_name, section_style))
            story.append(Paragraph(section_content, content_style))
            story.append(Spacer(1, 10))
        
        # AI Analysis Summary
        story.append(Paragraph("AI ANALYSIS SUMMARY", section_style))
        
        # Create findings table
        findings_data = [['Condition', 'Confidence', 'Status']]
        all_findings = triage_results.get('all_findings', [])
        
        for finding in all_findings[:5]:  # Top 5 findings
            condition = finding.get('condition', 'Unknown')
            confidence = finding.get('confidence', 0)
            is_critical = finding.get('critical', False)
            status = 'Critical' if is_critical else 'Detected' if confidence > 0.6 else 'Possible'
            
            findings_data.append([
                condition,
                f"{confidence:.1%}",
                status
            ])
        
        if len(findings_data) > 1:  # If we have findings
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
        else:
            story.append(Paragraph("No significant findings detected by AI analysis.", content_style))
        
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

    def get_urgency_text(self, urgency_score: int) -> str:
        """Convert urgency score to human-readable text"""
        if urgency_score >= 4:
            return "High Priority - Immediate Attention Required"
        elif urgency_score >= 3:
            return "Medium Priority - Prompt Evaluation Recommended"
        elif urgency_score >= 2:
            return "Low Priority - Routine Follow-up"
        else:
            return "Normal - No Immediate Concerns"

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'name': 'Report Agent',
            'status': 'active',
            'address': str(self.agent.address),
            'capabilities': [
                'Claude-powered report generation',
                'Professional PDF creation',
                'Medical terminology formatting',
                'Multi-modal analysis (image + text)',
                'User input integration'
            ],
            'report_sections': ['indication', 'comparison', 'findings', 'impression'],
            'last_updated': datetime.now().isoformat()
        }

    def run(self):
        """Start the agent"""
        print(f"📄 Starting Enhanced Report Agent...")
        print(f"📍 Agent address: {self.agent.address}")
        print(f"🤖 Claude integration: {'Enabled' if self.claude_client else 'Disabled'}")
        print(f"📊 PDF generation: Enabled")
        print("=" * 50)
        self.agent.run()

if __name__ == "__main__":
    # Initialize and run the enhanced report agent
    report_agent = ReportAgent()
    report_agent.run()