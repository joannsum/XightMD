# backend/agents/qa_agent.py
from uagents import Agent, Context, Model
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
import json

# Message models for agent communication
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

class QAAgent:
    def __init__(self):
        self.agent = Agent(
            name="qa_agent", 
            port=8003,
            seed="qa_agent_seed_789",
            endpoint=["http://localhost:8003/submit"]
        )
        
        # QA validation thresholds
        self.confidence_thresholds = {
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.4
        }
        
        # Setup agent handlers
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup agent message handlers"""
        
        @self.agent.on_message(model=QARequest)
        async def handle_qa_validation(ctx: Context, sender: str, msg: QARequest):
            """Handle incoming QA validation requests"""
            ctx.logger.info(f"✅ Received QA validation request: {msg.request_id}")
            
            try:
                # Perform QA validation
                validation_results = await self.validate_analysis_pipeline(
                    msg.triage_results,
                    msg.report_results,
                    msg.request_id
                )
                
                # Send response back to coordinator
                response = QAResponse(
                    request_id=msg.request_id,
                    validation_results=validation_results,
                    timestamp=datetime.now().isoformat()
                )
                
                await ctx.send(sender, response)
                ctx.logger.info(f"✅ QA validation completed for {msg.request_id}")
                
            except Exception as e:
                ctx.logger.error(f"❌ QA validation error for {msg.request_id}: {e}")
                error_response = QAResponse(
                    request_id=msg.request_id,
                    validation_results={},
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
                await ctx.send(sender, error_response)

    async def validate_analysis_pipeline(self, triage_results: Dict[str, Any], report_results: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Validate the entire analysis pipeline"""
        
        # Extract key metrics
        confidence_score = triage_results.get('confidence_score', 0.0)
        urgency_score = triage_results.get('urgency_score', 1)
        critical_findings = triage_results.get('critical_findings', [])
        all_findings = triage_results.get('all_findings', [])
        
        # Perform various validation checks
        confidence_validation = self.validate_confidence_levels(triage_results)
        consistency_validation = self.validate_triage_report_consistency(triage_results, report_results)
        completeness_validation = self.validate_report_completeness(report_results)
        clinical_validation = self.validate_clinical_logic(triage_results, report_results)
        
        # Calculate overall validation scores
        overall_score = self.calculate_overall_validation_score([
            confidence_validation,
            consistency_validation, 
            completeness_validation,
            clinical_validation
        ])
        
        # Generate recommendations
        recommendations = self.generate_qa_recommendations(
            confidence_score, urgency_score, critical_findings, overall_score
        )
        
        # Determine if manual review is needed
        review_required = self.determine_manual_review_requirement(
            confidence_score, urgency_score, critical_findings, overall_score
        )
        
        return {
            'overall_validation_score': overall_score,
            'confidence_validation': confidence_validation,
            'consistency_validation': consistency_validation,
            'completeness_validation': completeness_validation,
            'clinical_validation': clinical_validation,
            'recommendations': recommendations,
            'manual_review_required': review_required,
            'quality_flags': self.generate_quality_flags(triage_results, report_results),
            'validation_summary': self.generate_validation_summary(overall_score, review_required)
        }

    def validate_confidence_levels(self, triage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate confidence levels of predictions"""
        confidence_score = triage_results.get('confidence_score', 0.0)
        predictions = triage_results.get('predictions', {})
        
        high_conf_count = sum(1 for conf in predictions.values() if conf > self.confidence_thresholds['high_confidence'])
        medium_conf_count = sum(1 for conf in predictions.values() if conf > self.confidence_thresholds['medium_confidence'])
        low_conf_count = sum(1 for conf in predictions.values() if conf > self.confidence_thresholds['low_confidence'])
        
        validation_score = confidence_score
        flags = []
        
        if confidence_score < 0.5:
            flags.append("Low overall confidence - consider additional review")
        if high_conf_count == 0 and medium_conf_count == 0:
            flags.append("No high or medium confidence predictions")
        
        return {
            'score': validation_score,
            'high_confidence_predictions': high_conf_count,
            'medium_confidence_predictions': medium_conf_count,
            'low_confidence_predictions': low_conf_count,
            'flags': flags
        }

    def validate_triage_report_consistency(self, triage_results: Dict[str, Any], report_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consistency between triage findings and generated report"""
        critical_findings = triage_results.get('critical_findings', [])
        urgency_score = triage_results.get('urgency_score', 1)
        
        findings_text = report_results.get('findings', '').lower()
        impression_text = report_results.get('impression', '').lower()
        
        consistency_score = 0.8  # Start with base score
        flags = []
        
        # Check if critical findings are reflected in report
        for critical_finding in critical_findings:
            condition = critical_finding.split(' (confidence:')[0].lower()
            if condition not in findings_text and condition not in impression_text:
                consistency_score -= 0.2
                flags.append(f"Critical finding '{condition}' not adequately reflected in report")
        
        # Check urgency vs impression consistency
        if urgency_score >= 4:
            if 'immediate' not in impression_text and 'emergency' not in impression_text:
                consistency_score -= 0.1
                flags.append("High urgency case but impression lacks urgency indicators")
        
        if urgency_score <= 2:
            if 'normal' not in impression_text and 'no acute' not in impression_text:
                consistency_score -= 0.1
                flags.append("Low urgency case but impression suggests abnormalities")
        
        return {
            'score': max(0.0, consistency_score),
            'flags': flags
        }

    def validate_report_completeness(self, report_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate completeness of generated report"""
        required_sections = ['indication', 'comparison', 'findings', 'impression']
        completeness_score = 0.0
        flags = []
        
        for section in required_sections:
            if section in report_results and report_results[section].strip():
                completeness_score += 0.25
            else:
                flags.append(f"Missing or empty {section} section")
        
        # Check minimum length requirements
        findings_length = len(report_results.get('findings', ''))
        impression_length = len(report_results.get('impression', ''))
        
        if findings_length < 50:
            flags.append("Findings section too brief")
        if impression_length < 20:
            flags.append("Impression section too brief")
        
        return {
            'score': completeness_score,
            'sections_complete': len(required_sections) - len([f for f in flags if 'Missing' in f]),
            'total_sections': len(required_sections),
            'flags': flags
        }

    def validate_clinical_logic(self, triage_results: Dict[str, Any], report_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate clinical logic and medical reasoning"""
        all_findings = triage_results.get('all_findings', [])
        urgency_score = triage_results.get('urgency_score', 1)
        impression = report_results.get('impression', '')
        
        logic_score = 0.8  # Start with base score
        flags = []
        
        # Check for contradictory findings
        pneumonia_present = any(f.get('condition') == 'Pneumonia' and f.get('confidence', 0) > 0.6 for f in all_findings)
        normal_impression = 'no acute' in impression.lower() or 'normal' in impression.lower()
        
        if pneumonia_present and normal_impression:
            logic_score -= 0.3
            flags.append("Contradiction: Pneumonia detected but impression suggests normal")
        
        # Check urgency logic
        critical_conditions = ['Pneumothorax', 'Mass', 'Consolidation']
        has_critical = any(f.get('condition') in critical_conditions and f.get('confidence', 0) > 0.7 for f in all_findings)
        
        if has_critical and urgency_score <= 2:
            logic_score -= 0.2
            flags.append("Critical condition detected but urgency score is low")
        
        if not has_critical and urgency_score >= 4:
            logic_score -= 0.2
            flags.append("High urgency score but no critical conditions detected")
        
        return {
            'score': max(0.0, logic_score),
            'flags': flags
        }

    def calculate_overall_validation_score(self, validation_results: List[Dict[str, Any]]) -> float:
        """Calculate overall validation score"""
        scores = [result['score'] for result in validation_results]
        return sum(scores) / len(scores) if scores else 0.0

    def generate_qa_recommendations(self, confidence_score: float, urgency_score: int, critical_findings: List[str], overall_score: float) -> List[str]:
        """Generate QA recommendations"""
        recommendations = []
        
        if confidence_score < 0.6:
            recommendations.append("Consider manual radiologist review due to low confidence")
        
        if urgency_score >= 4:
            recommendations.append("High priority case - recommend immediate clinical attention")
        
        if critical_findings:
            recommendations.append("Critical findings detected - ensure clinical correlation")
        
        if overall_score < 0.7:
            recommendations.append("Overall validation score low - recommend quality review")
        
        if not recommendations:
            recommendations.append("Analysis passed all quality checks")
        
        return recommendations

    def determine_manual_review_requirement(self, confidence_score: float, urgency_score: int, critical_findings: List[str], overall_score: float) -> bool:
        """Determine if manual review is required"""
        return (
            confidence_score < 0.6 or
            urgency_score >= 4 or
            len(critical_findings) > 0 or
            overall_score < 0.7
        )

    def generate_quality_flags(self, triage_results: Dict[str, Any], report_results: Dict[str, Any]) -> List[str]:
        """Generate quality assurance flags"""
        flags = []
        
        confidence_score = triage_results.get('confidence_score', 0.0)
        if confidence_score < 0.5:
            flags.append("LOW_CONFIDENCE")
        
        urgency_score = triage_results.get('urgency_score', 1)
        if urgency_score >= 4:
            flags.append("HIGH_URGENCY")
        
        critical_findings = triage_results.get('critical_findings', [])
        if critical_findings:
            flags.append("CRITICAL_FINDINGS")
        
        if not report_results.get('findings') or len(report_results.get('findings', '')) < 50:
            flags.append("INCOMPLETE_REPORT")
        
        return flags

    def generate_validation_summary(self, overall_score: float, review_required: bool) -> str:
        """Generate human-readable validation summary"""
        if overall_score >= 0.9:
            quality_level = "Excellent"
        elif overall_score >= 0.8:
            quality_level = "Good"
        elif overall_score >= 0.7:
            quality_level = "Acceptable"
        else:
            quality_level = "Needs Improvement"
        
        review_status = "Manual review recommended" if review_required else "Automated processing acceptable"
        
        return f"Quality Assessment: {quality_level} (Score: {overall_score:.2f}). {review_status}."

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'name': 'QA Agent',
            'status': 'active',
            'address': str(self.agent.address),
            'capabilities': [
                'Quality assurance validation',
                'Consistency checking',
                'Confidence assessment',
                'Medical report validation',
                'Cross-agent result verification'
            ],
            'validation_categories': ['confidence', 'consistency', 'completeness', 'clinical_logic'],
            'thresholds': self.confidence_thresholds,
            'last_updated': datetime.now().isoformat()
        }

    def run(self):
        """Start the agent"""
        print(f"✅ Starting QA Agent...")
        print(f"📍 Agent address: {self.agent.address}")
        print(f"🔗 Agent endpoints: {self.agent.endpoints}")
        self.agent.run()

if __name__ == "__main__":
    # Initialize and run the QA agent
    qa_agent = QAAgent()
    qa_agent.run()