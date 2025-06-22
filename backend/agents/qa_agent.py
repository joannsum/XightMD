# backend/agents/qa_agent.py
from uagents import Agent, Context, Model
from typing import Dict, Any, Optional, List
from datetime import datetime

# Message models
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

class MedicalQAAgent:
    def __init__(self):
        self.agent = Agent(
            name="qa_agent", 
            port=8006,
            seed="qa_agent_seed_789",
            endpoint=["http://localhost:8006/submit"]
        )
        
        # Medical validation thresholds (optimized for chest X-rays)
        self.medical_thresholds = {
            # Critical conditions - lower thresholds for safety
            'critical_conditions': {
                'Pneumothorax': 0.3,    # Very low threshold - emergency
                'Mass': 0.4,            # Low threshold - cancer screening
                'Pneumonia': 0.5        # Moderate threshold - treatable
            },
            
            # Standard confidence levels
            'confidence_levels': {
                'high_confidence': 0.8,     # Very confident
                'medium_confidence': 0.6,   # Moderately confident  
                'low_confidence': 0.4,      # Low confidence
                'very_low': 0.2            # Very low confidence
            },
            
            # Urgency validation
            'urgency_validation': {
                'emergency': 5,      # Immediate intervention
                'urgent': 4,         # Within hours
                'semi_urgent': 3,    # Within day
                'routine': 2,        # Within week
                'normal': 1          # Normal finding
            }
        }
        
        # Medical consistency rules
        self.medical_rules = {
            'critical_combinations': {
                # Conditions that rarely occur together
                ('Pneumothorax', 'Effusion'): 0.1,  # Very rare combination
                ('Mass', 'Pneumonia'): 0.3,         # Possible but uncommon
            },
            
            'logical_combinations': {
                # Conditions that often occur together
                ('Cardiomegaly', 'Edema'): 0.8,     # Heart failure pattern
                ('Pneumonia', 'Consolidation'): 0.9, # Pneumonia causes consolidation
                ('Atelectasis', 'Effusion'): 0.6,   # Can occur together
            }
        }
        
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup QA message handlers"""
        
        @self.agent.on_message(model=QARequest)
        async def handle_qa_validation(ctx: Context, sender: str, msg: QARequest):
            """Perform comprehensive medical QA validation"""
            ctx.logger.info(f"🔍 Starting QA validation for {msg.request_id}")
            
            try:
                validation_results = await self.perform_medical_validation(
                    msg.triage_results,
                    msg.report_results,
                    msg.request_id
                )
                
                response = QAResponse(
                    request_id=msg.request_id,
                    validation_results=validation_results,
                    timestamp=datetime.now().isoformat()
                )
                
                ctx.logger.info(f"✅ QA validation completed for {msg.request_id}")
                await ctx.send(sender, response)
                
            except Exception as e:
                ctx.logger.error(f"❌ QA validation failed for {msg.request_id}: {e}")
                error_response = QAResponse(
                    request_id=msg.request_id,
                    validation_results={},
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
                await ctx.send(sender, error_response)

    async def perform_medical_validation(self, triage_results: Dict[str, Any], 
                                       report_results: Dict[str, Any], 
                                       request_id: str) -> Dict[str, Any]:
        """Comprehensive medical validation pipeline"""
        
        # Extract key metrics
        confidence_score = triage_results.get('confidence_score', 0.0)
        urgency_score = triage_results.get('urgency_score', 1)
        critical_findings = triage_results.get('critical_findings', [])
        all_findings = triage_results.get('all_findings', [])
        predictions = triage_results.get('predictions', {})
        
        # Perform validation checks
        medical_consistency = self.validate_medical_consistency(predictions, all_findings)
        confidence_validation = self.validate_confidence_appropriateness(predictions, all_findings)
        urgency_validation = self.validate_urgency_logic(urgency_score, critical_findings, all_findings)
        report_validation = self.validate_report_medical_accuracy(report_results, triage_results)
        safety_validation = self.validate_patient_safety(critical_findings, urgency_score)
        
        # Calculate overall validation score
        validation_scores = [
            medical_consistency['score'],
            confidence_validation['score'], 
            urgency_validation['score'],
            report_validation['score'],
            safety_validation['score']
        ]
        overall_score = sum(validation_scores) / len(validation_scores)
        
        # Generate recommendations
        recommendations = self.generate_medical_recommendations(
            overall_score, urgency_score, critical_findings, validation_scores
        )
        
        # Determine review requirements
        review_required = self.determine_medical_review_requirement(
            overall_score, urgency_score, critical_findings, confidence_score
        )
        
        return {
            'overall_validation_score': overall_score,
            'medical_consistency': medical_consistency,
            'confidence_validation': confidence_validation,
            'urgency_validation': urgency_validation,
            'report_validation': report_validation,
            'safety_validation': safety_validation,
            'recommendations': recommendations,
            'manual_review_required': review_required,
            'quality_flags': self.generate_medical_quality_flags(triage_results, report_results),
            'validation_summary': self.generate_medical_summary(overall_score, review_required),
            'patient_safety_score': safety_validation['score']
        }

    def validate_medical_consistency(self, predictions: Dict[str, float], 
                                   all_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate medical logic and consistency"""
        consistency_score = 1.0
        flags = []
        
        significant_conditions = [
            f['condition'] for f in all_findings 
            if f.get('confidence', 0) > 0.5
        ]
        
        # Check for impossible combinations
        for (cond1, cond2), max_prob in self.medical_rules['critical_combinations'].items():
            if cond1 in significant_conditions and cond2 in significant_conditions:
                conf1 = predictions.get(cond1, 0)
                conf2 = predictions.get(cond2, 0)
                if conf1 > 0.6 and conf2 > 0.6:
                    consistency_score -= 0.3
                    flags.append(f"Unlikely combination: {cond1} + {cond2}")
        
        # Check for expected combinations
        pneumonia_conf = predictions.get('Pneumonia', 0)
        consolidation_conf = predictions.get('Consolidation', 0)
        
        if pneumonia_conf > 0.7 and consolidation_conf < 0.3:
            consistency_score -= 0.2
            flags.append("Pneumonia without consolidation - unusual pattern")
        
        # Check severity consistency
        mass_conf = predictions.get('Mass', 0)
        if mass_conf > 0.8:
            # Mass should trigger high urgency
            urgency_expected = True
        
        return {
            'score': max(0.0, consistency_score),
            'flags': flags,
            'consistent_combinations': len([c for c in self.medical_rules['logical_combinations'] if all(cond in significant_conditions for cond in c)])
        }

    def validate_confidence_appropriateness(self, predictions: Dict[str, float], 
                                          all_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate if confidence levels are medically appropriate"""
        confidence_score = 0.8
        flags = []
        
        high_conf_count = 0
        medium_conf_count = 0
        critical_high_conf = 0
        
        for condition, confidence in predictions.items():
            if confidence > self.medical_thresholds['confidence_levels']['high_confidence']:
                high_conf_count += 1
                if condition in self.medical_thresholds['critical_conditions']:
                    critical_high_conf += 1
            elif confidence > self.medical_thresholds['confidence_levels']['medium_confidence']:
                medium_conf_count += 1
        
        # Check for overconfidence
        if high_conf_count > 3:
            confidence_score -= 0.2
            flags.append("Unusually high confidence in multiple conditions")
        
        # Check critical condition confidence
        for condition in self.medical_thresholds['critical_conditions']:
            conf = predictions.get(condition, 0)
            threshold = self.medical_thresholds['critical_conditions'][condition]
            
            if conf > threshold and conf < 0.6:
                flags.append(f"{condition} near threshold - consider manual review")
        
        return {
            'score': confidence_score,
            'flags': flags,
            'high_confidence_count': high_conf_count,
            'critical_high_confidence': critical_high_conf,
            'distribution_appropriate': high_conf_count <= 2
        }

    def validate_urgency_logic(self, urgency_score: int, critical_findings: List[str], 
                             all_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate urgency scoring logic"""
        urgency_validation_score = 0.9
        flags = []
        
        # Check urgency consistency with findings
        has_pneumothorax = any('Pneumothorax' in finding for finding in critical_findings)
        has_mass = any('Mass' in finding for finding in critical_findings)
        has_pneumonia = any('Pneumonia' in finding for finding in critical_findings)
        
        # Pneumothorax should always be urgent (4-5)
        if has_pneumothorax and urgency_score < 4:
            urgency_validation_score -= 0.3
            flags.append("Pneumothorax detected but urgency score too low")
        
        # Mass should be high priority (3-4)
        if has_mass and urgency_score < 3:
            urgency_validation_score -= 0.2
            flags.append("Mass detected but urgency score insufficient")
        
        # High urgency without critical findings
        if urgency_score >= 4 and not critical_findings:
            urgency_validation_score -= 0.2
            flags.append("High urgency without identifiable critical findings")
        
        # Normal urgency with critical findings
        if urgency_score <= 2 and critical_findings:
            urgency_validation_score -= 0.4
            flags.append("Critical findings present but urgency score too low")
        
        return {
            'score': max(0.0, urgency_validation_score),
            'flags': flags,
            'urgency_appropriate': len(flags) == 0,
            'expected_urgency_range': self.calculate_expected_urgency(critical_findings)
        }

    def validate_report_medical_accuracy(self, report_results: Dict[str, Any], 
                                       triage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate report medical accuracy"""
        report_score = 0.85
        flags = []
        
        findings_text = report_results.get('findings', '').lower()
        impression_text = report_results.get('impression', '').lower()
        critical_findings = triage_results.get('critical_findings', [])
        
        # Check if critical findings are mentioned in report
        for critical_finding in critical_findings:
            condition = critical_finding.split(' (')[0].lower()
            
            if condition not in findings_text and condition not in impression_text:
                report_score -= 0.3
                flags.append(f"Critical finding '{condition}' not adequately described in report")
        
        # Check for appropriate urgency language
        urgency_score = triage_results.get('urgency_score', 1)
        
        if urgency_score >= 5:
            if 'critical' not in impression_text and 'emergency' not in impression_text:
                report_score -= 0.2
                flags.append("Critical urgency case lacks appropriate language")
        
        # Check report completeness
        required_sections = ['indication', 'comparison', 'findings', 'impression']
        missing_sections = [s for s in required_sections if not report_results.get(s)]
        
        if missing_sections:
            report_score -= 0.1 * len(missing_sections)
            flags.extend([f"Missing {section} section" for section in missing_sections])
        
        return {
            'score': max(0.0, report_score),
            'flags': flags,
            'sections_complete': len(required_sections) - len(missing_sections),
            'critical_findings_documented': len(critical_findings) - len([f for f in flags if 'not adequately described' in f])
        }

    def validate_patient_safety(self, critical_findings: List[str], urgency_score: int) -> Dict[str, Any]:
        """Validate patient safety considerations"""
        safety_score = 1.0
        safety_flags = []
        
        # Check for missed critical conditions
        has_pneumothorax = any('Pneumothorax' in finding for finding in critical_findings)
        has_mass = any('Mass' in finding for finding in critical_findings)
        
        # Pneumothorax safety check
        if has_pneumothorax:
            if urgency_score < 4:
                safety_score -= 0.5
                safety_flags.append("SAFETY ALERT: Pneumothorax may need immediate intervention")
        
        # Mass safety check
        if has_mass:
            if urgency_score < 3:
                safety_score -= 0.3
                safety_flags.append("SAFETY ALERT: Mass requires prompt follow-up")
        
        # Multiple critical findings
        if len(critical_findings) >= 2:
            safety_flags.append("Multiple critical findings - enhanced monitoring recommended")
        
        return {
            'score': max(0.0, safety_score),
            'flags': safety_flags,
            'safety_alerts': len([f for f in safety_flags if 'SAFETY ALERT' in f]),
            'patient_risk_level': self.assess_patient_risk(critical_findings, urgency_score)
        }

    def calculate_expected_urgency(self, critical_findings: List[str]) -> tuple:
        """Calculate expected urgency range"""
        if any('Pneumothorax' in finding for finding in critical_findings):
            return (4, 5)
        elif any('Mass' in finding for finding in critical_findings):
            return (3, 4)
        elif any('Pneumonia' in finding for finding in critical_findings):
            return (3, 4)
        elif critical_findings:
            return (2, 3)
        else:
            return (1, 2)

    def assess_patient_risk(self, critical_findings: List[str], urgency_score: int) -> str:
        """Assess overall patient risk level"""
        if urgency_score >= 5 or any('Pneumothorax' in f for f in critical_findings):
            return "HIGH"
        elif urgency_score >= 4 or any('Mass' in f or 'Pneumonia' in f for f in critical_findings):
            return "MODERATE"
        elif urgency_score >= 3 or critical_findings:
            return "LOW"
        else:
            return "MINIMAL"

    def generate_medical_recommendations(self, overall_score: float, urgency_score: int, 
                                       critical_findings: List[str], validation_scores: List[float]) -> List[str]:
        """Generate medical QA recommendations"""
        recommendations = []
        
        # Overall quality recommendations
        if overall_score < 0.7:
            recommendations.append("Manual radiologist review recommended due to validation concerns")
        
        # Urgency-based recommendations
        if urgency_score >= 5:
            recommendations.append("URGENT: Immediate clinical attention required")
        elif urgency_score >= 4:
            recommendations.append("HIGH PRIORITY: Clinical evaluation within hours")
        elif urgency_score >= 3:
            recommendations.append("MODERATE PRIORITY: Clinical correlation within 24 hours")
        
        # Critical finding recommendations
        if any('Pneumothorax' in finding for finding in critical_findings):
            recommendations.append("EMERGENCY: Possible pneumothorax - immediate clinical assessment")
        
        if any('Mass' in finding for finding in critical_findings):
            recommendations.append("FOLLOW-UP: Pulmonary mass identified - oncology consultation")
        
        # Quality-based recommendations
        if min(validation_scores) < 0.6:
            recommendations.append("Quality concern identified - consider repeat analysis")
        
        if not recommendations:
            recommendations.append("Analysis meets quality standards - routine workflow")
        
        return recommendations

    def determine_medical_review_requirement(self, overall_score: float, urgency_score: int, 
                                           critical_findings: List[str], confidence_score: float) -> bool:
        """Determine if manual medical review is required"""
        return (
            overall_score < 0.75 or          # Quality concerns
            urgency_score >= 4 or            # High urgency cases
            len(critical_findings) > 0 or    # Any critical findings
            confidence_score < 0.6           # Low confidence
        )

    def generate_medical_quality_flags(self, triage_results: Dict[str, Any], 
                                     report_results: Dict[str, Any]) -> List[str]:
        """Generate medical quality flags"""
        flags = []
        
        confidence_score = triage_results.get('confidence_score', 0.0)
        urgency_score = triage_results.get('urgency_score', 1)
        critical_findings = triage_results.get('critical_findings', [])
        
        if confidence_score < 0.5:
            flags.append("LOW_CONFIDENCE")
        
        if urgency_score >= 5:
            flags.append("EMERGENCY_CASE")
        elif urgency_score >= 4:
            flags.append("HIGH_URGENCY")
        
        if critical_findings:
            flags.append("CRITICAL_FINDINGS")
        
        if len(critical_findings) >= 2:
            flags.append("MULTIPLE_CRITICAL")
        
        if not report_results.get('findings') or len(report_results.get('findings', '')) < 50:
            flags.append("INCOMPLETE_REPORT")
        
        return flags

    def generate_medical_summary(self, overall_score: float, review_required: bool) -> str:
        """Generate medical validation summary"""
        if overall_score >= 0.9:
            quality_level = "Excellent"
        elif overall_score >= 0.8:
            quality_level = "Good"
        elif overall_score >= 0.7:
            quality_level = "Acceptable"
        elif overall_score >= 0.6:
            quality_level = "Marginal"
        else:
            quality_level = "Poor"
        
        review_status = "Radiologist review recommended" if review_required else "AI analysis sufficient"
        
        return f"Medical QA: {quality_level} (Score: {overall_score:.2f}). {review_status}."

    def run(self):
        """Start the QA agent"""
        print(f"✅ Starting Medical QA Agent...")
        print(f"📍 Agent address: {self.agent.address}")
        print(f"🏥 Medical validation protocols active")
        print(f"⚕️  Patient safety monitoring enabled")
        print("=" * 50)
        self.agent.run()

if __name__ == "__main__":
    qa_agent = MedicalQAAgent()
    qa_agent.run()