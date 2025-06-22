export interface FindingResult {
  condition: string;
  confidence: number;
  confidence_level: string;
}

export interface SignificantFinding {
  confidence: number;
  significant: boolean;
  confidence_level: string;
}

export interface ModelAnalysis {
  predictions: Record<string, number>;
  significant_findings: Record<string, SignificantFinding>;
  urgency_level: number;
  top_findings: FindingResult[];
  has_critical_findings: boolean;
}

export interface Report {
  indication: string;
  comparison: string;
  findings: string;
  impression: string;
}

export interface ModelInfo {
  primary_model: string;
  report_generator: string;
  pipeline: string;
}

export interface AnalysisResult {
  id: string;
  timestamp: string;
  urgency: number;
  confidence: number;
  findings: string[];
  report: Report;
  image: string;
  model_analysis: ModelAnalysis;
  model_info: ModelInfo;
  pdf_data?: string;
}