// src/types/index.ts

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  services: {
    api: 'up' | 'down';
    agents: 'up' | 'down' | 'partial';
    database?: 'up' | 'down';
  };
  uptime?: number;
  version?: string;
}

export interface AnalysisResult {
  id: string;
  timestamp: string;
  urgency: number; // 1-5 scale
  confidence: number; // 0-1 scale
  findings: string[];
  report: {
    indication: string;
    comparison: string;
    findings: string;
    impression: string;
  };
  image: string; // base64 data URL
  pdf_data?: string; // base64 PDF data (optional)
  hf_analysis?: string; // Raw Hugging Face output
  model_info?: {
    primary_model: string;
    report_generator: string;
    processing_pipeline: string;
  };
}

export interface AgentStatuses {
  coordinator: AgentStatus;
  triage: AgentStatus;
  report: AgentStatus;
  qa: AgentStatus;
}

export interface AgentStatus {
  status: 'active' | 'idle' | 'error' | 'offline';
  lastSeen: Date | string;
  details?: any;
}

// Hugging Face specific types
export interface HuggingFaceConfig {
  model: string;
  parameters: {
    max_length: number;
    min_length: number;
    do_sample: boolean;
    temperature: number;
    top_p: number;
    repetition_penalty: number;
    return_full_text: boolean;
  };
  options: {
    wait_for_model: boolean;
    use_cache: boolean;
  };
}

export interface ModelStatus {
  ready: boolean;
  status: 'ready' | 'loading' | 'error' | 'checking';
  error?: string;
  estimated_time?: number;
  message?: string;
  model?: string;
}

export interface ProcessingInfo {
  hf_model: string;
  claude_model: string;
  timestamp: string;
}

// API Response types
export interface AnalysisAPIResponse {
  success: boolean;
  data?: AnalysisResult;
  processing_info?: ProcessingInfo;
  error?: string;
  details?: string;
}

export interface ModelStatusAPIResponse {
  ready: boolean;
  status: string;
  error?: string;
  estimated_time?: number;
  message?: string;
  model?: string;
}