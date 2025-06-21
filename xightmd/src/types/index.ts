// Shared types for XightMD application

export type AgentStatus = 'active' | 'idle' | 'error' | 'offline';

export interface AgentInfo {
  status: AgentStatus;
  lastSeen: Date;
}

export interface AgentStatuses {
  coordinator: AgentInfo;
  triage: AgentInfo;
  report: AgentInfo;
  qa: AgentInfo;
}

export interface AnalysisResult {
  id: string;
  timestamp: string;
  urgency: number;
  confidence: number;
  findings: string[];
  report: {
    indication: string;
    comparison: string;
    findings: string;
    impression: string;
  };
  image: string;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface HealthStatus {
  status: 'healthy' | 'unhealthy';
  timestamp: string;
  version: string;
  services: {
    frontend: { status: string; responseTime: string };
    backend: { status: string; responseTime: string; url: string };
    agents: {
      coordinator: { status: string; lastSeen: string };
      triage: { status: string; lastSeen: string };
      report: { status: string; lastSeen: string };
      qa: { status: string; lastSeen: string };
    };
    claude_api: { status: string; model: string };
  };
  uptime: number;
  memory: { used: number; total: number };
}