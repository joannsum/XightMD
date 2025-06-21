// API client functions for XightMD

import { AnalysisResult, ApiResponse, HealthStatus } from '@/types';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = '') {
    this.baseUrl = baseUrl;
  }

  async analyzeImage(file: File): Promise<ApiResponse<AnalysisResult>> {
    try {
      const formData = new FormData();
      formData.append('image', file);

      const response = await fetch(`${this.baseUrl}/api/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Analysis API error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      };
    }
  }

  async getHealthStatus(): Promise<ApiResponse<HealthStatus>> {
    try {
      const response = await fetch(`${this.baseUrl}/api/health`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return {
        success: true,
        data
      };
    } catch (error) {
      console.error('Health check API error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Health check failed'
      };
    }
  }

  async getAgentStatus(): Promise<ApiResponse<any>> {
    try {
      // This would typically call your backend agent status endpoint
      const response = await fetch(`${this.baseUrl}/api/agents/status`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Agent status API error:', error);
      // Return mock data for development
      return {
        success: true,
        data: {
          coordinator: { status: 'active', lastSeen: new Date().toISOString() },
          triage: { status: 'active', lastSeen: new Date().toISOString() },
          report: { status: 'active', lastSeen: new Date().toISOString() },
          qa: { status: 'active', lastSeen: new Date().toISOString() }
        }
      };
    }
  }

  async submitFeedback(analysisId: string, feedback: {
    rating: number;
    comments?: string;
    corrections?: any;
  }): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${this.baseUrl}/api/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          analysisId,
          ...feedback,
          timestamp: new Date().toISOString()
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Feedback API error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to submit feedback'
      };
    }
  }
}

// Export singleton instance
export const apiClient = new ApiClient();

// Utility functions
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const validateImageFile = (file: File): { valid: boolean; error?: string } => {
  // Check file type
  if (!file.type.startsWith('image/')) {
    return { valid: false, error: 'Please select an image file' };
  }

  // Check file size (max 10MB)
  const maxSize = 10 * 1024 * 1024;
  if (file.size > maxSize) {
    return { valid: false, error: 'File size must be less than 10MB' };
  }

  // Check supported formats
  const supportedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
  if (!supportedTypes.includes(file.type)) {
    return { valid: false, error: 'Supported formats: PNG, JPG, JPEG, WebP' };
  }

  return { valid: true };
};

export const downloadReport = (analysis: AnalysisResult, format: 'json' | 'txt' = 'txt'): void => {
  let content: string;
  let filename: string;
  let mimeType: string;

  if (format === 'json') {
    content = JSON.stringify(analysis, null, 2);
    filename = `xray-analysis-${analysis.id}.json`;
    mimeType = 'application/json';
  } else {
    content = `CHEST X-RAY ANALYSIS REPORT
Generated: ${new Date(analysis.timestamp).toLocaleString()}
Analysis ID: ${analysis.id}
Confidence: ${(analysis.confidence * 100).toFixed(1)}%
Priority: ${analysis.urgency >= 4 ? 'High' : analysis.urgency >= 3 ? 'Medium' : 'Normal'}

INDICATION:
${analysis.report.indication}

COMPARISON:
${analysis.report.comparison}

FINDINGS:
${analysis.report.findings}

IMPRESSION:
${analysis.report.impression}

KEY FINDINGS:
${analysis.findings.map(f => `• ${f}`).join('\n')}

---
Generated by XightMD AI Analysis System
`;
    filename = `xray-analysis-${analysis.id}.txt`;
    mimeType = 'text/plain';
  }

  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};