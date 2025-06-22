// API client functions for XightMD

import { AnalysisResult, ApiResponse, HealthStatus } from '@/types';

class ApiClient {
  private baseUrl: string;

  constructor() {
    // Use environment variable with fallback
    this.baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
    
    // Remove trailing slash if present
    this.baseUrl = this.baseUrl.replace(/\/$/, '');
    
    console.log('🔧 API Client initialized with baseUrl:', this.baseUrl);
  }

  async analyzeImage(file: File): Promise<ApiResponse<AnalysisResult>> {
    try {
      console.log('🔍 Sending analysis request to:', `${this.baseUrl}/api/analyze`);
      
      const formData = new FormData();
      formData.append('image', file);

      const response = await fetch(`${this.baseUrl}/api/analyze`, {
        method: 'POST',
        body: formData,
        // Add headers for CORS if needed
        headers: {
          // Don't set Content-Type for FormData - browser will set it with boundary
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('✅ Analysis response received:', result);
      return result;
    } catch (error) {
      console.error('❌ Analysis API error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      };
    }
  }

  async getHealthStatus(): Promise<ApiResponse<HealthStatus>> {
    try {
      console.log('🏥 Checking health at:', `${this.baseUrl}/api/health`);
      
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
      console.error('❌ Health check API error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Health check failed'
      };
    }
  }

  async getAgentStatus(): Promise<ApiResponse<any>> {
    try {
      console.log('🤖 Checking agent status at:', `${this.baseUrl}/api/agent-status`);
      
      const response = await fetch(`${this.baseUrl}/api/agent-status`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('📊 Agent status response:', result);
      return result;
    } catch (error) {
      console.error('❌ Agent status API error:', error);
      
      // Return mock data for development/fallback
      return {
        success: true,
        data: {
          agents: {
            coordinator: { status: 'offline', lastSeen: new Date().toISOString() },
            triage: { status: 'offline', lastSeen: new Date().toISOString() },
            report: { status: 'offline', lastSeen: new Date().toISOString() },
            qa: { status: 'offline', lastSeen: new Date().toISOString() }
          }
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

  // Helper method to check if we're using production API
  isProduction(): boolean {
    return this.baseUrl.includes('railway.app');
  }

  // Get current API info
  getApiInfo() {
    return {
      baseUrl: this.baseUrl,
      environment: process.env.NEXT_PUBLIC_ENV || 'development',
      isProduction: this.isProduction()
    };
  }
}

// Export singleton instance
export const apiClient = new ApiClient();

// Rest of your utility functions remain the same...
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
  // Your existing downloadReport function...
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