'use client';

import { AnalysisResult } from '@/types';

interface ReportDisplayProps {
  analysis: AnalysisResult | null;
  isLoading: boolean;
}

export default function ReportDisplay({ analysis, isLoading }: ReportDisplayProps) {
  if (isLoading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-6">Analysis Results</h2>
        <div className="space-y-4">
          <div className="animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          </div>
          <div className="animate-pulse">
            <div className="h-20 bg-gray-200 rounded"></div>
          </div>
          <div className="animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-2/3 mb-2"></div>
            <div className="h-4 bg-gray-200 rounded w-1/3"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-6">Analysis Results</h2>
        <div className="text-center py-12">
          <div className="w-16 h-16 mx-auto bg-gray-100 rounded-full flex items-center justify-center mb-4">
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <p className="text-gray-500 text-lg">Upload a chest X-ray to see AI analysis results</p>
          <p className="text-gray-400 text-sm mt-2">Our AI agents will provide detailed findings and impressions</p>
        </div>
      </div>
    );
  }

  const getUrgencyColor = (urgency: number) => {
    if (urgency >= 4) return 'bg-red-100 text-red-800 border-red-200';
    if (urgency >= 3) return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    return 'bg-green-100 text-green-800 border-green-200';
  };

  const getUrgencyLabel = (urgency: number) => {
    if (urgency >= 4) return 'High Priority';
    if (urgency >= 3) return 'Medium Priority';
    return 'Normal';
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-gray-900">Analysis Results</h2>
        <div className="text-sm text-gray-500">
          {new Date(analysis.timestamp).toLocaleString()}
        </div>
      </div>

      {/* Urgency and Confidence Indicators */}
      <div className="grid grid-cols-2 gap-4">
        <div className={`p-3 rounded-lg border ${getUrgencyColor(analysis.urgency)}`}>
          <div className="text-sm font-medium">Priority Level</div>
          <div className="text-lg font-bold">{getUrgencyLabel(analysis.urgency)}</div>
        </div>
        <div className="p-3 rounded-lg border bg-blue-50 text-blue-800 border-blue-200">
          <div className="text-sm font-medium">Confidence</div>
          <div className="text-lg font-bold">{(analysis.confidence * 100).toFixed(1)}%</div>
        </div>
      </div>

      {/* Quick Findings */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h3 className="font-semibold text-gray-900 mb-3">Key Findings</h3>
        <ul className="space-y-2">
          {analysis.findings.map((finding, index) => (
            <li key={index} className="flex items-start space-x-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
              <span className="text-gray-700 text-sm">{finding}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Detailed Report */}
      <div className="space-y-4">
        <h3 className="font-semibold text-gray-900">Radiology Report</h3>
        
        <div className="space-y-4">
          <div className="border-l-4 border-blue-500 pl-4 py-2 bg-blue-50">
            <h4 className="font-medium text-blue-900 mb-1">INDICATION</h4>
            <p className="text-blue-800 text-sm">{analysis.report.indication}</p>
          </div>

          <div className="border-l-4 border-green-500 pl-4 py-2 bg-green-50">
            <h4 className="font-medium text-green-900 mb-1">COMPARISON</h4>
            <p className="text-green-800 text-sm">{analysis.report.comparison}</p>
          </div>

          <div className="border-l-4 border-purple-500 pl-4 py-2 bg-purple-50">
            <h4 className="font-medium text-purple-900 mb-1">FINDINGS</h4>
            <p className="text-purple-800 text-sm leading-relaxed">{analysis.report.findings}</p>
          </div>

          <div className="border-l-4 border-orange-500 pl-4 py-2 bg-orange-50">
            <h4 className="font-medium text-orange-900 mb-1">IMPRESSION</h4>
            <p className="text-orange-800 text-sm font-medium">{analysis.report.impression}</p>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-3 pt-4 border-t">
        <button className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium">
          Download Report
        </button>
        <button className="flex-1 border border-gray-300 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-50 transition-colors text-sm font-medium">
          Share Results
        </button>
      </div>

      {/* Analysis ID */}
      <div className="text-xs text-gray-400 text-center pt-2 border-t">
        Analysis ID: {analysis.id}
      </div>
    </div>
  );
}