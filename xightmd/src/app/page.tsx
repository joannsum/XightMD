'use client';

import { useState, useEffect } from 'react';
import ImageUpload from '@/components/ImageUpload';
import ReportDisplay from '@/components/ReportDisplay';
import Dashboard from '@/components/Dashboard';
import { AnalysisResult } from '@/types';

interface ModelInfo {
  primary_model: string;
  report_generator: string;
  processing_pipeline: string;
}

interface ProcessingInfo {
  hf_model: string;
  gemini_model: string;
  timestamp: string;
}

export default function Home() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisResult | null>(null);
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisResult[]>([]);
  const [processingInfo, setProcessingInfo] = useState<ProcessingInfo | null>(null);
  const [modelStatus, setModelStatus] = useState<'checking' | 'ready' | 'loading' | 'error'>('checking');

  // Check model availability on startup
  useEffect(() => {
    const checkModelStatus = async () => {
      try {
        console.log('🔍 Checking Hugging Face model status...');
        setModelStatus('checking');
        
        // Test the Hugging Face API with a minimal request
        const response = await fetch('/api/model-status');
        
        if (response.ok) {
          const data = await response.json();
          setModelStatus(data.ready ? 'ready' : 'loading');
          console.log('✅ Model status:', data);
        } else {
          console.warn('⚠️ Model status check failed');
          setModelStatus('error');
        }
      } catch (error) {
        console.error('❌ Failed to check model status:', error);
        setModelStatus('error');
      }
    };

    checkModelStatus();
    
    // Check every 30 seconds if model is loading
    const interval = setInterval(() => {
      if (modelStatus === 'loading') {
        checkModelStatus();
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [modelStatus]);

  const handleImageUpload = async (file: File) => {
    setIsAnalyzing(true);
    setCurrentAnalysis(null);

    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('image', file);

      console.log('🔍 Sending image to Hugging Face + Claude pipeline...');
      
      // Call updated API
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Analysis failed: ${response.status}`);
      }

      const result = await response.json();
      console.log('✅ Analysis result received:', result);
      
      if (result.success && result.data) {
        // Store processing info
        setProcessingInfo(result.processing_info);
        
        // Use real API result
        // setCurrentAnalysis(result.data);
        const firstLabel = result.data?.hf_analysis?.split(',')[0]?.trim().toLowerCase();
        const pneumoniaDetected = firstLabel === 'pneumonia';
        
        const simplifiedResult: AnalysisResult = {
          id: `analysis-${Date.now()}`,
          timestamp: new Date().toISOString(),
          urgency: pneumoniaDetected ? 4 : 1,
          confidence: pneumoniaDetected ? 0.95 : 0.9,
          findings: [pneumoniaDetected ? 'Pneumonia detected' : 'No pneumonia detected'],
          report: {
            indication: 'Pneumonia screening',
            comparison: 'N/A',
            findings: pneumoniaDetected ? 'Signs consistent with pneumonia were detected.' : 'No radiographic evidence of pneumonia.',
            impression: pneumoniaDetected ? 'Positive for pneumonia' : 'Negative for pneumonia'
          },
          image: URL.createObjectURL(file),
          model_info: {
            primary_model: 'nickmuchi/vit-finetuned-chest-xray-pneumonia',
            report_generator: 'Gemini Flash 2.0',
            processing_pipeline: 'Gemini Flash 2.0 Only'
          }
        };

        setCurrentAnalysis(simplifiedResult);
        setAnalysisHistory(prev => [simplifiedResult, ...prev.slice(0, 9)]);

        setAnalysisHistory(prev => [result.data, ...prev.slice(0, 9)]);
        
        console.log('📊 Model info:', result.data.model_info);
        console.log('🤗 HF Analysis:', result.data.hf_analysis?.substring(0, 100) + '...');
      } else {
        throw new Error('Invalid response format');
      }
      
    } catch (error) {
      console.error('❌ Analysis failed:', error);
      
      // Create fallback mock result only on error
      const mockResult: AnalysisResult = {
        id: `analysis-${Date.now()}`,
        timestamp: new Date().toISOString(),
        urgency: 2,
        confidence: 0.0,
        findings: [
          'Analysis failed - check API keys and model availability', 
          'Ensure HUGGINGFACE_API_KEY is set in environment',
          'Ensure ANTHROPIC_API_KEY is set for Claude integration'
        ],
        report: {
          indication: 'System error occurred during analysis',
          comparison: 'No comparison available due to system error',
          findings: 'Unable to analyze image due to API error. Please check that the Hugging Face and Claude API keys are properly configured.',
          impression: 'API configuration required. Mock result displayed for demonstration purposes.'
        },
        image: URL.createObjectURL(file),
        model_info: {
          primary_model: 'microsoft/BiomedVLP-CXR-BERT-general (unavailable)',
          report_generator: 'gemini (unavailable)', 
          processing_pipeline: 'Error - API keys required'
        }
      };

      setCurrentAnalysis(mockResult);
      setAnalysisHistory(prev => [mockResult, ...prev.slice(0, 9)]);
      
      // Show user-friendly error
      alert(`Analysis failed: Please check API configuration. Showing mock data for demo purposes.`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getModelStatusColor = () => {
    switch (modelStatus) {
      case 'ready': return 'bg-green-500';
      case 'loading': return 'bg-yellow-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getModelStatusText = () => {
    switch (modelStatus) {
      case 'ready': return 'Models Ready';
      case 'loading': return 'Models Loading';
      case 'error': return 'API Error';
      default: return 'Checking...';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg mr-3"></div>
              <h1 className="text-2xl font-bold text-gray-900">XightMD</h1>
              <span className="ml-2 text-sm text-gray-500">AI-Powered Chest X-ray Analysis</span>
            </div>
            
            {/* Model Status Indicator */}
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 px-3 py-2 rounded-lg bg-gray-50">
                <div className={`w-3 h-3 rounded-full ${getModelStatusColor()} ${modelStatus === 'ready' ? 'animate-pulse' : ''}`}></div>
                <span className="text-sm font-medium text-gray-700">
                  {getModelStatusText()}
                </span>
              </div>
              
              {processingInfo && (
                <div className="text-xs text-gray-500">
                  <div>HF: {processingInfo.hf_model.split('/')[1]}</div>
                  <div>Gemini: {processingInfo.gemini_model}</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Model Info Banner */}
      {currentAnalysis?.model_info && (
        <div className="bg-blue-50 border-b border-blue-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4 text-sm">
                <div>
                  <span className="font-medium text-blue-900">Primary Model:</span>
                  <span className="text-blue-700 ml-1">{currentAnalysis.model_info.primary_model}</span>
                </div>
                <div>
                  <span className="font-medium text-blue-900">Report Generator:</span>
                  <span className="text-blue-700 ml-1">{currentAnalysis.model_info.report_generator}</span>
                </div>
                <div>
                  <span className="font-medium text-blue-900">Pipeline:</span>
                  <span className="text-blue-700 ml-1">{currentAnalysis.model_info.processing_pipeline}</span>
                </div>
              </div>
              {currentAnalysis.hf_analysis && (
                <button 
                  onClick={() => alert(`Hugging Face Analysis:\n\n${currentAnalysis.hf_analysis}`)}
                  className="text-xs text-blue-600 hover:text-blue-800 font-medium"
                >
                  View Raw HF Output
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Model Status Warning */}
      {modelStatus === 'loading' && (
        <div className="bg-yellow-50 border-b border-yellow-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
            <div className="flex items-center">
              <svg className="w-5 h-5 text-yellow-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
              <p className="text-yellow-800 text-sm">
                <strong>Hugging Face model is loading.</strong> This may take a few minutes on first use. Analysis will be faster once the model is ready.
              </p>
            </div>
          </div>
        </div>
      )}

      {modelStatus === 'error' && (
        <div className="bg-red-50 border-b border-red-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
            <div className="flex items-center">
              <svg className="w-5 h-5 text-red-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-red-800 text-sm">
                <strong>API Configuration Error.</strong> Please ensure HUGGINGFACE_API_KEY and ANTHROPIC_API_KEY are set in your environment.
              </p>
            </div>
          </div>
        </div>
      )}

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Upload and Dashboard */}
          <div className="lg:col-span-2 space-y-8">
            <ImageUpload 
              onUpload={handleImageUpload} 
              isAnalyzing={isAnalyzing} 
            />
            
            <Dashboard 
              analysisHistory={analysisHistory}
              onSelectAnalysis={setCurrentAnalysis}
            />
          </div>

          {/* Right Column - Results */}
          <div className="lg:col-span-1">
            <ReportDisplay 
              analysis={currentAnalysis}
              isLoading={isAnalyzing}
            />
          </div>
        </div>
      </main>
    </div>
  );
}