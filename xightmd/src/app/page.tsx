'use client';

import { useState, useEffect } from 'react';
import ImageUpload from '@/components/ImageUpload';
import ReportDisplay from '@/components/ReportDisplay';
import Dashboard from '@/components/Dashboard';
import { AnalysisResult } from '@/types';

export default function Home() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisResult | null>(null);
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisResult[]>([]);
  const [modelStatus, setModelStatus] = useState<'checking' | 'ready' | 'loading' | 'error'>('checking');

  useEffect(() => {
    const checkModelStatus = async () => {
      try {
        console.log('🔍 Checking local model status...');
        setModelStatus('checking');
        
        const response = await fetch('lung_classifier_best.pth');
        if (response.ok) {
          setModelStatus('ready');
          console.log('✅ Local model found');
        } else {
          console.warn('⚠️ Local model not found');
        setModelStatus('error');
      }
    } catch (error) {
        console.error('❌ Failed to check model status:', error);
        setModelStatus('error');
      }
      };

    checkModelStatus();
  }, []);

  const handleImageUpload = async (file: File) => {
    setIsAnalyzing(true);
    setCurrentAnalysis(null);

    try {
      const formData = new FormData();
      formData.append('image', file);

      console.log('🔍 Sending image for local model analysis...');

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
        setCurrentAnalysis(result.data);
        setAnalysisHistory(prev => [result.data, ...prev.slice(0, 9)]);
        console.log('📊 Model info:', result.data.model_info);
        console.log('🔍 Model analysis:', result.data.model_analysis);
      } else {
        throw new Error('Invalid response format');
      }
      
    } catch (error) {
      console.error('❌ Analysis failed:', error);

      alert(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
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
      case 'ready': return 'Local Model Ready';
      case 'loading': return 'Loading Model';
      case 'error': return 'Model Not Found';
      default: return 'Checking...';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg mr-3"></div>
              <h1 className="text-2xl font-bold text-gray-900">XightMD</h1>
              <span className="ml-2 text-sm text-gray-500">AI-Powered Chest X-ray Analysis</span>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 px-3 py-2 rounded-lg bg-gray-50">
                <div className={`w-3 h-3 rounded-full ${getModelStatusColor()} ${modelStatus === 'ready' ? 'animate-pulse' : ''}`}></div>
                <span className="text-sm font-medium text-gray-700">
                  {getModelStatusText()}
                </span>
              </div>
                </div>
          </div>
            </div>
      </header>

      {modelStatus === 'error' && (
        <div className="bg-red-50 border-b border-red-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
            <div className="flex items-center">
              <svg className="w-5 h-5 text-red-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-red-800 text-sm">
                <strong>Model not found.</strong> Please ensure lung_classifier_best.pth is in the correct location.
              </p>
            </div>
          </div>
        </div>
      )}

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
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