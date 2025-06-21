'use client';

import { useState, useEffect } from 'react';
import ImageUpload from '@/components/ImageUpload';
import ReportDisplay from '@/components/ReportDisplay';
import AgentStatus from '@/components/AgentStatus';
import Dashboard from '@/components/Dashboard';
import { AnalysisResult, AgentStatuses } from '@/types';

export default function Home() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisResult | null>(null);
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisResult[]>([]);
  const [agentStatuses, setAgentStatuses] = useState<AgentStatuses>({
    coordinator: { status: 'active', lastSeen: new Date() },
    triage: { status: 'active', lastSeen: new Date() },
    report: { status: 'active', lastSeen: new Date() },
    qa: { status: 'active', lastSeen: new Date() }
  });

  // Simulate agent status updates
  useEffect(() => {
    const interval = setInterval(() => {
      setAgentStatuses(prev => ({
        ...prev,
        coordinator: { ...prev.coordinator, lastSeen: new Date() },
        triage: { ...prev.triage, lastSeen: new Date() },
        report: { ...prev.report, lastSeen: new Date() },
        qa: { ...prev.qa, lastSeen: new Date() }
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const handleImageUpload = async (file: File) => {
    setIsAnalyzing(true);
    setCurrentAnalysis(null);

    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('image', file);

      // Simulate API call to backend
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const result = await response.json();
      
      // For demo purposes, create mock result if API not available
      const mockResult: AnalysisResult = {
        id: `analysis-${Date.now()}`,
        timestamp: new Date().toISOString(),
        urgency: Math.floor(Math.random() * 5) + 1,
        confidence: Math.random() * 0.3 + 0.7,
        findings: ['No acute cardiopulmonary abnormality', 'Heart size normal', 'Lungs clear'],
        report: {
          indication: 'Chest pain, rule out pneumonia',
          comparison: 'No prior studies available for comparison',
          findings: 'The heart size is normal. The lungs are clear bilaterally without evidence of consolidation, pleural effusion, or pneumothorax. The mediastinal and hilar contours appear normal. No acute osseous abnormalities are identified.',
          impression: 'No acute cardiopulmonary abnormality. Normal chest radiograph.'
        },
        image: URL.createObjectURL(file)
      };

      setCurrentAnalysis(result.data || mockResult);
      setAnalysisHistory(prev => [result.data || mockResult, ...prev.slice(0, 9)]);
    } catch (error) {
      console.error('Analysis failed:', error);
      // Handle error appropriately
    } finally {
      setIsAnalyzing(false);
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
            <AgentStatus agents={agentStatuses} />
          </div>
        </div>
      </header>

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