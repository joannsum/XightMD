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
    coordinator: { status: 'offline', lastSeen: new Date() },
    triage: { status: 'offline', lastSeen: new Date() },
    report: { status: 'offline', lastSeen: new Date() },
    qa: { status: 'offline', lastSeen: new Date() }
  });

  // Fetch real agent status updates
  useEffect(() => {
    const fetchAgentStatus = async () => {
      try {
        console.log('🔍 Fetching agent status...');
        const response = await fetch('/api/agent-status');
        console.log('📡 Response status:', response.status);
        
        if (response.ok) {
          const data = await response.json();
          console.log('📊 Agent status response:', data);
          if (data.success && data.agents) {
            // Convert API response to the format expected by frontend
            const formattedAgents: AgentStatuses = {
              coordinator: {
                status: data.agents.coordinator?.status || 'offline',
                lastSeen: data.agents.coordinator?.lastSeen ? new Date(data.agents.coordinator.lastSeen) : new Date(),
                details: data.agents.coordinator?.details || {}
              },
              triage: {
                status: data.agents.triage?.status || 'offline', 
                lastSeen: data.agents.triage?.lastSeen ? new Date(data.agents.triage.lastSeen) : new Date(),
                details: data.agents.triage?.details || {}
              },
              report: {
                status: data.agents.report?.status || 'offline',
                lastSeen: data.agents.report?.lastSeen ? new Date(data.agents.report.lastSeen) : new Date(), 
                details: data.agents.report?.details || {}
              },
              qa: {
                status: data.agents.qa?.status || 'offline',
                lastSeen: data.agents.qa?.lastSeen ? new Date(data.agents.qa.lastSeen) : new Date(),
                details: data.agents.qa?.details || {}
              }
            };
            
            setAgentStatuses(formattedAgents);
            console.log('✅ Agent statuses updated:', formattedAgents);
            console.log('🔍 Active agents count:', Object.values(formattedAgents).filter(a => a.status === 'active').length);
          } else {
            console.warn('⚠️ API response missing success or agents data:', data);
          }
        } else {
          console.warn('⚠️ Agent status API returned non-OK status:', response.status);
          // Try to read error response
          try {
            const errorData = await response.text();
            console.error('Error response:', errorData);
          } catch (e) {
            console.error('Could not read error response');
          }
        }
      } catch (error) {
        console.error('❌ Failed to fetch agent status:', error);
        // Set all agents to offline on error
        setAgentStatuses(prev => ({
          coordinator: { ...prev.coordinator, status: 'offline' },
          triage: { ...prev.triage, status: 'offline' },
          report: { ...prev.report, status: 'offline' },
          qa: { ...prev.qa, status: 'offline' }
        }));
      }
    };

    // Initial fetch
    fetchAgentStatus();

    // Set up polling every 10 seconds (reduced frequency to avoid spam)
    const interval = setInterval(fetchAgentStatus, 10000);

    return () => clearInterval(interval);
  }, []);

  const handleImageUpload = async (file: File) => {
    setIsAnalyzing(true);
    setCurrentAnalysis(null);

    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('image', file);

      console.log('🔍 Sending image to backend for analysis...');
      
      // Call actual API
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log('✅ Analysis result received:', result);
      
      if (result.success && result.data) {
        // Use real API result
        setCurrentAnalysis(result.data);
        setAnalysisHistory(prev => [result.data, ...prev.slice(0, 9)]);
      } else {
        throw new Error('Invalid response format');
      }
      
    } catch (error) {
      console.error('❌ Analysis failed:', error);
      
      // Create fallback mock result only on error
      const mockResult: AnalysisResult = {
        id: `analysis-${Date.now()}`,
        timestamp: new Date().toISOString(),
        urgency: Math.floor(Math.random() * 5) + 1,
        confidence: Math.random() * 0.3 + 0.7,
        findings: ['Analysis failed - using mock data', 'Check server connection', 'Ensure backend is running'],
        report: {
          indication: 'System error occurred during analysis',
          comparison: 'No comparison available due to system error',
          findings: 'Unable to analyze image due to system error. Please check that the backend server and agent network are running properly.',
          impression: 'System maintenance required. Mock result displayed for demonstration purposes.'
        },
        image: URL.createObjectURL(file)
      };

      setCurrentAnalysis(mockResult);
      setAnalysisHistory(prev => [mockResult, ...prev.slice(0, 9)]);
      
      // Show user-friendly error
      alert('Analysis failed. Please check that the backend server is running. Showing mock data for demo purposes.');
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