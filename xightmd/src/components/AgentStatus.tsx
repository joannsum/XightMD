'use client';

import { useState } from 'react';
import { AgentStatuses } from '@/types';

interface AgentStatusProps {
  agents: AgentStatuses;
}

export default function AgentStatus({ agents }: AgentStatusProps) {
  const [showDetails, setShowDetails] = useState(false);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500';
      case 'idle': return 'bg-yellow-500';
      case 'error': return 'bg-red-500';
      case 'offline': return 'bg-gray-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'active': return 'Active';
      case 'idle': return 'Idle';
      case 'error': return 'Error';
      case 'offline': return 'Offline';
      default: return 'Unknown';
    }
  };

  const formatLastSeen = (lastSeen: Date | string) => {
    if (!lastSeen) return 'Never';
    
    try {
      // Convert string to Date if needed
      const date = typeof lastSeen === 'string' ? new Date(lastSeen) : lastSeen;
      
      // Check if date is valid
      if (isNaN(date.getTime())) return 'Unknown';
      
      return date.toLocaleTimeString();
    } catch (error) {
      console.warn('Error formatting lastSeen:', error);
      return 'Unknown';
    }
  };

  const agentDescriptions = {
    coordinator: 'Orchestrates analysis pipeline',
    triage: 'Analyzes X-rays and determines urgency',
    report: 'Generates structured radiology reports',
    qa: 'Validates analysis and ensures quality'
  };

  const agentNames = {
    coordinator: 'Coordinator Agent',
    triage: 'Triage Agent',
    report: 'Report Agent',
    qa: 'QA Agent'
  };

  const allAgentsActive = Object.values(agents).every(agent => agent.status === 'active');

  return (
    <div className="relative">
      {/* Status Indicator */}
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="flex items-center space-x-2 px-3 py-2 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors"
      >
        <div className={`w-3 h-3 rounded-full ${allAgentsActive ? 'bg-green-500' : 'bg-yellow-500'} animate-pulse`}></div>
        <span className="text-sm font-medium text-gray-700">
          Agent Network
        </span>
        <span className="text-xs text-gray-500">
          ({Object.values(agents).filter(a => a.status === 'active').length}/4)
        </span>
        <svg 
          className={`w-4 h-4 text-gray-500 transition-transform ${showDetails ? 'rotate-180' : ''}`} 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Detailed Status Panel */}
      {showDetails && (
        <div className="absolute right-0 top-full mt-2 w-80 bg-white rounded-xl shadow-xl border border-gray-200 p-4 z-50">
          <h3 className="font-semibold text-gray-900 mb-4">Agent Network Status</h3>
          
          <div className="space-y-3">
            {Object.entries(agents).map(([agentKey, agent]) => (
              <div key={agentKey} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${getStatusColor(agent.status)}`}></div>
                  <div>
                    <div className="font-medium text-gray-900 text-sm">
                      {agentNames[agentKey as keyof typeof agentNames]}
                    </div>
                    <div className="text-xs text-gray-500">
                      {agentDescriptions[agentKey as keyof typeof agentDescriptions]}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-xs font-medium ${
                    agent.status === 'active' ? 'text-green-600' : 
                    agent.status === 'idle' ? 'text-yellow-600' : 
                    agent.status === 'error' ? 'text-red-600' : 'text-gray-600'
                  }`}>
                    {getStatusText(agent.status)}
                  </div>
                  <div className="text-xs text-gray-400">
                    {formatLastSeen(agent.lastSeen)}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Network Health Summary */}
          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Network Health</span>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${allAgentsActive ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
                <span className={`text-sm font-medium ${allAgentsActive ? 'text-green-600' : 'text-yellow-600'}`}>
                  {allAgentsActive ? 'Optimal' : 'Degraded'}
                </span>
              </div>
            </div>
            
            {!allAgentsActive && (
              <p className="text-xs text-yellow-600 mt-2">
                Some agents are not active. Analysis may be slower.
              </p>
            )}
          </div>

          {/* Action Buttons */}
          <div className="mt-4 flex space-x-2">
            <button 
              onClick={() => window.location.reload()}
              className="flex-1 bg-blue-600 text-white py-2 px-3 rounded text-xs font-medium hover:bg-blue-700 transition-colors"
            >
              Refresh Status
            </button>
            <button 
              onClick={() => console.log('Agent details:', agents)}
              className="flex-1 border border-gray-300 text-gray-700 py-2 px-3 rounded text-xs font-medium hover:bg-gray-50 transition-colors"
            >
              View Details
            </button>
          </div>
        </div>
      )}
    </div>
  );
}