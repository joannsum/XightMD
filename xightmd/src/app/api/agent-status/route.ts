// src/api/agent-status/route.ts
import { NextRequest, NextResponse } from 'next/server';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function GET(request: NextRequest) {
  try {
    console.log('🔍 Fetching agent status from backend...');
    console.log(`📡 Backend URL: ${API_BASE_URL}/api/agents/status`);
    
    // Call the correct FastAPI backend endpoint
    const response = await fetch(`${API_BASE_URL}/api/agents/status`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      // Increase timeout and handle connection issues
      signal: AbortSignal.timeout(10000) // 10 second timeout
    });

    console.log(`📊 Backend response status: ${response.status}`);

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    console.log('✅ Received agent status from backend:', data);

    if (data.success && data.agents) {
      // Transform the backend response to match frontend expectations
      const transformedAgents = {
        coordinator: {
          status: data.agents.coordinator?.status || 'offline',
          lastSeen: data.agents.coordinator?.lastSeen || new Date().toISOString(),
          details: data.agents.coordinator?.details || {}
        },
        triage: {
          status: data.agents.triage?.status || 'offline',
          lastSeen: data.agents.triage?.lastSeen || new Date().toISOString(),
          details: data.agents.triage?.details || {}
        },
        report: {
          status: data.agents.report?.status || 'offline',
          lastSeen: data.agents.report?.lastSeen || new Date().toISOString(),
          details: data.agents.report?.details || {}
        },
        qa: {
          status: data.agents.qa?.status || 'offline',
          lastSeen: data.agents.qa?.lastSeen || new Date().toISOString(),
          details: data.agents.qa?.details || {}
        }
      };

      console.log('🔄 Transformed agent status:', transformedAgents);

      return NextResponse.json({
        success: true,
        agents: transformedAgents,
        timestamp: data.timestamp || new Date().toISOString(),
        network_health: data.network_health || 'unknown'
      });
    } else {
      throw new Error('Backend returned unsuccessful response or missing agents data');
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    console.error('❌ Error checking agent status:', errorMessage);

    // Check if it's a connection error
    const isConnectionError = errorMessage.includes('fetch') || 
                             errorMessage.includes('timeout') || 
                             errorMessage.includes('ECONNREFUSED');

    const errorType = isConnectionError ? 'connection_failed' : 'backend_error';

    // Fallback status when backend is unavailable
    const fallbackStatus = {
      coordinator: {
        status: 'offline',
        lastSeen: '',
        details: { 
          error: isConnectionError ? 'Cannot connect to backend server' : errorMessage,
          errorType: errorType
        }
      },
      triage: {
        status: 'offline',
        lastSeen: '',
        details: { 
          error: isConnectionError ? 'Backend server offline' : 'Backend error',
          errorType: errorType
        }
      },
      report: {
        status: 'offline',
        lastSeen: '',
        details: { 
          error: isConnectionError ? 'Backend server offline' : 'Backend error',
          errorType: errorType
        }
      },
      qa: {
        status: 'offline',
        lastSeen: '',
        details: { 
          error: isConnectionError ? 'Backend server offline' : 'Backend error',
          errorType: errorType
        }
      }
    };

    return NextResponse.json({
      success: false,
      error: `Failed to check agent status: ${errorMessage}`,
      agents: fallbackStatus,
      timestamp: new Date().toISOString(),
      network_health: 'offline'
    });
  }
}