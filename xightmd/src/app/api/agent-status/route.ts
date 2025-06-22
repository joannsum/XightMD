// src/api/agent-status/route.ts
import { NextRequest, NextResponse } from 'next/server';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function GET(request: NextRequest) {
  try {
    console.log('🔍 Fetching agent status from backend...');
    
    // Use the FastAPI backend's agent status endpoint
    const response = await fetch(`${API_BASE_URL}/api/agents/status`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(5000) // 5 second timeout
    });

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`);
    }

    const data = await response.json();
    console.log('✅ Received agent status:', data);
    
    if (data.success && data.agents) {
      return NextResponse.json({
        success: true,
        agents: data.agents,
        timestamp: data.timestamp || new Date().toISOString()
      });
    } else {
      throw new Error('Backend returned unsuccessful response');
    }

  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    console.error('❌ Error checking agent status:', errorMessage);
    
    // Fallback to showing backend as offline
    const fallbackStatus = {
      coordinator: {
        status: 'offline',
        lastSeen: '',
        details: { error: 'Unable to connect to backend' }
      },
      triage: {
        status: 'offline', 
        lastSeen: '',
        details: { error: 'Backend unavailable' }
      },
      report: {
        status: 'offline',
        lastSeen: '',
        details: { error: 'Backend unavailable' }
      },
      qa: {
        status: 'offline',
        lastSeen: '',
        details: { error: 'Backend unavailable' }
      }
    };

    return NextResponse.json({
      success: false,
      error: `Failed to check agent status: ${errorMessage}`,
      agents: fallbackStatus,
      timestamp: new Date().toISOString()
    });
  }
}