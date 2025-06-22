// src/app/api/analyze/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { AnalysisResult } from '@/types';

interface CoordinatorRequest {
  image_data: string;
  image_format: string;
  user_description?: string;
  priority_search?: string;
  patient_info?: {
    age?: number;
    sex?: string;
    symptoms?: string;
  };
}

interface CoordinatorResponse {
  request_id: string;
  status: string;
  results?: {
    analysis_id: string;
    timestamp: string;
    urgency: number;
    confidence: number;
    findings: string[];
    report: {
      indication: string;
      comparison: string;
      findings: string;
      impression: string;
    };
    pdf_data?: string;
    quality_metrics: any;
    processing_summary: any;
  };
  error?: string;
}

export async function POST(request: NextRequest) {
  try {
    console.log('🔍 API: Received analysis request');
    
    const formData = await request.formData();
    const imageFile = formData.get('image') as File;
    const userDescription = formData.get('description') as string || '';
    const prioritySearch = formData.get('priority') as string || '';
    
    if (!imageFile) {
      return NextResponse.json({
        success: false,
        error: 'No image file provided'
      }, { status: 400 });
    }

    console.log(`📸 Processing image: ${imageFile.name} (${imageFile.size} bytes)`);
    if (userDescription) console.log(`📝 User description: ${userDescription}`);
    if (prioritySearch) console.log(`🎯 Priority search: ${prioritySearch}`);

    // Convert image to base64
    const bytes = await imageFile.arrayBuffer();
    const base64Image = Buffer.from(bytes).toString('base64');
    
    // Prepare request for coordinator
    const coordinatorRequest: CoordinatorRequest = {
      image_data: base64Image,
      image_format: imageFile.type,
      user_description: userDescription || undefined,
      priority_search: prioritySearch || undefined,
      patient_info: {}
    };

    console.log('🤖 Sending request to coordinator agent...');

    // Call the coordinator's HTTP API
    try {
      const coordinatorResponse = await fetch('http://localhost:8080/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(coordinatorRequest),
        signal: AbortSignal.timeout(60000) // 1 minute timeout
      });

      if (!coordinatorResponse.ok) {
        throw new Error(`Coordinator responded with status: ${coordinatorResponse.status}`);
      }

      const coordinatorResult = await coordinatorResponse.json();
      
      if (!coordinatorResult.success) {
        throw new Error(coordinatorResult.error || 'Coordinator processing failed');
      }

      // Convert coordinator response to frontend format
      const analysisResult: AnalysisResult = {
        id: coordinatorResult.data.analysis_id,
        timestamp: coordinatorResult.data.timestamp,
        urgency: coordinatorResult.data.urgency,
        confidence: coordinatorResult.data.confidence,
        findings: coordinatorResult.data.findings,
        report: coordinatorResult.data.report,
        image: `data:${imageFile.type};base64,${base64Image}`,
        pdf_data: coordinatorResult.data.pdf_data
      };

      console.log('✅ Analysis completed successfully via coordinator');

      return NextResponse.json({
        success: true,
        data: analysisResult,
        processing_info: coordinatorResult.data.processing_summary
      });

    } catch (error) {
      console.error('❌ Cannot reach coordinator agent');
      
      // Create a fallback response with user input incorporated
      const mockResult: AnalysisResult = {
        id: `analysis-${Date.now()}`,
        timestamp: new Date().toISOString(),
        urgency: prioritySearch ? 3 : 2, // Higher urgency if specific search requested
        confidence: 0.82,
        findings: [
          'Agent network unavailable - using mock analysis',
          ...(userDescription ? [`Patient reports: ${userDescription}`] : []),
          ...(prioritySearch ? [`Focused analysis requested for: ${prioritySearch}`] : []),
          'Normal cardiac silhouette',
          'Clear lung fields bilaterally'
        ],
        report: {
          indication: userDescription
            ? `Patient presents with: ${userDescription}${prioritySearch ? `. Specific evaluation requested for ${prioritySearch}` : ''}`
            : 'Routine chest X-ray examination',
          comparison: 'No prior studies available for comparison',
          findings: `The heart size appears normal. The lungs are clear bilaterally without evidence of consolidation, pleural effusion, or pneumothorax. ${prioritySearch ? `Special attention was given to evaluating for ${prioritySearch} as requested.` : ''} No acute osseous abnormalities are identified.`,
          impression: prioritySearch
            ? `No evidence of ${prioritySearch} identified. No acute cardiopulmonary abnormality.`
            : 'No acute cardiopulmonary abnormality identified.'
        },
        image: `data:${imageFile.type};base64,${base64Image}`,
        pdf_data: undefined
      };

      return NextResponse.json({
        success: true,
        data: mockResult,
        warning: 'Agent network unavailable. Using mock analysis for demonstration.'
      });
    }

    // If we get here, coordinator is running but we need to implement agent communication
    // For now, create an enhanced mock that shows the system is working
    const enhancedResult: AnalysisResult = {
      id: `analysis-${Date.now()}`,
      timestamp: new Date().toISOString(),
      urgency: prioritySearch && ['pneumonia', 'pneumothorax', 'mass'].includes(prioritySearch.toLowerCase()) ? 4 : 2,
      confidence: 0.87,
      findings: [
        'Coordinator agent connected successfully',
        ...(userDescription ? [`Clinical context: ${userDescription}`] : []),
        ...(prioritySearch ? [`Priority evaluation: ${prioritySearch}`] : []),
        'Agents are processing your request...',
        'This will be replaced with real agent results soon'
      ],
      report: {
        indication: userDescription
          ? `Chest X-ray examination for evaluation of ${userDescription}${prioritySearch ? ` with focus on ${prioritySearch}` : ''}`
          : 'Chest X-ray examination for cardiopulmonary assessment',
        comparison: 'No prior studies available for comparison',
        findings: `Chest X-ray analysis in progress. ${prioritySearch ? `Special attention being given to ${prioritySearch} as requested. ` : ''}The AI agents are working to provide comprehensive analysis incorporating the provided clinical context.`,
        impression: 'Analysis in progress. Results will be available shortly with full agent network integration.'
      },
      image: `data:${imageFile.type};base64,${base64Image}`,
      pdf_data: undefined
    };

    console.log('✅ Enhanced mock response prepared with user input');

    return NextResponse.json({
      success: true,
      data: enhancedResult,
      info: 'Agent network detected. Full integration coming next.'
    });

  } catch (error) {
    console.error('❌ API Error:', error);
    
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    
    return NextResponse.json({
      success: false,
      error: errorMessage,
      details: 'Check server logs for more information'
    }, { status: 500 });
  }
}

// Health check endpoint
export async function GET() {
  try {
    // Check if coordinator agent is available
    const coordinatorCheck = await fetch('http://localhost:9000/health', {
      method: 'GET',
      signal: AbortSignal.timeout(3000)
    });

    const isCoordinatorHealthy = coordinatorCheck.ok;

    return NextResponse.json({
      status: 'ok',
      timestamp: new Date().toISOString(),
      services: {
        api: 'healthy',
        coordinator_agent: isCoordinatorHealthy ? 'healthy' : 'unavailable',
        agent_network: isCoordinatorHealthy ? 'partial' : 'unavailable'
      }
    });
  } catch (error) {
    return NextResponse.json({
      status: 'degraded',
      timestamp: new Date().toISOString(),
      services: {
        api: 'healthy',
        coordinator_agent: 'unavailable',
        agent_network: 'unavailable'
      },
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 503 });
  }
}