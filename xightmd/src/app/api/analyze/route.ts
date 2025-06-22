// src/api/analyze/route.ts
import { NextRequest, NextResponse } from 'next/server';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('image') as File;
    
    if (!file) {
      return NextResponse.json(
        { success: false, error: 'No image file provided' },
        { status: 400 }
      );
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      return NextResponse.json(
        { success: false, error: 'File size must be less than 10MB' },
        { status: 400 }
      );
    }

    // Validate file type
    if (!file.type.startsWith('image/')) {
      return NextResponse.json(
        { success: false, error: 'File must be an image' },
        { status: 400 }
      );
    }

    // Forward the file to the FastAPI backend
    const backendFormData = new FormData();
    backendFormData.append('file', file);

    console.log(`📤 Sending analysis request to backend: ${API_BASE_URL}/api/analyze`);

    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      body: backendFormData,
      signal: AbortSignal.timeout(60000) // 60 second timeout for analysis
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      console.error(`❌ Backend error: ${response.status}`, errorData);
      
      return NextResponse.json(
        { 
          success: false, 
          error: errorData.detail || `Backend error: ${response.status}` 
        },
        { status: response.status }
      );
    }

    const result = await response.json();
    console.log(`✅ Analysis completed successfully`);
    
    return NextResponse.json({
      success: true,
      data: result.data
    });

  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    const errorName = error instanceof Error ? error.name : 'UnknownError';
    
    console.error('❌ Analysis error:', errorMessage);
    
    if (errorName === 'AbortError') {
      return NextResponse.json(
        { 
          success: false, 
          error: 'Analysis timeout - please try again' 
        },
        { status: 408 }
      );
    }
    
    return NextResponse.json(
      { 
        success: false, 
        error: 'Internal server error' 
      },
      { status: 500 }
    );
  }
}