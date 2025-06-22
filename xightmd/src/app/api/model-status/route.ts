// src/app/api/model-status/route.ts
import { NextResponse } from 'next/server';

const HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/microsoft/BiomedVLP-CXR-BERT-general";

export async function GET() {
  try {
    console.log('🔍 Checking Hugging Face model status...');
    
    // Check if API keys are configured
    if (!process.env.HUGGINGFACE_API_KEY) {
      return NextResponse.json({
        ready: false,
        error: 'HUGGINGFACE_API_KEY not configured',
        status: 'error'
      });
    }
    
    if (!process.env.ANTHROPIC_API_KEY) {
      return NextResponse.json({
        ready: false,
        error: 'ANTHROPIC_API_KEY not configured', 
        status: 'error'
      });
    }

    // Test Hugging Face API with a minimal request
    const response = await fetch(HUGGINGFACE_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        inputs: "test",
        parameters: {
          max_length: 10
        },
        options: {
          wait_for_model: false,
          use_cache: true
        }
      }),
    });

    const data = await response.json();
    
    // Check if model is loading
    if (data.error && data.error.includes('loading')) {
      return NextResponse.json({
        ready: false,
        status: 'loading',
        estimated_time: data.estimated_time || 60,
        message: 'Model is loading, please wait...'
      });
    }
    
    // Check if model is ready (even if we get an error due to minimal input)
    if (response.status === 200 || (data.error && !data.error.includes('loading'))) {
      return NextResponse.json({
        ready: true,
        status: 'ready',
        model: 'microsoft/BiomedVLP-CXR-BERT-general',
        message: 'Model ready for inference'
      });
    }
    
    // Other errors
    return NextResponse.json({
      ready: false,
      status: 'error',
      error: data.error || 'Unknown error',
      message: 'Model unavailable'
    });

  } catch (error) {
    console.error('❌ Model status check failed:', error);
    
    return NextResponse.json({
      ready: false,
      status: 'error',
      error: error instanceof Error ? error.message : 'Network error',
      message: 'Unable to check model status'
    });
  }
}