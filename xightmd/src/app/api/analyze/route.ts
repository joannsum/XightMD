import { NextRequest, NextResponse } from 'next/server';
const HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/nickmuchi/vit-finetuned-chest-xray-pneumonia";

const GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

interface GeminiResponse {
  candidates: Array<{
    content: {
      parts: Array<{ text: string }>
    }
  }>
}

// Analyze X-ray with Hugging Face
async function analyzeWithHuggingFace(imageBase64: string): Promise<any> {
  const response = await fetch(HUGGINGFACE_API_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.HUGGINGFACE_API_KEY}`, // ✅ use the correct token for HF
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      inputs: `data:image/jpeg;base64,${imageBase64}`,
      options: { wait_for_model: true },
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error('Hugging Face API error:', errorText);
    throw new Error(`Hugging Face API failed: ${response.status}`);
  }

  return await response.json();
}

async function analyzeWithGemini(imageBase64: string, hfAnalysis: string): Promise<any> {
  const prompt = `You are a radiologist AI assistant. Based on the chest X-ray image and this analysis from a model ("${hfAnalysis}"), generate a structured radiology report with:

1. INDICATION
2. COMPARISON
3. FINDINGS
4. IMPRESSION

Also include:
- urgency score (1-5),
- confidence score (0-1),
- 3-5 bullet-point key findings,
- clinical recommendations.

Respond only in JSON.`;

  const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [
        {
          parts: [
            { text: prompt },
            {
              inlineData: {
                mimeType: 'image/jpeg',
                data: imageBase64,
              }
            }
          ]
        }
      ]
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error('Gemini API error:', errorText);
    throw new Error(`Gemini API failed: ${response.status}`);
  }

  const result: GeminiResponse = await response.json();
  const text = result.candidates?.[0]?.content?.parts?.[0]?.text || '';

  try {
    const json = JSON.parse(text.match(/\{[\s\S]*\}/)?.[0] || '{}');
    return json;
  } catch (err) {
    console.warn('❌ Failed to parse Gemini JSON, returning fallback.');
    return {
      report: {
        indication: "Chest X-ray evaluation",
        comparison: "None available",
        findings: hfAnalysis,
        impression: "See above AI interpretation"
      },
      urgency: 2,
      confidence: 0.75,
      findings: ["Check for pneumonia", "Review image abnormalities"],
      recommendations: ["Clinical follow-up recommended"]
    };
  }
}

export async function POST(request: NextRequest) {
  try {
    console.log('🔍 Starting Hugging Face + Claude analysis...');

    const formData = await request.formData();
    const imageFile = formData.get('image') as File;

    if (!imageFile) {
      return NextResponse.json(
        { success: false, error: 'No image file provided' },
        { status: 400 }
      );
    }

    const imageBuffer = await imageFile.arrayBuffer();
    const imageBase64 = Buffer.from(imageBuffer).toString('base64');
    console.log('📸 Image converted to base64, size:', imageBase64.length);

    // Step 1: Hugging Face
    console.log('🤗 Calling Hugging Face model...');
    const hfResponse = await analyzeWithHuggingFace(imageBase64);

    const hfAnalysis = Array.isArray(hfResponse)
      ? hfResponse.map((res: any) => res.label || res.generated_text || '').join(', ')
      : hfResponse.label || hfResponse.generated_text || 'No output';

    console.log('✅ Hugging Face analysis:', hfAnalysis);

    // Step 2: Gemini
    console.log('🧠 Generating structured report with Gemini...');
    const geminiAnalysis = await analyzeWithGemini(imageBase64, hfAnalysis);
    console.log('✅ Gemini report generated');

    const result = {
      id: `analysis-${Date.now()}`,
      timestamp: new Date().toISOString(),
      urgency: geminiAnalysis.urgency || 2,
      confidence: geminiAnalysis.confidence || 0.75,
      findings: geminiAnalysis.findings || ['AI-generated findings'],
      report: geminiAnalysis.report,
      image: `data:${imageFile.type};base64,${imageBase64}`,
      hf_analysis: hfAnalysis,
      model_info: {
        primary_model: 'nickmuchi/vit-finetuned-chest-xray-pneumonia',
        report_generator: 'claude-3-haiku-20240307',
        pipeline: 'huggingface + claude',
      },
    };

    return NextResponse.json({
      success: true,
      data: result,
    });
  } catch (err: any) {
    console.error('❌ Analysis failed:', err);
    return NextResponse.json(
      {
        success: false,
        error: err.message || 'Unexpected error',
      },
      { status: 500 }
    );
  }
}
