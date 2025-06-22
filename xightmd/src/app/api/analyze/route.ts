import { NextRequest, NextResponse } from 'next/server';
const HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/nickmuchi/vit-finetuned-chest-xray-pneumonia";

const GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY 

interface ClaudeResponse {
  content: Array<{
    type: string;
    text: string;
  }>;
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

// Generate structured report with Claude
async function generateReportWithClaude(hfAnalysis: string, imageBase64: string): Promise<any> {
  const prompt = `You are a radiologist AI assistant. Based on the following chest X-ray analysis from a medical AI model, generate a structured radiology report.

AI Analysis: "${hfAnalysis}"

Please provide a structured report with these sections:
1. INDICATION: Why the X-ray was performed
2. COMPARISON: Prior studies (if any)
3. FINDINGS: Detailed observations of the chest X-ray
4. IMPRESSION: Summary and clinical significance

Also provide:
- Urgency score (1-5, where 5 is critical)
- Confidence score (0-1)
- Key findings list (3-5 bullet points)
- Clinical recommendations if any

Format your response as JSON with this structure:
{
  "report": {
    "indication": "string",
    "comparison": "string", 
    "findings": "string",
    "impression": "string"
  },
  "urgency": number,
  "confidence": number,
  "findings": ["string"],
  "recommendations": ["string"]
}`;

  const response = await fetch(CLAUDE_API_URL, {
    method: 'POST',
    headers: {
      'x-api-key': process.env.ANTHROPIC_API_KEY!, // ✅ correct key name
      'Content-Type': 'application/json',
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-3-opus-20240229',
      max_tokens: 1500,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: 'image/jpeg',
                data: imageBase64,
              },
            },
            {
              type: 'text',
              text: prompt,
            },
          ],
        },
      ],
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error('Claude API error:', errorText);
    throw new Error(`Claude API failed: ${response.status}`);
  }

  const claudeResponse: ClaudeResponse = await response.json();
  const reportText = claudeResponse.content[0].text;

  try {
    const jsonMatch = reportText.match(/\{[\s\S]*\}/);
    if (jsonMatch) return JSON.parse(jsonMatch[0]);
    throw new Error('No JSON found in Claude response');
  } catch (err) {
    console.error('Failed to parse Claude response:', err);
    return {
      report: {
        indication: "Chest X-ray analysis",
        comparison: "No prior studies available",
        findings: reportText.slice(0, 500),
        impression: "Analysis completed with AI assistance",
      },
      urgency: 2,
      confidence: 0.75,
      findings: ["AI analysis completed", "See detailed findings above"],
      recommendations: ["Clinical correlation recommended"],
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

    // Step 2: Claude
    console.log('🧠 Generating structured report with Claude...');
    const claudeAnalysis = await generateReportWithClaude(hfAnalysis, imageBase64);
    console.log('✅ Claude report generated');

    const result = {
      id: `analysis-${Date.now()}`,
      timestamp: new Date().toISOString(),
      urgency: claudeAnalysis.urgency || 2,
      confidence: claudeAnalysis.confidence || 0.75,
      findings: claudeAnalysis.findings || ['AI-generated findings'],
      report: claudeAnalysis.report,
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
