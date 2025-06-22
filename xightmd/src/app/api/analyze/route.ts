import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs/promises';

const GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

interface GeminiResponse {
  candidates: Array<{
    content: {
      parts: Array<{ text: string }>
    }
  }>
}

async function analyzeWithLocalModel(imageBuffer: Buffer): Promise<any> {
  const tmpDir = path.join(process.cwd(), 'tmp');
  await fs.mkdir(tmpDir, { recursive: true });

  const tmpFilePath = path.join(tmpDir, `upload-${Date.now()}.jpg`);
  await fs.writeFile(tmpFilePath, imageBuffer);
  try {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', [
        'inference.py',
        '--model_path', 'lung_classifier_best.pth',
        '--image_path', tmpFilePath
      ]);

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      pythonProcess.on('close', (code) => {
        fs.unlink(tmpFilePath).catch(console.error);

        if (code !== 0) {
          console.error('Python process error:', errorOutput);
          reject(new Error(`Python process failed with code ${code}`));
          return;
        }
        try {
          const result = JSON.parse(output);
          resolve(result);
  } catch (err) {
          reject(new Error('Failed to parse Python output'));
        }
      });
    });
  } catch (err) {
    await fs.unlink(tmpFilePath).catch(console.error);
    throw err;
  }
}

async function analyzeWithGemini(imageBase64: string, modelAnalysis: any): Promise<any> {
  const analysisText = `Model prediction: ${modelAnalysis.prediction}, Confidence: ${modelAnalysis.confidence}`;

  const prompt = `You are a radiologist AI assistant. Based on the chest X-ray image and this analysis from our lung classifier model ("${analysisText}"), generate a structured radiology report with:

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
    console.warn('Failed to parse Gemini JSON, returning fallback.');
    return {
      report: {
        indication: "Chest X-ray evaluation",
        comparison: "None available",
        findings: analysisText,
        impression: "See above AI interpretation"
      },
      urgency: modelAnalysis.prediction === "abnormal" ? 3 : 1,
      confidence: modelAnalysis.confidence,
      findings: [modelAnalysis.prediction === "abnormal" ?
        "Potential abnormality detected" :
        "No significant abnormalities detected"],
      recommendations: ["Clinical correlation recommended"]
    };
}
}

export async function POST(request: NextRequest) {
  try {
    console.log('Starting local model + Gemini analysis...');

    const formData = await request.formData();
    const imageFile = formData.get('image') as File;

    if (!imageFile) {
      return NextResponse.json(
        { success: false, error: 'No image file provided' },
        { status: 400 }
      );
    }

    const imageBuffer = Buffer.from(await imageFile.arrayBuffer());
    const imageBase64 = imageBuffer.toString('base64');

    console.log('Running local lung classifier model...');
    const modelResponse = await analyzeWithLocalModel(imageBuffer);
    console.log('Local model analysis:', modelResponse);

    console.log('Generating structured report with Gemini...');
    const geminiAnalysis = await analyzeWithGemini(imageBase64, modelResponse);
    console.log('Gemini report generated');

    const result = {
      id: `analysis-${Date.now()}`,
      timestamp: new Date().toISOString(),
      urgency: geminiAnalysis.urgency || 2,
      confidence: modelResponse.confidence || 0.75,
      findings: geminiAnalysis.findings || ['AI-generated findings'],
      report: geminiAnalysis.report,
      image: `data:${imageFile.type};base64,${imageBase64}`,
      model_analysis: modelResponse,
      model_info: {
        primary_model: 'lung_classifier_best.pth (Local)',
        report_generator: 'Gemini Flash 2.0',
        pipeline: 'Local PyTorch + Gemini',
      },
    };

    return NextResponse.json({
      success: true,
      data: result,
    });
  } catch (err: any) {
    console.error('Analysis failed:', err);
    return NextResponse.json(
      {
        success: false,
        error: err.message || 'Unexpected error',
      },
      { status: 500 }
    );
  }
}
