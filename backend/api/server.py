#connection
from fastapi import FastAPI, File, UploadFile
from agents.triage_agent import TriageAgent

app = FastAPI()
triage_agent = TriageAgent()

@app.post("/analyze-xray")
async def analyze_xray(file: UploadFile = File(...)):
    """Analyze uploaded X-ray image"""
    try:
        # Save uploaded file
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Analyze with your lung classifier
        results = await triage_agent.analyze_image(file_path)

        return {
            "status": "success",
            "analysis": results,
            "detected_conditions": results.get('high_confidence_findings', []),
            "urgency_score": results.get('urgency_score', 0.0)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
