# CLAUDE.md - XightMD Project

## Project Overview

**XightMD** is a multi-agent AI system that assists radiologists in analyzing chest X-rays and generating structured reports. The system uses Fetch.ai's uAgents framework with Claude 4 for multimodal analysis and report generation.

### Key Features
- **Multimodal AI**: Combines computer vision (chest X-ray analysis) with natural language processing (report generation)
- **Multi-Agent Architecture**: Coordinated agents for triage, analysis, and reporting
- **Real-time Processing**: Instant analysis and structured report generation
- **Healthcare Focus**: Designed to assist radiologists, not replace them

## Architecture

```
Frontend (Next.js) ↔ API Bridge (FastAPI) ↔ Agent Network (uAgents) ↔ Claude 4 API
```

### Components
1. **Frontend**: Next.js React application for image upload and results display
2. **API Bridge**: FastAPI server connecting frontend to agent network
3. **Agent Network**: Three coordinated uAgents handling different aspects
4. **AI Models**: Claude 4 for vision analysis and report generation

## Technology Stack

### Frontend
- **Framework**: Next.js 14 with TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Custom React components
- **API Communication**: Fetch API with FormData for file uploads

### Backend
- **Agent Framework**: Fetch.ai uAgents
- **API Server**: FastAPI with CORS support
- **AI Integration**: Anthropic Claude API
- **File Handling**: Python multipart for image uploads

### AI & ML
- **Vision Model**: Claude 3.5 Sonnet (multimodal)
- **Text Generation**: Claude 4 for structured reports
- **Medical Dataset**: NIH Chest X-ray dataset for validation
- **Report Structure**: Based on ReXGradient-160K format

## Project Structure

```
XightMD/
├── backend/
│   ├── api/
│   │   └── server.py              # FastAPI bridge
│   ├── agents/
│   │   ├── coordinator.py         # Main coordination agent
│   │   ├── triage_agent.py        # Image analysis & urgency
│   │   ├── report_agent.py        # Report generation
│   │   └── qa_agent.py            # Quality assurance
│   ├── utils/
│   │   ├── image_processing.py    # Image preprocessing
│   │   └── claude_client.py       # Claude API wrapper
│   ├── requirements.txt
│   └── .env                       # API keys
└── xightmd/                       # Next.js frontend
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx           # Main dashboard
    │   │   ├── upload/            # Upload interface
    │   │   └── api/               # API routes
    │   ├── components/
    │   │   ├── ImageUpload.tsx    # File upload component
    │   │   ├── ReportDisplay.tsx  # Results display
    │   │   ├── AgentStatus.tsx    # Agent monitoring
    │   │   └── Dashboard.tsx      # Main layout
    │   └── lib/
    │       └── api.ts             # API client functions
    ├── package.json
    └── tailwind.config.js
```

## Agent Architecture

### 1. Coordinator Agent
- **Role**: Orchestrates the entire analysis pipeline
- **Responsibilities**: 
  - Receives image upload requests
  - Coordinates between other agents
  - Manages workflow state
  - Returns final results to API

### 2. Triage Agent
- **Role**: Initial image analysis and urgency classification
- **Capabilities**:
  - Chest X-ray abnormality detection
  - Urgency level scoring (1-5)
  - Critical finding identification (pneumothorax, fractures)
  - Confidence scoring

### 3. Report Agent
- **Role**: Structured radiology report generation
- **Output Format**:
  - **Indication**: Reason for examination
  - **Comparison**: Prior studies reference
  - **Findings**: Detailed observations
  - **Impression**: Summary and recommendations

### 4. QA Agent
- **Role**: Quality assurance and validation
- **Functions**:
  - Cross-validation between agents
  - Consistency checking
  - Confidence assessment
  - Flag discrepancies for review

## API Endpoints

### Core Endpoints
- `POST /api/analyze` - Upload and analyze chest X-ray
- `GET /api/health` - Service health check
- `GET /api/agents/status` - Agent network status
- `POST /api/feedback` - Submit analysis feedback

### Analysis Workflow
```
1. Image Upload → 2. Triage Analysis → 3. Report Generation → 4. QA Validation → 5. Results
```

## Development Workflow

### Setup Commands
```bash
# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend setup
cd xightmd
npm install
```

### Running the Application
```bash
# Terminal 1: Backend API
cd backend && python api/server.py

# Terminal 2: Frontend
cd xightmd && npm run dev

# Terminal 3: Agent Network
cd backend && python agents/coordinator.py
```

### Environment Variables
```bash
# backend/.env
ANTHROPIC_API_KEY=your_claude_api_key
FETCH_AI_AGENT_MAILBOX_KEY=your_agent_key
```

## Data Sources

### NIH Chest X-ray Dataset
- **Source**: `gs://gcs-public-data--healthcare-nih-chest-xray`
- **Usage**: Validation and demo data
- **Format**: PNG images with metadata
- **Size**: 100,000 de-identified chest X-rays

### ReXGradient-160K Dataset
- **Source**: Hugging Face datasets
- **Usage**: Report structure templates
- **Content**: 160,000 studies with structured reports
- **Format**: Four-section radiology reports

## Claude Integration Patterns

### Multimodal Analysis
```python
# Image analysis with Claude Vision
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "data": image_data}},
            {"type": "text", "text": "Analyze this chest X-ray for abnormalities..."}
        ]
    }]
)
```

### Structured Report Generation
```python
# Report generation with specific medical format
template = """
Generate a radiology report with this structure:
INDICATION: {indication}
COMPARISON: {comparison}
FINDINGS: {findings}
IMPRESSION: {impression}

Based on these findings: {analysis_results}
"""
```

## Testing Strategy

### Unit Tests
- Agent communication protocols
- Image processing pipelines
- API endpoint functionality
- Claude API integration

### Integration Tests
- End-to-end workflow testing
- Multi-agent coordination
- Frontend-backend communication
- Error handling scenarios

### Medical Validation
- Test with NIH dataset samples
- Compare outputs with known diagnoses
- Validate report structure consistency
- Measure analysis accuracy

## Deployment Considerations

### Agentverse Deployment
- Register agents on Fetch.ai Agentverse
- Implement Chat Protocol for ASI:One discovery
- Configure agent addresses and communication
- Set up agent monitoring and logging

### Production Readiness
- HIPAA compliance considerations
- Data de-identification requirements
- Performance optimization
- Error handling and recovery
- Audit logging for medical applications

## Hackathon Specifics

### Target Prizes
- **Fetch AI ($3,000)**: Multi-agent architecture with uAgents
- **Social Impact Track ($5,000)**: Healthcare accessibility
- **Best Use of Claude 4 ($2,500)**: Multimodal medical analysis
- **Best LLM Agents ($4,000)**: Coordinated agent system

### Demo Highlights
1. **Real-time Analysis**: Upload → Results in seconds
2. **Multi-agent Coordination**: Visual agent status monitoring
3. **Medical Accuracy**: Validation with real chest X-rays
4. **Structured Output**: Professional radiology reports
5. **Accessibility Impact**: Bringing specialist analysis to underserved areas

## Future Enhancements

### Short-term
- Voice integration with Vapi
- Mobile-responsive interface
- Batch processing capabilities
- Enhanced error handling

### Long-term
- Integration with hospital PACS systems
- Multi-modal support (CT, MRI)
- Real-time collaboration features
- Advanced analytics dashboard

## Contributing Guidelines

### Code Style
- Python: Follow PEP 8, use type hints
- TypeScript: Strict mode, ESLint configuration
- Commits: Conventional commit messages
- Documentation: Inline comments for complex logic

### Pull Request Process
1. Feature branch from main
2. Unit tests for new functionality
3. Update documentation
4. Code review approval
5. Merge to main

---

## Quick Reference

### Key Commands
```bash
# Start development environment
make dev

# Run tests
make test

# Deploy to Agentverse
make deploy

# Check agent status
make status
```

### Important URLs
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Agent Network: http://localhost:8001
- Agentverse: https://agentverse.ai

### Support Contacts
- Technical Lead: [Team Member 1]
- AI/ML Lead: [Team Member 2] 
- Integration Lead: [Team Member 3]