#!/bin/bash
# start_agents.sh - Start all XightMD agents

echo "🚀 Starting XightMD Agent Network (API-Based)"
echo "=============================================="

# Create logs directory
mkdir -p logs

# Function to start agent in background
start_agent() {
    local agent_name=$1
    local script_path=$2
    local port=$3
    
    echo "📍 Starting $agent_name on port $port..."
    cd backend
    python $script_path > ../logs/$agent_name.log 2>&1 &
    local pid=$!
    echo $pid > ../logs/$agent_name.pid
    echo "✅ $agent_name started (PID: $pid)"
    cd ..
}

# Start agents in correct order
echo ""
echo "🔬 Starting Triage Agent (API-based classifier)..."
start_agent "triage" "agents/triage_agent.py" "8001"
sleep 2

echo ""
echo "📄 Starting Report Agent (medical reporting)..."
start_agent "report" "agents/report_agent.py" "8002"
sleep 2

echo ""
echo "✅ Starting QA Agent (medical validation)..."
start_agent "qa" "agents/qa_agent.py" "8006"
sleep 2

echo ""
echo "🎯 Starting Coordinator Agent (orchestration)..."
start_agent "coordinator" "agents/coordinator.py" "9000"
sleep 3

echo ""
echo "🌐 Starting FastAPI Server..."
start_agent "api" "api/server.py" "8000"
sleep 2

echo ""
echo "🎉 XightMD Agent Network Started!"
echo "=================================="
echo "📊 Agent Status:"
echo "   🔬 Triage Agent:     http://localhost:8001"
echo "   📄 Report Agent:     http://localhost:8002" 
echo "   ✅ QA Agent:         http://localhost:8006"
echo "   🎯 Coordinator:      http://localhost:9000"
echo "   🌐 API Server:       http://localhost:8000"
echo ""
echo "🏥 Frontend: http://localhost:3000"
echo "📋 API Docs: http://localhost:8000/docs"
echo ""
echo "To stop all agents: ./stop_agents.sh"
echo "To view logs: tail -f logs/{agent_name}.log"