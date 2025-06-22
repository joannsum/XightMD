#!/bin/bash

# Function to handle shutdown gracefully
cleanup() {
    echo "🛑 Shutting down all services..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

echo "🚀 Starting XightMD Agent Network..."

# Start all agents in background
echo "📍 Starting Coordinator Agent..."
python agents/coordinator.py &
COORDINATOR_PID=$!

echo "🔍 Starting Triage Agent..."
python agents/triage_agent.py &
TRIAGE_PID=$!

echo "📄 Starting Report Agent..."
python agents/report_agent.py &
REPORT_PID=$!

echo "✅ Starting QA Agent..."
python agents/qa_agent.py &
QA_PID=$!

# Wait a moment for agents to initialize
sleep 5 

echo "🌐 Starting API Server..."
python api/server.py &
API_PID=$!

echo "🎉 All services started!"
echo "📊 Process IDs:"
echo "   Coordinator: $COORDINATOR_PID"
echo "   Triage: $TRIAGE_PID"
echo "   Report: $REPORT_PID"
echo "   QA: $QA_PID"
echo "   API: $API_PID"

# Wait for all background processes
wait