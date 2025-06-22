#!/bin/bash
# stop_agents.sh - Stop all XightMD agents

echo "🛑 Stopping XightMD Agent Network"
echo "================================="

# Function to stop agent
stop_agent() {
    local agent_name=$1
    local pid_file="logs/$agent_name.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        echo "🔴 Stopping $agent_name (PID: $pid)..."
        
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            sleep 1
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo "   💥 Force killing $agent_name..."
                kill -9 "$pid"
            fi
            
            echo "   ✅ $agent_name stopped"
        else
            echo "   ⚠️  $agent_name not running"
        fi
        
        rm -f "$pid_file"
    else
        echo "   ⚠️  No PID file for $agent_name"
    fi
}

# Stop agents
stop_agent "api"
stop_agent "coordinator" 
stop_agent "qa"
stop_agent "report"
stop_agent "triage"

echo ""
echo "✅ All agents stopped"
echo "📋 Log files preserved in logs/ directory"