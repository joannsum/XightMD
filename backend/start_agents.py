#!/usr/bin/env python3
"""
Startup script for XightMD agent network
"""

import asyncio
import subprocess
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting API Server...")
    os.chdir(os.path.join(os.path.dirname(__file__), 'api'))
    subprocess.run([sys.executable, 'server.py'])

def start_coordinator_agent():
    """Start the coordinator agent"""
    print("🤖 Starting Coordinator Agent...")
    os.chdir(os.path.join(os.path.dirname(__file__), 'agents'))
    subprocess.run([sys.executable, 'coordinator.py'])

def start_triage_agent():
    """Start the triage agent"""
    print("🔍 Starting Triage Agent...")
    os.chdir(os.path.join(os.path.dirname(__file__), 'agents'))
    subprocess.run([sys.executable, 'triage_agent.py'])

def check_requirements():
    """Check if required packages are installed"""
    try:
        import torch
        import uagents
        import fastapi
        import PIL
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_model_files():
    """Check if model files exist"""
    model_path = "models/lung_classifier.pth"
    if os.path.exists(model_path):
        print("✅ Model file found")
        return True
    else:
        print("⚠️  Model file not found. Using pre-trained DenseNet-121")
        print("To train your own model, run: python train_model.py")
        return True  # Continue anyway with pre-trained model

def main():
    """Main startup function"""
    print("🏥 XightMD Agent Network Startup")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check model files
    check_model_files()
    
    print("\n🎯 Starting services...")
    
    # Start services in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Start API server
        api_future = executor.submit(start_api_server)
        
        # Wait a bit for API to start
        time.sleep(2)
        
        # Start agents
        coordinator_future = executor.submit(start_coordinator_agent)
        triage_future = executor.submit(start_triage_agent)
        
        print("\n✨ All services starting...")
        print("📡 API Server: http://localhost:8000")
        print("🌐 Frontend: http://localhost:3000")
        print("\n🔄 Services running. Press Ctrl+C to stop.")
        
        try:
            # Wait for all services
            api_future.result()
            coordinator_future.result()
            triage_future.result()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down services...")

if __name__ == "__main__":
    main()