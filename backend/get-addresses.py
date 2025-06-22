#!/usr/bin/env python3
# backend/get_agent_addresses.py
"""
Script to get real uAgent addresses for all agents in the XightMD system.
Run this before starting the agents to get the correct addresses.
"""

from uagents import Agent

def get_agent_addresses():
    """Get all agent addresses using the same seeds as the actual agents"""
    
    print("🔍 Generating agent addresses...")
    print("=" * 60)
    
    # Create agents with the same configuration as your actual agents
    api_server = Agent(
        name="api_server",
        port=8080,
        seed="api_server_seed_123"
    )
    
    coordinator = Agent(
        name="coordinator", 
        port=9000,
        seed="coordinator_recovery_phrase"
    )
    
    triage = Agent(
        name="triage_agent",
        port=8001, 
        seed="triage_agent_seed_123"
    )
    
    report = Agent(
        name="report_agent",
        port=8002,
        seed="report_agent_seed_456" 
    )
    
    qa = Agent(
        name="qa_agent",
        port=8006,
        seed="qa_agent_seed_789"
    )
    
    # Print addresses
    addresses = {
        "api_server": str(api_server.address),
        "coordinator": str(coordinator.address), 
        "triage": str(triage.address),
        "report": str(report.address),
        "qa": str(qa.address)
    }
    
    print("📍 Agent Addresses:")
    for name, address in addresses.items():
        print(f"   {name:12}: {address}")
    
    print("\n🔧 Update these in your agent files:")
    print(f"API_SERVER_ADDRESS = \"{addresses['api_server']}\"")
    
    print("\n📋 Environment Variables (add to .env):")
    for name, address in addresses.items():
        env_name = f"{name.upper()}_ADDRESS"
        print(f"{env_name}={address}")
    
    print("\n✅ Copy these addresses to your agent configuration files!")
    
    return addresses

if __name__ == "__main__":
    get_agent_addresses()