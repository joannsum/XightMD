#!/bin/bash
# test_endpoints.sh - Test all API endpoints

echo "🧪 Testing XightMD API Endpoints"
echo "================================="

# Test backend health
echo ""
echo "1. Testing Backend Health:"
curl -s http://localhost:8000/api/health | jq '.' || echo "❌ Backend health failed"

# Test backend agent status
echo ""
echo "2. Testing Backend Agent Status:"
curl -s http://localhost:8000/api/agent-status | jq '.' || echo "❌ Backend agent status failed"

# Test frontend health
echo ""
echo "3. Testing Frontend Health:"
curl -s http://localhost:3000/api/analyze | jq '.' || echo "❌ Frontend health failed"

# Test frontend agent status
echo ""
echo "4. Testing Frontend Agent Status:"
curl -s http://localhost:3000/api/agent-status | jq '.' || echo "❌ Frontend agent status failed"

echo ""
echo "✅ Endpoint testing complete!"