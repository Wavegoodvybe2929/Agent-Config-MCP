#!/bin/bash

# MCP Swarm Intelligence Server Test Script
# Tests the Docker-based MCP server functionality

echo "🔧 Testing MCP Swarm Intelligence Server Docker Integration..."

# Create test script for multi-step MCP interaction
cat > mcp_test_sequence.json << 'EOF'
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test-client", "version": "1.0.0"}}}
{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
EOF

echo "📋 Running MCP test sequence..."
cat mcp_test_sequence.json | docker run --rm -i -v /tmp/mcp-swarm-data:/app/data --name mcp-swarm-test mcp-swarm-server:latest

echo ""
echo "✅ Test completed. Check above output for:"
echo "   - Server initialization response"
echo "   - Available tools list (should show 29+ tools)"
echo "   - No error messages"

# Cleanup
rm -f mcp_test_sequence.json