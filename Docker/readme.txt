# MCP Swarm Intelligence Server

A Model Context Protocol (MCP) server that implements collective intelligence for multi-agent coordination featuring agent ecosystem management, hive mind knowledge bases, persistent memory systems, and automated workflow orchestration.

## Purpose

This MCP server provides a secure interface for AI assistants to coordinate multiple agents using swarm intelligence algorithms, maintain collective knowledge, and optimize task assignment through persistent learning systems.

## Features

### Core Swarm Intelligence Features

- **`agent_assignment`** - Assign tasks to optimal agents using Ant Colony Optimization (ACO) algorithms
- **`hive_mind_query`** - Query collective knowledge base with semantic search capabilities
- **`swarm_consensus`** - Reach consensus on decisions using democratic swarm algorithms
- **`adaptive_coordination`** - Dynamically coordinate multiple agents with strategy selection
- **`agent_config_manager`** - Manage agent configuration files and capabilities
- **`knowledge_contribution`** - Contribute knowledge to the hive mind collective database
- **`ecosystem_management`** - Monitor and manage agent ecosystem health
- **`semantic_search`** - Advanced semantic search across the knowledge base
- **`performance_metrics`** - Comprehensive performance monitoring and analytics
- **`decision_confidence`** - Calculate decision confidence metrics with multi-factor analysis
- **`self_monitoring`** - Monitor server and agent health with real-time diagnostics

### Advanced Intelligence Capabilities

- **Agent Assignment**: Optimal task assignment using ACO (Ant Colony Optimization) algorithms
- **Hive Mind Knowledge Management**: Collective knowledge base with semantic search
- **Swarm Consensus**: Democratic decision-making across multiple agents
- **Adaptive Coordination**: Dynamic strategy selection and real-time coordination
- **Memory Persistence**: Cross-session state management with SQLite backend

### Planned Additional Tools (26+ more)

- **`agent_discovery`** - Automatically discover and register agent capabilities
- **`dynamic_coordination`** - Real-time coordination strategy selection
- **`complete_pipeline`** - Execute complete multi-agent workflows
- **`knowledge_extraction`** - Extract structured knowledge from data
- **`knowledge_synthesis`** - Synthesize knowledge from multiple sources
- **`knowledge_validation`** - Validate knowledge quality and consistency
- **`automation_validation`** - Validate automated processes and workflows
- **`decision_audit`** - Audit decision-making processes
- **`risk_assessment`** - Assess risks in agent coordination
- **`directory_manager`** - Manage project directory structures
- **`agent_hooks`** - Execute agent lifecycle hooks
- **`confidence_aggregation`** - Aggregate confidence scores across agents
- **`consensus_algorithms`** - Apply various consensus mechanisms
- **`coordination_strategies`** - Select optimal coordination strategies
- **`explanation`** - Provide explanations for agent decisions
- **`fuzzy_matcher`** - Match capabilities using fuzzy logic
- **`knowledge_classifier`** - Classify knowledge by domain and type
- **`knowledge_quality`** - Assess knowledge quality metrics
- **`load_balancer`** - Balance workload across agents
- **`mcda`** - Multi-criteria decision analysis
- **`minority_opinion`** - Capture and analyze minority opinions
- **`strategy_selector`** - Select optimal coordination strategies
- **`mcp_server_manager`** - Manage MCP server lifecycle
- **`adaptive_learning`** - Machine learning for agent optimization
- **`knowledge_updater`** - Update knowledge base with versioning
- **`coordination_pattern_learning`** - Learn from successful coordination patterns

## Prerequisites

- Docker Desktop with MCP Toolkit enabled
- Docker MCP CLI plugin (`docker mcp` command)
- Python 3.11+ (for development)
- SQLite 3.38+ with JSON1 and FTS5 extensions

## Installation

See the step-by-step instructions provided with the files.

## Usage Examples

In Claude Desktop, you can ask:

### Agent Coordination
- "Assign the task 'optimize database queries' to the best available agent"
- "Coordinate 5 agents to work on the frontend redesign project"
- "What's the consensus on implementing the new authentication system?"

### Knowledge Management
- "Search the hive mind for information about Python async patterns"
- "Add knowledge about Docker security best practices to the database"
- "What does our collective knowledge say about microservices architecture?"

### Performance & Monitoring
- "Show me the performance metrics for all agents this week"
- "Check the ecosystem health and identify any issues"
- "Monitor the current system status and agent activity"

### Decision Support
- "Calculate confidence for the decision to migrate to Kubernetes"
- "Get semantic search results for 'CI/CD pipeline optimization'"
- "What's the performance score for agent 'backend-specialist'?"

## Architecture

```
Claude Desktop → MCP Gateway → MCP Swarm Intelligence Server → SQLite Database
                                          ↓
                               Agent Configuration Files
                                          ↓
                            Swarm Intelligence Algorithms (ACO/PSO)
                                          ↓
                              Hive Mind Knowledge Base
```

### Database Schema

- **agents** - Agent registry with capabilities and performance tracking
- **tasks** - Task assignment history with confidence scores
- **hive_knowledge** - FTS5-enabled collective knowledge base
- **consensus_decisions** - Decision history with consensus metrics
- **performance_metrics** - System performance and analytics data

### Swarm Intelligence Implementation

- **Ant Colony Optimization (ACO)**: For optimal task assignment
- **Particle Swarm Optimization (PSO)**: For parameter optimization
- **Democratic Consensus**: For collective decision-making
- **Queen-Led Coordination**: Hierarchical coordination patterns

## Development

### Local Testing

```bash
# Set environment variables for testing
export SWARM_DB_PATH="data/test_memory.db"
export SWARM_DB_ENCRYPTION_KEY="test-key"
export SWARM_ADMIN_TOKEN="test-admin-token"

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "import asyncio; from mcp_swarm_intelligence_server import startup; asyncio.run(startup())"

# Run directly
python mcp_swarm_intelligence_server.py

# Test MCP protocol
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python mcp_swarm_intelligence_server.py
```

### Database Management

```bash
# View database schema
sqlite3 data/memory.db ".schema"

# Check agent status
sqlite3 data/memory.db "SELECT * FROM agents;"

# View knowledge entries
sqlite3 data/memory.db "SELECT domain, substr(content, 1, 50) FROM hive_knowledge LIMIT 10;"

# Performance metrics
sqlite3 data/memory.db "SELECT metric_type, AVG(value) FROM performance_metrics GROUP BY metric_type;"
```

### Adding New Tools

1. Add the function to `mcp_swarm_intelligence_server.py`
2. Decorate with `@mcp.tool()`
3. Follow single-line docstring pattern
4. Include comprehensive error handling
5. Update the catalog entry with the new tool name
6. Rebuild the Docker image

## Configuration

### Environment Variables

- **`SWARM_DB_PATH`** - Path to SQLite database file (default: `data/memory.db`)
- **`SWARM_DB_ENCRYPTION_KEY`** - Optional database encryption key
- **`SWARM_ADMIN_TOKEN`** - Optional admin access token

### Agent Configuration

Agents can be configured with:
- **ID**: Unique identifier
- **Name**: Human-readable name
- **Capabilities**: Comma-separated capability list
- **Performance Score**: 0.0-1.0 performance rating

## Troubleshooting

### Tools Not Appearing

- Verify Docker image built successfully: `docker build -t mcp-swarm-intelligence-server .`
- Check catalog and registry files for syntax errors
- Ensure Claude Desktop config includes custom catalog
- Restart Claude Desktop completely

### Database Connection Errors

- Ensure data directory exists and is writable
- Check SQLite version supports JSON1 and FTS5: `sqlite3 --version`
- Verify file permissions for database file

### Agent Assignment Issues

- Check agent registration: Use `agent_config_manager` with action "list"
- Verify agent performance scores are reasonable (0.0-1.0)
- Ensure sufficient active agents for coordination

### Performance Issues

- Monitor database size: Large knowledge bases may need optimization
- Check agent activity: Too many idle agents can affect performance
- Use performance metrics to identify bottlenecks

### Memory Management

- Database auto-cleanup runs for completed tasks older than 30 days
- Inactive agents are removed after 7 days of inactivity
- Use `ecosystem_management` with action "cleanup" for manual cleanup

## Security Considerations

- All secrets stored in Docker Desktop secrets
- Never hardcode credentials in code
- Running as non-root user (mcpuser)
- Database encryption available via `SWARM_DB_ENCRYPTION_KEY`
- Sensitive data never logged to prevent information leakage
- Agent access controlled through configuration management
- Optional admin token for privileged operations

## Performance Optimization

- SQLite FTS5 indexing for fast semantic search
- Connection pooling for database operations
- Async/await patterns for concurrent processing
- Numpy-based calculations for swarm algorithms
- Efficient agent discovery and capability matching

## Integration Examples

### VS Code Extension Integration
```typescript
// Register MCP server in VS Code extension
const mcpServer = vscode.lm.registerMcpServerDefinitionProvider({
    name: 'MCP Swarm Intelligence',
    serverPath: 'docker',
    serverArgs: ['run', '-i', '--rm', 'mcp-swarm-intelligence-server']
});
```

### Claude Desktop Configuration
```json
{
  "mcpServers": {
    "mcp-toolkit-gateway": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",
        "-v", "/Users/username/.docker/mcp:/mcp",
        "docker/mcp-gateway",
        "--catalog=/mcp/catalogs/custom.yaml"
      ]
    }
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-tool`
3. Add comprehensive tests for new tools
4. Follow existing code patterns and error handling
5. Update documentation and catalog entries
6. Submit pull request with detailed description

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
1. Check troubleshooting section above
2. Review server logs: `docker logs <container_name>`
3. Test with minimal configuration first
4. Use `self_monitoring` tool for diagnostics