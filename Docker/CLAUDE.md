# MCP Swarm Intelligence Server - Claude Desktop Integration Guide

## Implementation Overview

The MCP Swarm Intelligence Server is a comprehensive Model Context Protocol implementation that provides collective intelligence capabilities for multi-agent coordination. This server integrates seamlessly with Claude Desktop to enable sophisticated swarm-based task assignment, hive mind knowledge management, and persistent learning systems.

## Architecture

### Core Components

- **MCP Protocol Layer**: Full JSON-RPC 2.0 compliance with stdio transport
- **Swarm Intelligence Engine**: ACO/PSO algorithms for optimal coordination
- **Persistent Memory System**: SQLite database with FTS5 semantic search
- **Agent Configuration Management**: Real-time file system monitoring
- **Performance Analytics**: Comprehensive metrics and health monitoring

### Database Schema

```sql
-- Agent registry with performance tracking
CREATE TABLE agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    capabilities TEXT,
    status TEXT DEFAULT 'active',
    performance_score REAL DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Task assignment with confidence scores
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    assigned_agent_id TEXT,
    confidence_score REAL DEFAULT 0.0,
    status TEXT DEFAULT 'pending',
    FOREIGN KEY (assigned_agent_id) REFERENCES agents (id)
);

-- FTS5-enabled knowledge base
CREATE VIRTUAL TABLE hive_knowledge USING fts5(
    domain, content, confidence_score UNINDEXED, 
    source_agent UNINDEXED, created_at UNINDEXED
);
```

## Tool Implementation Details

### Agent Coordination Tools

#### `agent_assignment`
- **Algorithm**: Ant Colony Optimization (ACO) with pheromone-based selection
- **Input**: Task description, priority level (0.0-1.0)
- **Output**: Optimal agent assignment with confidence score
- **Performance**: O(n) complexity for n agents, sub-second response

#### `adaptive_coordination` 
- **Strategies**: Auto, Hierarchical, Democratic, Expert, Round-robin
- **Selection Logic**: Agent count, performance variance analysis
- **Coordination Plans**: Role assignment with clear hierarchy

#### `swarm_consensus`
- **Algorithm**: Weighted voting with performance-based influence
- **Threshold**: Configurable minimum confidence (default 0.7)
- **Output**: Consensus decision with participation metrics

### Knowledge Management Tools

#### `hive_mind_query`
- **Search Engine**: SQLite FTS5 with semantic ranking
- **Response Format**: Domain-categorized results with confidence scores
- **Performance**: Indexed full-text search, millisecond response times

#### `knowledge_contribution`
- **Validation**: Content quality assessment
- **Storage**: FTS5 virtual table with metadata
- **Versioning**: Timestamp-based knowledge evolution

#### `semantic_search`
- **Advanced Features**: Query expansion, relevance ranking
- **Result Limiting**: Configurable result count (default 10)
- **Performance Metrics**: Search latency and accuracy tracking

### System Monitoring Tools

#### `ecosystem_management`
- **Health Metrics**: Agent availability, task completion rates
- **Cleanup Operations**: Automated pruning of old data
- **Status Reporting**: Color-coded health indicators

#### `performance_metrics`
- **System Overview**: Completion rates, knowledge growth
- **Agent Analytics**: Individual performance tracking
- **Trend Analysis**: Historical performance data

#### `self_monitoring`
- **Database Health**: Connection status, query performance
- **Resource Usage**: Memory, disk space monitoring  
- **Agent Activity**: Idle detection, health checks

## Claude Desktop Configuration

### Step 1: Custom Catalog Configuration

Create or edit `~/.docker/mcp/catalogs/custom.yaml`:

```yaml
version: 2
name: custom
displayName: Custom MCP Servers
registry:
  mcp-swarm-intelligence:
    description: "Collective intelligence for multi-agent coordination featuring agent ecosystem management, hive mind knowledge bases, persistent memory systems, and automated workflow orchestration"
    title: "MCP Swarm Intelligence Server"
    type: server
    dateAdded: "2025-09-25T00:00:00Z"
    image: mcp-swarm-intelligence-server:latest
    ref: ""
    readme: ""
    toolsUrl: ""
    source: ""
    upstream: ""
    icon: ""
    tools:
      - name: agent_assignment
      - name: hive_mind_query
      - name: swarm_consensus
      - name: adaptive_coordination
      - name: agent_config_manager
      - name: knowledge_contribution
      - name: ecosystem_management
      - name: semantic_search
      - name: performance_metrics
      - name: decision_confidence
      - name: self_monitoring
    secrets:
      - name: SWARM_DB_ENCRYPTION_KEY
        env: SWARM_DB_ENCRYPTION_KEY
        example: "your-encryption-key-here"
      - name: SWARM_ADMIN_TOKEN
        env: SWARM_ADMIN_TOKEN
        example: "admin-token-123"
    metadata:
      category: automation
      tags:
        - swarm-intelligence
        - multi-agent
        - knowledge-management
        - coordination
        - machine-learning
      license: MIT
      owner: local
```

### Step 2: Registry Configuration

Edit `~/.docker/mcp/registry.yaml`:

```yaml
registry:
  # ... existing servers ...
  mcp-swarm-intelligence:
    ref: ""
```

### Step 3: Claude Desktop Integration

Configure `claude_desktop_config.json`:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mcp-toolkit-gateway": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",
        "-v", "/Users/your_username/.docker/mcp:/mcp",
        "docker/mcp-gateway",
        "--catalog=/mcp/catalogs/docker-mcp.yaml",
        "--catalog=/mcp/catalogs/custom.yaml",
        "--config=/mcp/config.yaml",
        "--registry=/mcp/registry.yaml",
        "--tools-config=/mcp/tools.yaml",
        "--transport=stdio"
      ]
    }
  }
}
```

## Usage Patterns

### Natural Language Examples

#### Agent Task Assignment
```
User: "I need to optimize our database queries. Which agent should handle this?"
Claude: Uses agent_assignment tool to analyze available agents and assign based on capabilities and performance scores.
```

#### Knowledge Management
```
User: "What does our team know about microservices architecture patterns?"
Claude: Uses hive_mind_query to search collective knowledge and return relevant insights.
```

#### Team Coordination
```
User: "Coordinate 4 agents to work on the frontend redesign project using a democratic approach"
Claude: Uses adaptive_coordination with democratic strategy to assign roles and create coordination plan.
```

#### Decision Support
```
User: "Should we migrate to Kubernetes? I want high confidence in this decision."
Claude: Uses swarm_consensus to gather agent opinions and decision_confidence to calculate metrics.
```

## Advanced Features

### Swarm Intelligence Algorithms

#### Ant Colony Optimization (ACO)
```python
def calculate_ant_colony_optimization(agents_data, task_complexity):
    pheromone_levels = np.random.random(len(agents_data))
    performance_scores = [agent['performance_score'] for agent in agents_data]
    combined_scores = np.array(performance_scores) * (1 + pheromone_levels)
    probabilities = combined_scores / np.sum(combined_scores)
    optimal_idx = np.argmax(probabilities)
    return agents_data[optimal_idx]
```

#### Democratic Consensus
```python
def calculate_consensus(options, agents, min_confidence=0.7):
    votes = {}
    total_weight = 0
    for agent_id, performance in agents:
        weight = float(performance)
        chosen_option = select_best_option(options, agent_preferences)
        votes[chosen_option] = votes.get(chosen_option, 0) + weight
        total_weight += weight
    return max(votes.items(), key=lambda x: x[1])
```

### Performance Optimization

#### Database Indexing
- FTS5 full-text search on knowledge base
- B-tree indexes on frequently queried columns
- Query optimization for complex joins

#### Memory Management
- Connection pooling for SQLite operations
- Async/await patterns for non-blocking I/O
- Efficient data structures for swarm calculations

#### Caching Strategies
- In-memory caching of frequently accessed agent data
- Result caching for expensive swarm calculations
- Intelligent cache invalidation on data updates

## Error Handling & Recovery

### Database Recovery
```python
async def handle_db_error(error):
    if "database is locked" in str(error):
        await asyncio.sleep(0.1)  # Retry after brief delay
        return await retry_operation()
    elif "no such table" in str(error):
        await init_database()  # Reinitialize schema
        return await retry_operation()
    else:
        logger.error(f"Unrecoverable database error: {error}")
        return default_response()
```

### Agent Coordination Failures
```python
async def handle_coordination_failure(agents, task):
    if len(agents) < 2:
        return await fallback_single_agent_mode()
    else:
        return await retry_with_reduced_agents(agents[:-1], task)
```

## Security Implementation

### Database Security
- Optional encryption via `SWARM_DB_ENCRYPTION_KEY`
- Parameterized queries to prevent SQL injection
- Connection timeout and retry limits

### Access Control
- Admin token validation for privileged operations
- Agent capability-based permissions
- Rate limiting on expensive operations

### Data Privacy
- No sensitive data logging
- Secure handling of agent configuration
- Optional data anonymization

## Development Guidelines

### Adding New Tools

1. **Tool Structure**:
```python
@mcp.tool()
async def new_tool(param: str = "") -> str:
    """Single-line description of functionality."""
    logger.info(f"Executing new_tool with {param}")
    
    try:
        # Validation
        if not param.strip():
            return "❌ Error: Parameter required"
        
        # Implementation
        result = await perform_operation(param)
        
        # Success response
        return f"✅ Operation completed: {result}"
        
    except Exception as e:
        logger.error(f"New tool error: {e}")
        return f"❌ Error: {str(e)}"
```

2. **Database Operations**:
```python
# Always use parameterized queries
await execute_query(
    "INSERT INTO table (column) VALUES (?)",
    (value,)
)

# Handle results properly
results = await execute_query("SELECT * FROM table")
if results:
    first_row = results[0]
```

3. **Error Handling**:
- Always include try-catch blocks
- Log errors for debugging
- Return user-friendly error messages
- Use consistent emoji indicators (✅ ❌ ⚠️)

### Testing Strategies

#### Unit Testing
```python
@pytest.mark.asyncio
async def test_agent_assignment():
    result = await agent_assignment("test task", "0.8")
    assert "✅" in result
    assert "confidence" in result.lower()
```

#### Integration Testing
```python
@pytest.mark.asyncio
async def test_full_workflow():
    # Add agent
    await agent_config_manager("add", "test-agent", "python,testing")
    
    # Assign task
    result = await agent_assignment("test task")
    
    # Verify assignment
    assert "test-agent" in result
```

#### Performance Testing
```python
@pytest.mark.benchmark
async def test_query_performance():
    start_time = time.time()
    result = await hive_mind_query("test query")
    duration = time.time() - start_time
    assert duration < 0.1  # Sub-100ms response
```

## Troubleshooting Guide

### Common Issues

#### "Tools not appearing in Claude"
1. Check Docker image build: `docker images | grep mcp-swarm`
2. Verify catalog syntax: `yaml-lint custom.yaml`
3. Restart Claude Desktop completely
4. Check logs: `docker logs <container_name>`

#### "Database connection errors"
1. Ensure data directory exists and is writable
2. Check SQLite version: `sqlite3 --version`
3. Verify file permissions: `ls -la data/memory.db`
4. Test connection: `sqlite3 data/memory.db ".tables"`

#### "Agent assignment not working"
1. Check agent registration: Use `agent_config_manager` list
2. Verify agent status: Ensure agents are "active"
3. Check performance scores: Must be between 0.0-1.0
4. Review task complexity: Ensure valid priority values

### Performance Issues

#### "Slow semantic search"
1. Check knowledge base size: `SELECT COUNT(*) FROM hive_knowledge`
2. Rebuild FTS5 index: `INSERT INTO hive_knowledge(hive_knowledge) VALUES('rebuild')`
3. Optimize query: Use more specific search terms
4. Consider pagination: Limit result count

#### "High memory usage"
1. Check database size: `du -h data/memory.db`
2. Run cleanup: Use `ecosystem_management` cleanup action
3. Monitor connections: Check for connection leaks
4. Restart container: `docker restart <container_name>`

## Integration Examples

### VS Code Extension Integration

```typescript
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    // Register MCP server definition
    const mcpProvider = vscode.lm.registerMcpServerDefinitionProvider({
        name: 'MCP Swarm Intelligence',
        description: 'Collective intelligence for multi-agent coordination',
        serverDefinition: {
            command: 'docker',
            args: ['run', '-i', '--rm', 'mcp-swarm-intelligence-server:latest'],
            transport: 'stdio'
        }
    });

    // Register chat participant
    const chatParticipant = vscode.chat.createChatParticipant(
        'swarm-intelligence',
        async (request, context, stream, token) => {
            // Use MCP tools through language model
            const response = await vscode.lm.sendRequest(
                'mcp-swarm-intelligence',
                'agent_assignment',
                { task_description: request.prompt }
            );
            
            stream.markdown(response);
        }
    );

    context.subscriptions.push(mcpProvider, chatParticipant);
}
```

### API Integration Examples

```python
# External API integration
async def integrate_with_external_api():
    async with httpx.AsyncClient() as client:
        # Get agent recommendations
        agents = await client.post('/api/agents/recommend', json={
            'task_type': 'optimization',
            'skill_requirements': ['python', 'database']
        })
        
        # Use MCP server for assignment
        assignment = await agent_assignment(
            agents['task_description'], 
            str(agents['priority'])
        )
        
        return assignment
```

## Future Enhancements

### Planned Features
- **Real-time Agent Discovery**: Automatic capability detection
- **Machine Learning Integration**: Adaptive performance optimization
- **Multi-modal Knowledge**: Support for images, documents
- **Distributed Coordination**: Cross-server agent coordination
- **Advanced Analytics**: Predictive performance modeling

### Extension Points
- **Custom Algorithms**: Pluggable swarm intelligence algorithms
- **External Integrations**: REST API endpoints for third-party systems
- **Workflow Templates**: Pre-defined coordination patterns
- **Knowledge Connectors**: Integration with external knowledge bases

This comprehensive guide provides everything needed to successfully integrate and use the MCP Swarm Intelligence Server with Claude Desktop, including advanced configuration, troubleshooting, and development guidelines.