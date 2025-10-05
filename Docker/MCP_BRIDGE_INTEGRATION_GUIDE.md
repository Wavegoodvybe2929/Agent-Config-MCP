# MCP Swarm Intelligence Server - Docker Integration Guide

## 🎯 Successfully Built & Ready for Integration

Your **MCP Swarm Intelligence Server** Docker image has been successfully built and tested! The image is now available locally as:

- **Image Name**: `mcp-swarm-server:latest`
- **Size**: ~1.04GB
- **Status**: ✅ Production Ready

## 🐳 Available Docker Images

```bash
# List your local MCP images
docker images | grep mcp

# You should see:
# mcp-swarm-server:latest
# mcp-swarm-intelligence-server:latest (alternative tag)
```

## 🚀 MCP Bridge Integration Options

### Option 1: VS Code Integration

Add to your VS Code `settings.json`:

```json
{
  "mcpServers": {
    "mcp-swarm-intelligence": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "/tmp/mcp-data:/app/data",
        "mcp-swarm-server:latest"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

### Option 2: Claude Desktop Integration

Add to your Claude Desktop MCP configuration file (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "mcp-swarm-intelligence": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "/tmp/mcp-data:/app/data",
        "mcp-swarm-server:latest"
      ]
    }
  }
}
```

### Option 3: Direct Docker Usage (Interactive)

```bash
# Run the server interactively with stdio
docker run --rm -i \\
  -v "$(pwd)/mcp-data:/app/data" \\
  mcp-swarm-server:latest

# Or run as a background service with port mapping
docker run -d --name mcp-swarm \\
  -p 8080:8080 \\
  -v "$(pwd)/mcp-data:/app/data" \\
  mcp-swarm-server:latest
```

## 🛠️ Available MCP Tools (29 Total)

Your Docker image includes all 29 swarm intelligence tools:

### 🐝 Core Swarm Intelligence
- `agent_assignment` - Optimal task assignment using ACO algorithms
- `swarm_consensus` - Democratic decision-making with PSO
- `adaptive_coordination` - Dynamic multi-agent coordination
- `coordination_strategies` - Optimal coordination pattern selection

### 🧠 Knowledge & Hive Mind
- `hive_mind_query` - Semantic search across collective knowledge
- `knowledge_contribution` - Contribute to hive mind database
- `semantic_search` - Advanced semantic search capabilities
- `knowledge_synthesis_tool` - Synthesize multi-source knowledge

### 💾 Memory & Persistence
- `memory_management` - Core persistent memory operations
- `persistent_memory_manager` - Advanced memory management
- `pattern_analysis_tool` - Pattern recognition and analysis

### ⚙️ Configuration & Management
- `agent_config_manager` - Agent configuration management
- `copilot_instructions_manager` - Copilot instructions management
- `ecosystem_management` - Agent ecosystem health monitoring
- `agent_hooks` - Agent lifecycle hook execution

### 📊 Performance & Monitoring
- `performance_metrics` - Comprehensive performance monitoring
- `self_monitoring_tool` - System self-monitoring
- `predictive_maintenance_tool` - Predictive maintenance
- `resource_optimization_tool` - Resource optimization

### 🔍 Quality & Validation
- `quality_assurance_tool` - Automated quality assurance
- `compliance_validator_tool` - Standards compliance validation
- `anomaly_detection_tool` - System anomaly detection
- `knowledge_quality_validator` - Knowledge quality validation

### 🚀 Advanced Intelligence
- `adaptive_learning_tool` - Adaptive learning and evolution
- `workflow_automation_tool` - Automated workflow management

### 🔧 Utilities
- `directory_manager` - File system operations
- `confidence_aggregation` - Multi-agent confidence scoring
- `consensus_algorithms` - Democratic consensus mechanisms
- `decision_audit` - Decision process auditing

## 📁 Data Persistence

The container uses `/app/data` for SQLite database storage. Mount this volume to persist data:

```bash
# Create local data directory
mkdir -p ./mcp-data

# Run with persistent data
docker run --rm -i \\
  -v "$(pwd)/mcp-data:/app/data" \\
  mcp-swarm-server:latest
```

## 🔧 Environment Configuration

Optional environment variables:

```bash
# Database configuration
SWARM_DB_PATH=/app/data/memory.db
SWARM_DB_ENCRYPTION_KEY=your_key_here  # Optional encryption

# Security configuration  
SWARM_ADMIN_TOKEN=your_admin_token      # Optional admin access

# Logging configuration
PYTHONUNBUFFERED=1                      # Unbuffered output
```

## 🧪 Testing the Integration

Test your MCP bridge integration:

```bash
# 1. Check if the image is available
docker images mcp-swarm-server

# 2. Test basic functionality
echo '{"method": "tools/list"}' | docker run --rm -i mcp-swarm-server:latest

# 3. Test with data persistence
mkdir -p ./test-data
docker run --rm -i -v "$(pwd)/test-data:/app/data" mcp-swarm-server:latest
```

## 🚨 Troubleshooting

### Common Issues:

1. **"Parameter cannot start with '_'" Error**: 
   - ✅ **FIXED** - All underscore parameters have been corrected

2. **Permission Issues**:
   ```bash
   # Fix data directory permissions
   sudo chown -R 1000:1000 ./mcp-data
   ```

3. **Container Exits Immediately**:
   - This is normal for MCP servers using stdio
   - Use `-i` flag for interactive mode
   - Check logs: `docker logs <container_name>`

### Health Check:
```bash
# Verify the server initializes correctly
docker run --rm mcp-swarm-server:latest 2>&1 | head -10
```

## 🎯 Next Steps

1. **Choose your integration method** (VS Code, Claude Desktop, or direct Docker)
2. **Configure the MCP bridge** with the appropriate JSON configuration
3. **Test the connection** using your chosen MCP client
4. **Start using the 29 available swarm intelligence tools**

## 📋 Integration Checklist

- ✅ Docker image built successfully (`mcp-swarm-server:latest`)
- ✅ All 29 MCP tools available and functional
- ✅ Parameter naming issues resolved
- ✅ SQLite database with persistent storage
- ✅ Non-root security configuration
- ✅ Volume mounting for data persistence
- ✅ Environment variable support
- ✅ MCP protocol compliance verified

Your MCP Swarm Intelligence Server is now ready for production use with the MCP bridge! 🚀