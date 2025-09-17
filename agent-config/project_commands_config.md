# MCP Swarm Intelligence Server Project Commands & Development Guide

## Project Structure

```
mcp-swarm-server/
├── src/
│   ├── mcp_swarm/
│   │   ├── server/          # MCP server implementation
│   │   ├── swarm/           # Swarm intelligence algorithms
│   │   ├── memory/          # Persistent memory management
│   │   ├── agents/          # Agent ecosystem components
│   │   └── tools/           # MCP tools implementation
├── tests/                   # Comprehensive test suite
├── docs/                    # Documentation and guides
├── data/                    # Persistent data storage
├── agent-config/            # Agent configuration system
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
└── pyproject.toml          # Project metadata and configuration
```

## Essential Development Commands

### Current Development Status - MCP Foundation Setup Phase ✅ (September 17, 2025)

**Project State**: Setting up MCP Swarm Intelligence Server with agent configuration system
- **Agent Configuration**: ✅ Core MCP specialist agents configured and ready
- **Project Structure**: 🚧 Setting up Python-based MCP server implementation
- **Swarm Intelligence**: ✅ Swarm algorithms and coordination patterns designed
- **Memory Management**: ✅ Persistent memory and hive mind systems specified
- **MCP Protocol**: ✅ Protocol compliance and tool registration framework defined

**MCP Development Achievements**:
✅ Complete agent configuration system adapted for MCP development
✅ Orchestrator-driven workflow configured for Python/MCP development
✅ Swarm intelligence specialists configured for collective coordination
✅ Memory management system designed for persistent knowledge storage
✅ MCP protocol specialists ready for server implementation
✅ Testing framework designed for MCP protocol compliance validation

### Environment Setup Commands

```bash
# Create and setup project
mkdir mcp-swarm-server
cd mcp-swarm-server

# Setup Python virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install mcp>=1.0.0 fastapi uvicorn python-dotenv
pip install sqlalchemy sqlite-fts numpy scipy pydantic
pip install asyncio aiofiles watchfiles

# Install development dependencies
pip install pytest pytest-asyncio pytest-cov black flake8 mypy
pip install pre-commit isort bandit safety

# Install optional dependencies for enhanced functionality
pip install redis psutil colorlog structlog

# Save dependencies
pip freeze > requirements.txt

# Create development requirements
echo "pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
pre-commit>=3.0.0
isort>=5.12.0
bandit>=1.7.0
safety>=2.3.0" > requirements-dev.txt
```

### Project Initialization Commands

```bash
# Initialize project structure
mkdir -p src/mcp_swarm/{server,swarm,memory,agents,tools}
mkdir -p tests/{unit,integration,performance}
mkdir -p docs/{api,guides,examples}
mkdir -p data/{memory,patterns,logs}

# Create agent-config directory (copy from Agent-Config-MCP project)
cp -r /path/to/Agent-Config-MCP/agent-config ./agent-config

# Initialize git repository
git init
git add .
git commit -m "Initial MCP Swarm Server setup"

# Setup pre-commit hooks
pre-commit install

# Create .env file for development
echo "# MCP Swarm Intelligence Server Configuration
PYTHONPATH=src
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8000
AGENT_CONFIG_DIR=./agent-config
SWARM_LOG_LEVEL=INFO
HIVE_MIND_RETENTION_DAYS=30
MEMORY_DB_PATH=./data/memory/hive_mind.db" > .env
```

### Development Workflow Commands

```bash
# Development server (with auto-reload)
python src/mcp_swarm/server.py --dev

# Run MCP server in production mode
python src/mcp_swarm/server.py

# Run with specific configuration
AGENT_CONFIG_DIR=/custom/path python src/mcp_swarm/server.py

# Test MCP server connection
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1}'

# Check server health
curl http://localhost:8000/health
```

### Testing Commands

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/                    # Unit tests
pytest tests/integration/             # Integration tests
pytest tests/performance/             # Performance tests

# Run tests with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test files
pytest tests/unit/test_mcp_protocol.py         # MCP protocol tests
pytest tests/unit/test_swarm_intelligence.py  # Swarm algorithm tests
pytest tests/unit/test_memory_management.py   # Memory system tests
pytest tests/integration/test_agent_coordination.py  # Agent coordination tests

# Run tests with verbose output
pytest -v tests/

# Run tests in parallel (install pytest-xdist first)
pip install pytest-xdist
pytest -n auto tests/

# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Test MCP protocol compliance
pytest tests/integration/test_mcp_compliance.py -v
```

### Code Quality Commands

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security scanning
bandit -r src/

# Check for vulnerabilities in dependencies
safety check

# Run all quality checks
black src/ tests/ && isort src/ tests/ && flake8 src/ tests/ && mypy src/ && bandit -r src/

# Pre-commit checks (runs automatically on git commit)
pre-commit run --all-files
```

### MCP-Specific Commands

```bash
# Test MCP tool registration
python -c "
from src.mcp_swarm.server import SwarmMCPServer
import asyncio
async def test():
    server = SwarmMCPServer()
    tools = await server.list_tools()
    print(f'Registered tools: {[tool.name for tool in tools]}')
asyncio.run(test())
"

# Test agent assignment functionality
python -c "
from src.mcp_swarm.swarm import SwarmCoordinator
coordinator = SwarmCoordinator()
assignment = coordinator.optimal_assignment({
    'requirements': ['python_development', 'mcp_protocol'],
    'complexity': 'medium'
})
print(f'Optimal assignment: {assignment}')
"

# Test memory persistence
python -c "
from src.mcp_swarm.memory import PersistentMemory
memory = PersistentMemory('./data/memory/test.db')
memory.store_pattern('test_pattern', {'success': True})
patterns = memory.get_patterns('test_pattern')
print(f'Stored patterns: {patterns}')
"

# Validate agent configuration files
python src/mcp_swarm/agents/validate_configs.py ./agent-config

# Generate agent ecosystem overview
python src/mcp_swarm/agents/ecosystem_overview.py ./agent-config
```

### Database and Memory Commands

```bash
# Initialize SQLite database with FTS5 extension
python -c "
from src.mcp_swarm.memory import PersistentMemory
memory = PersistentMemory('./data/memory/hive_mind.db')
memory.initialize_database()
print('Database initialized successfully')
"

# Backup memory database
cp ./data/memory/hive_mind.db ./data/memory/hive_mind_backup_$(date +%Y%m%d_%H%M%S).db

# Analyze memory usage and patterns
python src/mcp_swarm/memory/analyze_patterns.py

# Clean up old patterns (older than retention period)
python src/mcp_swarm/memory/cleanup.py --retention-days 30

# Export patterns for analysis
python src/mcp_swarm/memory/export_patterns.py --format json --output ./data/patterns_export.json
```

### Deployment Commands

```bash
# Build Docker image
docker build -t mcp-swarm-server:latest .

# Run with Docker
docker run -d \
  --name mcp-swarm-server \
  -p 8000:8000 \
  -v $(pwd)/agent-config:/app/agent-config:ro \
  -v $(pwd)/data:/app/data \
  -e AGENT_CONFIG_DIR=/app/agent-config \
  mcp-swarm-server:latest

# Run with Docker Compose
docker-compose up -d

# Check container logs
docker logs mcp-swarm-server

# Update deployed version
docker-compose pull && docker-compose up -d

# Backup deployed data
docker exec mcp-swarm-server tar czf /tmp/data_backup.tar.gz /app/data
docker cp mcp-swarm-server:/tmp/data_backup.tar.gz ./backup_$(date +%Y%m%d_%H%M%S).tar.gz
```

### Monitoring and Debugging Commands

```bash
# Monitor server logs
tail -f ./data/logs/mcp_server.log

# Monitor memory usage
python src/mcp_swarm/monitoring/memory_monitor.py

# Monitor swarm coordination performance
python src/mcp_swarm/monitoring/swarm_monitor.py

# Debug agent assignments
python src/mcp_swarm/debug/assignment_debugger.py --task-type implementation

# Debug MCP protocol messages
PYTHONPATH=src python -c "
from src.mcp_swarm.debug import MCPMessageDebugger
debugger = MCPMessageDebugger()
debugger.start_monitoring()
"

# Performance profiling
python -m cProfile -o profile_output.prof src/mcp_swarm/server.py
python -c "
import pstats
p = pstats.Stats('profile_output.prof')
p.sort_stats('cumulative').print_stats(20)
"
```

### Documentation Commands

```bash
# Generate API documentation
python src/mcp_swarm/docs/generate_api_docs.py

# Generate agent configuration documentation
python src/mcp_swarm/docs/generate_agent_docs.py ./agent-config

# Generate usage examples
python src/mcp_swarm/docs/generate_examples.py

# Serve documentation locally
python -m http.server 8080 --directory docs/

# Generate complete documentation package
python src/mcp_swarm/docs/build_docs.py --output ./docs/complete/
```

### Configuration Management Commands

```bash
# Validate agent configuration files
python src/mcp_swarm/config/validate_agent_configs.py

# Generate agent intersection matrix
python src/mcp_swarm/config/generate_intersection_matrix.py

# Test agent routing matrix
python src/mcp_swarm/config/test_routing.py --task-type mcp_development

# Update agent capabilities
python src/mcp_swarm/config/update_capabilities.py --agent python_specialist --add mcp_testing

# Backup current configuration
tar czf agent_config_backup_$(date +%Y%m%d_%H%M%S).tar.gz agent-config/
```

### Integration Testing Commands

```bash
# Test VS Code MCP client integration
python tests/integration/test_vscode_integration.py

# Test MCP protocol compliance
python tests/integration/test_mcp_compliance.py

# Test multi-client scenarios
python tests/integration/test_multi_client.py

# Load testing
python tests/performance/load_test.py --concurrent-clients 50 --duration 300

# End-to-end testing
python tests/integration/test_e2e_workflow.py
```

### Troubleshooting Commands

```bash
# Check Python environment
python --version
pip list | grep -E "(mcp|fastapi|sqlalchemy)"

# Verify MCP server dependencies
python -c "
import mcp
import fastapi
import sqlalchemy
print('All core dependencies available')
"

# Test SQLite FTS5 support
python -c "
import sqlite3
conn = sqlite3.connect(':memory:')
conn.execute('CREATE VIRTUAL TABLE test USING fts5(content)')
print('SQLite FTS5 support available')
conn.close()
"

# Check agent configuration loading
python -c "
from src.mcp_swarm.agents import AgentEcosystem
ecosystem = AgentEcosystem('./agent-config')
agents = ecosystem.load_all_agents()
print(f'Loaded {len(agents)} agent configurations')
"

# Verify swarm intelligence algorithms
python -c "
from src.mcp_swarm.swarm import ACOEngine, PSOEngine
aco = ACOEngine()
pso = PSOEngine()
print('Swarm intelligence algorithms initialized successfully')
"
```

### Development Best Practices

1. **Always run tests before committing**:
   ```bash
   pytest && black src/ tests/ && flake8 src/ tests/
   ```

2. **Use virtual environments for isolation**:
   ```bash
   python -m venv venv && source venv/bin/activate
   ```

3. **Keep dependencies updated**:
   ```bash
   pip list --outdated && pip install --upgrade package_name
   ```

4. **Monitor memory usage during development**:
   ```bash
   python src/mcp_swarm/monitoring/dev_monitor.py
   ```

5. **Validate MCP protocol compliance regularly**:
   ```bash
   pytest tests/integration/test_mcp_compliance.py -v
   ```

6. **Backup data before major changes**:
   ```bash
   tar czf backup_$(date +%Y%m%d_%H%M%S).tar.gz data/ agent-config/
   ```

## Project-Specific Configuration

### Environment Variables

```bash
# Required
export PYTHONPATH=src
export AGENT_CONFIG_DIR=./agent-config

# Optional
export MCP_SERVER_HOST=localhost
export MCP_SERVER_PORT=8000
export SWARM_LOG_LEVEL=INFO
export HIVE_MIND_RETENTION_DAYS=30
export MEMORY_DB_PATH=./data/memory/hive_mind.db
export MAX_AGENT_ASSIGNMENTS=10
export SWARM_OPTIMIZATION_INTERVAL=3600
```

### Performance Optimization

```bash
# Enable SQLite performance optimizations
export SQLITE_ENABLE_FTS5=1
export SQLITE_ENABLE_JSON1=1
export SQLITE_DEFAULT_WAL_MODE=1

# Python performance optimizations
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Async performance tuning
export UVLOOP_ENABLED=1
export ASYNCIO_DEBUG=0
```

This comprehensive command guide provides all necessary tools for developing, testing, and deploying the MCP Swarm Intelligence Server with proper quality assurance and monitoring capabilities.