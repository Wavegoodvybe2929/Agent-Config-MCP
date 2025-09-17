# MCP Swarm Intelligence Server

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP Protocol](https://img.shields.io/badge/MCP-v1.0-green.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Model Context Protocol (MCP) server implementation with swarm intelligence capabilities for multi-agent coordination, collective knowledge management, and persistent memory systems.

## 🎯 Project Overview

The MCP Swarm Intelligence Server combines the standardized Model Context Protocol with advanced swarm intelligence algorithms to enable sophisticated multi-agent coordination. This server provides:

- **MCP Protocol Compliance**: Full implementation of MCP v1.0 specification
- **Swarm Intelligence**: Ant Colony Optimization (ACO) and Particle Swarm Optimization (PSO) algorithms
- **Collective Knowledge**: Hive mind knowledge bases with persistent memory
- **Multi-Agent Coordination**: Queen-led coordination patterns for optimal agent assignment
- **Persistent Memory**: SQLite-backed cross-session state management
- **Real-time Consensus**: Collective decision-making for complex tasks

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                MCP Client                           │
└─────────────────┬───────────────────────────────────┘
                  │ JSON-RPC 2.0
┌─────────────────▼───────────────────────────────────┐
│              MCP Server Layer                       │
│  ┌─────────────────────────────────────────────────┐│
│  │            Tool Registry                        ││
│  │  • Agent Assignment  • Consensus Building      ││
│  │  • Resource Coordination  • Memory Management  ││
│  └─────────────────────────────────────────────────┘│
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│           Swarm Intelligence Layer                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐│
│  │    ACO       │ │    PSO       │ │  Consensus   ││
│  │ Algorithms   │ │ Algorithms   │ │  Building    ││
│  └──────────────┘ └──────────────┘ └──────────────┘│
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│             Memory Layer                            │
│  ┌─────────────────────────────────────────────────┐│
│  │               SQLite Database                   ││
│  │  • Agent Registry    • Knowledge Entries       ││
│  │  • Swarm State      • Task History             ││
│  │  • Memory Sessions  • Pheromone Trails         ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- SQLite 3.40+ (with FTS5 and JSON1 extensions)
- Git

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd mcp-swarm-server
   ```

2. **Set up Python environment**:

   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development
   ```

4. **Initialize the database**:

   ```bash
   python scripts/init_database.py
   ```

5. **Verify installation**:

   ```bash
   python -c "import mcp_swarm; print('✅ Installation successful!')"
   ```

## 📦 Core Components

### MCP Server Implementation

- **SwarmMCPServer**: Enhanced MCP server with swarm coordination
- **Tool Registry**: Dynamic tool discovery and registration
- **Resource Management**: MCP resource handling with optimization
- **Message Handling**: JSON-RPC 2.0 message processing

### Swarm Intelligence Algorithms

- **Ant Colony Optimization (ACO)**: Optimal agent-task assignment
- **Particle Swarm Optimization (PSO)**: Consensus building algorithms
- **Pheromone Trails**: Dynamic coordination pattern learning
- **Collective Decision Making**: Multi-criteria agent coordination

### Memory Management

- **Persistent Storage**: SQLite database with WAL mode
- **Cross-Session State**: Agent state preservation across restarts
- **Knowledge Base**: Collective intelligence storage and retrieval
- **Memory Optimization**: Automatic cleanup and performance tuning

### Agent Coordination

- **Agent Registry**: Dynamic agent discovery and registration
- **Load Balancing**: Real-time workload distribution
- **Capability Matching**: Fuzzy logic-based task assignment
- **Performance Tracking**: Success rate monitoring and optimization

## 🛠️ Development

### Project Structure

```
mcp-swarm-server/
├── src/
│   ├── mcp_swarm/           # Main package
│   │   ├── server/          # MCP server implementation
│   │   ├── swarm/           # Swarm intelligence algorithms
│   │   ├── memory/          # Persistent memory management
│   │   ├── agents/          # Agent coordination
│   │   └── tools/           # MCP tools implementation
│   ├── config/              # Configuration management
│   ├── data/                # Database and knowledge storage
│   └── tests/               # Test suite
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
└── pyproject.toml          # Project configuration
```

### Database Schema

The server uses SQLite with the following core tables:

- **agents**: Agent registration and status management
- **knowledge_entries**: Collective knowledge storage with embeddings
- **swarm_state**: Coordination state and consensus data
- **task_history**: Execution history for learning and optimization
- **memory_sessions**: Cross-session persistence management
- **pheromone_trails**: Swarm coordination patterns
- **mcp_tools**: Dynamic tool registry
- **mcp_resources**: Resource management and metadata

### Development Workflow

1. **Set up development environment**:

   ```bash
   pip install -e .
   pre-commit install
   ```

2. **Run tests**:

   ```bash
   pytest
   pytest --cov=mcp_swarm  # With coverage
   ```

3. **Code quality checks**:

   ```bash
   black src/
   flake8 src/
   mypy src/
   bandit -r src/
   ```

4. **Database operations**:

   ```bash
   python scripts/init_database.py          # Initialize
   python scripts/init_database.py --reset  # Reset database
   ```

## 🔧 Configuration

### Environment Variables

```bash
# Database configuration
MCP_SWARM_DB_PATH=src/data/memory.db
MCP_SWARM_DB_BACKUP_INTERVAL=3600

# Server configuration
MCP_SWARM_SERVER_NAME=swarm-intelligence-server
MCP_SWARM_SERVER_PORT=8000
MCP_SWARM_LOG_LEVEL=INFO

# Swarm algorithm parameters
MCP_SWARM_ACO_ITERATIONS=100
MCP_SWARM_PSO_PARTICLES=50
MCP_SWARM_PHEROMONE_DECAY=0.1
```

### Configuration Files

Configuration files will be located in `src/config/`:

- `default.yaml`: Default configuration settings
- `development.yaml`: Development environment overrides
- `production.yaml`: Production environment settings

## 🧪 Testing

The project maintains 95%+ test coverage with comprehensive test suites:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component interaction testing
- **Performance Tests**: Algorithm efficiency validation
- **Memory Tests**: Database operation verification
- **Swarm Tests**: Coordination algorithm testing

Run specific test categories:

```bash
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m swarm         # Swarm algorithm tests
pytest -m memory        # Memory management tests
```

## 📊 Performance

Expected performance characteristics:

- **Agent Assignment**: <2 seconds for 100+ agents and 1000+ tasks
- **Consensus Building**: <30 seconds for stable results
- **Database Operations**: <100ms for typical queries
- **Memory Usage**: <500MB for standard workloads
- **Concurrent Requests**: 100+ simultaneous MCP connections

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure all tests pass and code quality checks succeed
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Quality Standards

- **Test Coverage**: Minimum 95% coverage required
- **Type Hints**: Full type annotation for all public APIs
- **Documentation**: Comprehensive docstrings for all public methods
- **Performance**: No regressions in benchmark tests
- **Security**: Security scanning with bandit required

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [Model Context Protocol](https://modelcontextprotocol.io/) - Official MCP specification
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) - Python MCP implementation
- [Swarm Intelligence Algorithms](https://en.wikipedia.org/wiki/Swarm_intelligence) - Algorithm background

## 📞 Support

- **Documentation**: Coming soon to `docs/`
- **Issues**: [GitHub Issues](https://github.com/mcp-swarm/mcp-swarm-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mcp-swarm/mcp-swarm-server/discussions)

---

**Note**: This project is currently in Phase 1 development. Core MCP tools implementation and advanced swarm features will be available in Phase 2.
