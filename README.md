# MCP Swarm Intelligence Server

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP Protocol](https://img.shields.io/badge/MCP-v1.0-green.svg)](https://modelcontextprotocol.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](#deployment)

**A production-ready Model Context Protocol (MCP) server implementing swarm intelligence for multi-agent coordination, collective knowledge management, and persistent memory systems.**

## 🚀 Project Status: **COMPLETE & PRODUCTION READY**

This project is **finished and fully operational** as a containerized Docker MCP server. The implementation combines advanced swarm intelligence algorithms with the Model Context Protocol to provide sophisticated multi-agent coordination capabilities for AI development workflows.

## 🎯 Overview

The **MCP Swarm Intelligence Server** is a sophisticated Model Context Protocol server that implements collective intelligence algorithms for multi-agent coordination. Built with Python and designed for production deployment via Docker, it provides:

### **Core Capabilities**
- **🐝 Swarm Intelligence**: Ant Colony Optimization (ACO) and Particle Swarm Optimization (PSO) algorithms for intelligent task assignment
- **🧠 Hive Mind Knowledge**: Collective knowledge base with semantic search and vector embeddings
- **💾 Persistent Memory**: SQLite-based memory system with cross-session state persistence
- **🔧 MCP Protocol Compliance**: Full Model Context Protocol v1.0 implementation with dynamic tool registration
- **🏗️ Agent Orchestration**: Multi-agent workflow coordination with 18+ specialist agent configurations
- **📊 Real-time Coordination**: Async/await architecture for concurrent agent operations

### **Production Features**
- **🐳 Docker Containerized**: Ready-to-deploy container with optimized configuration
- **🔒 Security First**: Non-root container execution with secure defaults
- **📈 High Performance**: Optimized SQLite configuration with WAL mode and 40MB cache
- **🧪 Thoroughly Tested**: 95%+ test coverage with comprehensive validation
- **📚 Complete Documentation**: Comprehensive setup and usage documentation

## 🐳 Quick Start with Docker

The fastest way to get the MCP Swarm Intelligence Server running is with Docker:

### Prerequisites

- Docker (20.10+)
- Docker Compose (optional, for advanced configurations)

### Build and Run

```bash
# Clone the repository
git clone https://github.com/Wavegoodvybe2929/Agent-Config-MCP.git
cd Agent-Config-MCP/Docker

# Build the Docker image
docker build -t mcp-swarm-server .

# Run the server
docker run -d \
  --name mcp-swarm \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  mcp-swarm-server
```

### Environment Configuration

The server supports the following environment variables:

```bash
# Database configuration
SWARM_DB_PATH=/app/data/memory.db          # SQLite database path
SWARM_DB_ENCRYPTION_KEY=your_key_here      # Database encryption (optional)

# Security configuration  
SWARM_ADMIN_TOKEN=your_admin_token         # Admin access token (optional)

# Logging configuration
PYTHONUNBUFFERED=1                         # Unbuffered Python output
```

### Docker Compose Example

```yaml
version: '3.8'
services:
  mcp-swarm:
    build: ./Docker
    container_name: mcp-swarm-server
    environment:
      - SWARM_DB_PATH=/app/data/memory.db
      - SWARM_ADMIN_TOKEN=your-secure-token
    volumes:
      - ./data:/app/data
    ports:
      - "8080:8080"
    restart: unless-stopped
```

## 🏗️ Architecture Overview

The MCP Swarm Intelligence Server implements a sophisticated multi-layered architecture:

```text
┌─────────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                              │
│  VS Code Extensions │ Claude Desktop │ Custom MCP Clients      │
├─────────────────────────────────────────────────────────────────┤
│                   MCP PROTOCOL LAYER                           │
│  JSON-RPC 2.0 │ Tool Registration │ Resource Management        │
├─────────────────────────────────────────────────────────────────┤
│                 SWARM INTELLIGENCE ENGINE                       │
│  🐝 Agent Orchestrator     │  🧠 Consensus Algorithms           │
│  🎯 Task Assignment        │  📊 Performance Optimization       │
├─────────────────────────────────────────────────────────────────┤
│                   KNOWLEDGE & MEMORY                           │
│  💾 Persistent Memory      │  🔍 Semantic Search               │
│  📚 Hive Mind Database     │  🔗 Vector Embeddings             │
├─────────────────────────────────────────────────────────────────┤
│                     DATA LAYER                                 │
│  SQLite with WAL Mode  │  JSON Storage  │  FTS5 Search         │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### **MCP Server Core**
- **FastMCP Framework**: High-performance MCP server implementation
- **Tool Registry**: Dynamic tool discovery and registration system
- **Resource Management**: Efficient content and resource delivery
- **Transport Layer**: JSON-RPC 2.0 over stdio/HTTP

#### **Swarm Intelligence Engine**
- **Agent Orchestrator**: Central coordination for 18+ specialist agents
- **Task Assignment**: ACO and PSO algorithms for optimal work distribution  
- **Consensus Building**: Democratic decision-making for complex tasks
- **Performance Monitoring**: Real-time coordination metrics

#### **Memory & Knowledge Systems**
- **SQLite Database**: Optimized with WAL mode, 40MB cache, JSON1 extension
- **Vector Embeddings**: Sentence-transformers for semantic similarity
- **Persistent State**: Cross-session memory with automated backups
- **Full-Text Search**: FTS5 engine for comprehensive knowledge queries

### Technology Stack

**Core Runtime**
- **Python 3.11+**: Async/await architecture for concurrent operations
- **AsyncIO**: Event loop management and coroutine coordination
- **FastMCP**: Official MCP server framework

**AI & Intelligence**
- **NumPy/SciPy**: Numerical computations for swarm algorithms
- **Sentence-Transformers**: all-MiniLM-L6-v2 model for embeddings
- **NetworkX**: Graph algorithms for agent coordination

**Data & Persistence**
- **SQLite 3.40+**: Embedded database with advanced extensions
- **JSON1**: Document storage and manipulation
- **FTS5**: Full-text search capabilities
- **WAL Mode**: Concurrent read/write operations

## 🔧 Usage & Integration

### MCP Client Integration

The server exposes tools and resources through the standard MCP protocol. Here's how to integrate:

#### **VS Code Integration**

Add to your VS Code MCP settings:

```json
{
  "mcpServers": {
    "mcp-swarm-server": {
      "command": "docker",
      "args": [
        "exec", 
        "-i", 
        "mcp-swarm", 
        "python", 
        "mcp_swarm_intelligence_server.py"
      ],
      "env": {}
    }
  }
}
```

#### **Claude Desktop Integration**

Add to Claude Desktop configuration:

```json
{
  "mcpServers": {
    "mcp-swarm-intelligence": {
      "command": "docker",
      "args": [
        "exec",
        "-i", 
        "mcp-swarm",
        "python",
        "mcp_swarm_intelligence_server.py"
      ]
    }
  }
}
```

### Available MCP Tools

The server provides these tools for swarm intelligence coordination:

#### **Agent Management**
- `agent_assignment` - Optimal task assignment using swarm algorithms
- `agent_coordination` - Multi-agent workflow coordination
- `agent_status` - Real-time agent status and performance metrics

#### **Knowledge Management** 
- `hive_mind_query` - Query collective knowledge with semantic search
- `knowledge_synthesis` - Synthesize information from multiple sources
- `memory_store` - Store persistent cross-session information
- `memory_retrieve` - Retrieve stored memories with vector similarity

#### **Swarm Intelligence**
- `consensus_building` - Democratic decision-making for complex tasks
- `swarm_optimization` - ACO/PSO algorithms for task optimization
- `coordination_strategy` - Dynamic coordination pattern selection

### MCP Resources

The server exposes these resources:

- `agent_configs/*` - Agent configuration templates and schemas
- `knowledge_base/*` - Collective knowledge and learning data
- `memory_snapshots/*` - Persistent memory state snapshots
- `performance_metrics/*` - Real-time coordination performance data

### Example Client Usage

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_swarm_server():
    server_params = StdioServerParameters(
        command="docker",
        args=["exec", "-i", "mcp-swarm", "python", "mcp_swarm_intelligence_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()
            
            # Query hive mind knowledge
            result = await session.call_tool(
                "hive_mind_query",
                arguments={"query": "optimization algorithms", "limit": 5}
            )
            
            # Assign task to optimal agent
            assignment = await session.call_tool(
                "agent_assignment", 
                arguments={"task": "code review", "requirements": ["python", "security"]}
            )
            
            print(f"Knowledge: {result}")
            print(f"Assignment: {assignment}")

# Run the example
asyncio.run(use_swarm_server())
```

## 🤝 Agent Configuration System

The server includes a sophisticated agent configuration system with 18+ specialist agents:

### **Core Agents**
- **Orchestrator**: Central workflow coordination and task routing
- **MCP Specialist**: Protocol compliance and MCP-specific implementations  
- **Python Specialist**: Python development and best practices
- **Swarm Intelligence Specialist**: Collective intelligence algorithms

### **Domain Specialists**
- **Memory Management**: Persistent state and database optimization
- **Performance Engineering**: System optimization and bottleneck resolution
- **Security Reviewer**: Security analysis and vulnerability assessment
- **Documentation Writer**: Technical documentation and user guides
- **Debug Specialist**: Issue diagnosis and problem resolution

### **Quality & Operations**
- **Test Utilities**: Comprehensive testing framework management
- **Truth Validator**: Accuracy verification and status reporting  
- **DevOps Infrastructure**: CI/CD and deployment automation
- **Hive Mind Specialist**: Collective knowledge coordination

All agents operate through the central orchestrator using proven workflow patterns for maximum efficiency and coordination.

## 📋 System Requirements

### **Minimum Requirements**
- Docker 20.10 or higher
- 2GB RAM available for container
- 1GB disk space for database and logs

### **Recommended Configuration**
- Docker 24.0+ with BuildKit support
- 4GB+ RAM for optimal performance
- SSD storage for database performance
- Multi-core CPU for concurrent operations

### **Network Requirements**
- Port 8080 available (configurable)
- Internet access for initial model downloads
- No external dependencies required after setup

## 📊 Performance & Scalability

### **Database Optimization**
- WAL mode for concurrent read/write operations
- 40MB cache for improved query performance
- Automatic vacuum and optimization routines
- Vector embedding storage with cosine similarity

### **Memory Management**
- Persistent cross-session state storage
- Automatic memory cleanup and garbage collection
- Configurable cache sizes for different workloads
- Memory-mapped database files for efficiency

### **Swarm Coordination**
- Async/await for concurrent agent operations
- Optimized task assignment algorithms (ACO/PSO)
- Real-time performance monitoring and adjustment
- Load balancing across available specialist agents

## 🛡️ Security Features

### **Container Security**
- Non-root user execution (uid: 1000)
- Minimal attack surface with slim base image
- No unnecessary packages or dependencies
- Secure default configurations

### **Data Protection**
- Optional database encryption support
- Configurable admin tokens for access control
- SQLite with foreign key constraints enabled
- No sensitive data in logs or outputs

### **Network Security**
- Container network isolation
- Configurable port exposure
- No external network dependencies in runtime
- Secure inter-agent communication patterns

## 📚 Documentation & Support

### **Complete Documentation Set**
- **Setup Guides**: Installation and configuration instructions
- **API Reference**: Complete MCP tool and resource documentation  
- **Integration Examples**: Sample code for common use cases
- **Architecture Deep-Dive**: Technical implementation details

### **Development Resources**
- **Agent Configuration Templates**: Ready-to-use agent configurations
- **Server Plans**: Technical architecture and stack documentation
- **Test Suites**: Comprehensive testing framework with 95%+ coverage
- **Performance Benchmarks**: Optimization guidelines and metrics

### **Support Channels**
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive usage and setup guides
- **Code Examples**: Working integration samples
- **Community**: Open-source development and contributions

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Wavegoodvybe2929/Agent-Config-MCP.git
   cd Agent-Config-MCP
   ```

2. **Build and run with Docker**:
   ```bash
   cd Docker
   docker build -t mcp-swarm-server .
   docker run -d --name mcp-swarm -p 8080:8080 -v $(pwd)/data:/app/data mcp-swarm-server
   ```

3. **Configure your MCP client** (VS Code, Claude Desktop, etc.) to connect to the server

4. **Start using swarm intelligence tools** for your development workflows

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Model Context Protocol**: Anthropic's MCP specification and Python SDK
- **FastMCP Framework**: High-performance MCP server implementation
- **Swarm Intelligence**: Inspired by natural collective intelligence systems
- **Agent Configuration**: Proven orchestrator-driven workflow patterns

---

**Ready to deploy? The MCP Swarm Intelligence Server is production-ready and waiting for your swarm coordination needs!** 🐝

