# Agent-Config-MCP: Orchestrator-Driven MCP Swarm Intelligence Server

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP Protocol](https://img.shields.io/badge/MCP-v1.0-green.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Agent System](https://img.shields.io/badge/Agent%20System-Orchestrator%20Driven-purple.svg)](#agent-configuration-system)

A comprehensive implementation combining the **Orchestrator-Driven Multi-Agent Workflow** system with an advanced **Model Context Protocol (MCP) server** that features swarm intelligence capabilities for multi-agent coordination, collective knowledge management, and persistent memory systems.

## 🎯 Project Overview

This project demonstrates the integration of two powerful systems:

1. **Agent Configuration System**: A proven orchestrator-driven multi-agent workflow framework adapted from BitNet-Rust
2. **MCP Swarm Intelligence Server**: An advanced MCP server implementation with collective intelligence capabilities

The combination provides a foundation for building sophisticated AI-powered development workflows with automated multi-agent coordination and persistent learning capabilities.

## 🏗️ Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                 ORCHESTRATOR-DRIVEN ARCHITECTURE               │
├─────────────────────────────────────────────────────────────────┤
│  🎯 Central Orchestrator (agent-config/orchestrator.md)        │
│  ├── Task Routing & Agent Selection                            │
│  ├── Workflow Coordination                                     │
│  ├── Quality Gate Management                                   │
│  └── Progress Tracking                                         │
├─────────────────────────────────────────────────────────────────┤
│                    SPECIALIST AGENTS                           │
│  ├── 🐍 Python Specialist (MCP Implementation)                 │
│  ├── 🔧 MCP Specialist (Protocol Compliance)                   │
│  ├── 🐝 Swarm Intelligence Specialist (Coordination)           │
│  ├── 💾 Hive Mind Specialist (Collective Knowledge)            │
│  ├── 🧠 Memory Management Specialist (Persistent Memory)       │
│  ├── 💻 Code Development Specialist                            │
│  ├── 🧪 Test Utilities Specialist                              │
│  └── 📚 Documentation Writer Specialist                        │
├─────────────────────────────────────────────────────────────────┤
│                    MCP SERVER LAYER                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Tool Registry & Resources                      ││
│  │  • Agent Assignment  • Consensus Building                  ││
│  │  • Resource Coordination  • Memory Management              ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Git
- VS Code (recommended for MCP client integration)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Wavegoodvybe2929/Agent-Config-MCP.git
   cd Agent-Config-MCP
   ```

2. **Set up the MCP server**:

   ```bash
   cd mcp-swarm-server
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Initialize the database**:

   ```bash
   python scripts/init_database.py
   ```

4. **Start the MCP server**:

   ```bash
   python -m mcp_swarm
   ```

## 📁 Project Structure

```text
Agent-Config-MCP/
├── agent-config/                    # 🎯 Agent Configuration System
│   ├── orchestrator.md             # Central workflow coordinator (START HERE)
│   ├── specialists/                # Specialist agent configurations
│   │   ├── python_specialist.md    # Python/MCP development
│   │   ├── mcp_specialist.md       # MCP protocol expertise
│   │   ├── swarm_intelligence_specialist.md # Swarm algorithms
│   │   ├── hive_mind_specialist.md # Collective knowledge
│   │   ├── memory_management_specialist.md # Persistent memory
│   │   └── ...                     # Additional specialists
│   ├── agent-hooks.md              # Agent coordination system
│   └── project_*.md                # Project rules and commands
├── mcp-swarm-server/               # 🐝 MCP Server Implementation
│   ├── src/mcp_swarm/              # Core server code
│   │   ├── server/                 # MCP protocol implementation
│   │   ├── swarm/                  # Swarm intelligence algorithms
│   │   ├── memory/                 # Persistent memory system
│   │   └── agents/                 # Agent coordination
│   ├── docs/                       # Detailed documentation
│   ├── tests/                      # Comprehensive test suite
│   └── requirements.txt            # Python dependencies
├── .github/
│   └── copilot-instructions.md     # GitHub Copilot workflow integration
├── mcp_comprehensive_todo.md       # Project roadmap and tasks
└── README.md                       # This file
```

## 🎯 Getting Started: The Orchestrator-First Workflow

**🚨 IMPORTANT**: This project follows an **orchestrator-driven workflow**. All development work should start by consulting the orchestrator:

### Step 1: Consult the Orchestrator (MANDATORY)

```bash
# Always start here for any development work
cat agent-config/orchestrator.md
```

The orchestrator provides:

- **Task routing** to appropriate specialist agents
- **Workflow coordination** for complex multi-agent tasks
- **Quality gate requirements** and validation procedures
- **Current project context** and priority alignment

### Step 2: Follow Agent Routing

Based on the task type, the orchestrator will route you to appropriate specialists:

| Task Type | Primary Agent | Secondary Agents |
|-----------|---------------|------------------|
| **MCP Development** | `mcp_specialist.md` | `python_specialist.md`, `code.md` |
| **Swarm Intelligence** | `swarm_intelligence_specialist.md` | `hive_mind_specialist.md`, `memory_management_specialist.md` |
| **Documentation** | `documentation_writer.md` | Domain specialists |
| **Testing** | `test_utilities_specialist.md` | `debug.md`, `code.md` |
| **Architecture** | `architect.md` | `security_reviewer.md`, `performance_engineering_specialist.md` |

### Step 3: Execute with Coordination

All specialist work maintains orchestrator coordination throughout the development process.

## 🐝 MCP Swarm Intelligence Server

The MCP server implementation provides:

### Core Features

- **✅ MCP Protocol Compliance**: Full implementation of MCP v1.0 specification
- **🐝 Swarm Intelligence**: Ant Colony Optimization (ACO) and Particle Swarm Optimization (PSO) algorithms
- **💾 Collective Knowledge**: Hive mind knowledge bases with persistent memory
- **👑 Queen-Led Coordination**: Hierarchical coordination patterns for optimal agent assignment
- **🧠 Persistent Memory**: SQLite-backed cross-session state management
- **🤝 Real-time Consensus**: Collective decision-making for complex tasks

### Quick Usage

```python
# Connect to the MCP server
from mcp_swarm import SwarmServer

# Initialize with swarm intelligence
server = SwarmServer(
    swarm_config={
        "algorithm": "aco",  # or "pso"
        "agents": 5,
        "memory_persistence": True
    }
)

# Register tools for agent coordination
server.register_tool("assign_agent", assign_optimal_agent)
server.register_tool("consensus_build", build_consensus)

# Start the server
await server.run()
```

For detailed server documentation, see [`mcp-swarm-server/README.md`](./mcp-swarm-server/README.md).

## 📚 Documentation

### Agent Configuration System

- **[Orchestrator Guide](agent-config/orchestrator.md)** - Central workflow coordinator (START HERE)
- **[Agent Hooks System](agent-config/agent-hooks.md)** - Agent coordination and lifecycle management
- **[Specialist Configurations](agent-config/specialists/)** - Individual agent expertise areas
- **[Project Rules](agent-config/project_rules_config.md)** - Development standards and guidelines

### MCP Server Implementation

- **[MCP Server README](mcp-swarm-server/README.md)** - Detailed server documentation
- **[API Reference](mcp-swarm-server/docs/)** - Complete API documentation
- **[Swarm Intelligence Guide](mcp-swarm-server/docs/swarm-intelligence.md)** - Algorithm documentation
- **[Memory System Guide](mcp-swarm-server/docs/memory-system.md)** - Persistent memory documentation

### Setup and Integration

- **[Agent Config Setup Guide](mcp_agent_config_setup_guide.md)** - How to deploy the agent system
- **[MCP Server Guide](mcp_server_guide.md)** - Server setup and configuration
- **[Comprehensive TODO](mcp_comprehensive_todo.md)** - Project roadmap and development phases

## 🔧 Development Workflow

### Current Phase: Foundation Setup (Week 1)

**Priority Tasks** (as managed by orchestrator):

1. **Enhanced Project Structure** - Complete automated scaffolding with memory/swarm components
2. **Core MCP Server Foundation** - Python-based MCP server with swarm intelligence integration
3. **Agent Configuration Deployment** - Deploy MCP-specific agent configurations
4. **CI/CD Pipeline Setup** - Automated testing and quality assurance

### Development Process

1. **🎯 Always start with the orchestrator** - Read `agent-config/orchestrator.md` first
2. **Get proper routing** - Use the orchestrator's agent selection matrix
3. **Follow specialist guidance** - Consult routed specialist configurations
4. **Maintain coordination** - Keep orchestrator informed of progress
5. **Apply quality gates** - Follow orchestrator-defined validation requirements

## 🤝 Contributing

### Before Contributing

1. **Read the orchestrator configuration**: Start with `agent-config/orchestrator.md`
2. **Understand the agent system**: Review the specialist configurations
3. **Follow the workflow**: Use the orchestrator-driven development process
4. **Check current priorities**: Review `mcp_comprehensive_todo.md`

### Contribution Process

1. Fork the repository
2. Create a feature branch following the agent routing guidance
3. Make changes according to specialist agent guidelines
4. Ensure all orchestrator quality gates pass
5. Submit a pull request with orchestrator workflow documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **BitNet-Rust Project**: For the proven orchestrator-driven multi-agent workflow patterns
- **Model Context Protocol**: For the standardized protocol specification
- **Claude-Flow**: For inspiration on swarm intelligence and collective coordination patterns
- **VS Code Team**: For the excellent MCP client integration capabilities

## 🚀 What's Next?

### Phase 2: MCP Tools Implementation (Week 2)

- Advanced agent assignment automation with swarm algorithms
- Hive mind knowledge management tools
- Memory persistence and learning systems

### Phase 3: Advanced Swarm Features (Week 3)

- Queen-led coordination implementation
- Collective decision-making algorithms
- Real-time swarm adaptation

### Phase 4: Memory & Learning Systems (Week 4)

- Advanced pattern learning and optimization
- Cross-session knowledge persistence
- Intelligent task routing based on historical patterns

---

**🎯 Remember**: Always start with the orchestrator (`agent-config/orchestrator.md`) for any development work. The orchestrator provides the foundation for all successful multi-agent coordination in this project.
