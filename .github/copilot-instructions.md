# MCP Swarm Intelligence Server Copilot Instructions

## Project Overview

MCP Swarm Intelligence Server is a production-ready implementation of collective intelligence for multi-agent coordination, featuring agent ecosystem management, hive mind knowledge bases, persistent memory systems, and automated workflow orchestration using the Model Context Protocol.

## Agent Configuration System - Orchestrator-Driven Multi-Agent Workflow

This project uses the EXACT SAME agent configuration system as proven in BitNet-Rust, adapted for MCP development. **THE ORCHESTRATOR IS THE CENTRAL COMMAND** that routes all work and manages all specialist coordination with enhanced swarm intelligence and persistent memory capabilities.

### 🎯 MANDATORY ORCHESTRATOR-FIRST WORKFLOW

**ALWAYS START WITH THE ORCHESTRATOR** - This is non-negotiable for any development work:

#### **Step 1: ORCHESTRATOR CONSULTATION (REQUIRED)**
Before doing ANY work, **ALWAYS read `agent-config/orchestrator.md` FIRST** to:
- **Understand current project context** and MCP development priorities
- **Get proper task routing** to appropriate MCP specialist agents
- **Identify multi-agent coordination needs** for complex MCP features
- **Access workflow management** and quality gate requirements
- **Integrate with agent hooks system** for automated lifecycle management

#### **Step 2: ORCHESTRATOR ROUTING DECISION**
The orchestrator will route you to appropriate specialists using this framework:
- **Primary Agent Selection**: Based on task domain and complexity
- **Secondary Agent Coordination**: For cross-domain or complex requirements
- **Quality Gate Assignment**: Validation and review requirements
- **Workflow Coordination**: Timeline and dependency management

#### **Step 3: SPECIALIST CONSULTATION (ORCHESTRATOR-GUIDED)**
After orchestrator routing, consult the specific specialist agents identified:
- **Read specialist agent configs** for domain-specific context and expertise
- **Understand agent intersections** and collaboration patterns
- **Follow established workflows** and handoff procedures
- **Maintain orchestrator coordination** throughout the work

## MCP Server Integration

### Available MCP Tools

The MCP server provides these tools for swarm intelligence coordination:

#### **Agent Management Tools**
- **`agent_assignment(task_description, priority)`**: Optimal task assignment using Ant Colony Optimization (ACO) algorithms
- **`adaptive_coordination(coordination_mode, agents_involved)`**: Dynamic coordination with multiple strategies (hierarchical, democratic, expert, round_robin)
- **`agent_registration(agent_id, capabilities)`**: Register new agents in the swarm ecosystem
- **`agent_status_monitor()`**: Real-time monitoring of agent performance and health

#### **Swarm Intelligence Tools**
- **`swarm_consensus(decision_options, min_confidence)`**: Democratic decision-making using Particle Swarm Optimization (PSO)
- **`pheromone_trail_analysis(task_type)`**: Analyze success patterns for reinforcement learning
- **`swarm_optimization(optimization_target)`**: Apply swarm algorithms for parameter optimization
- **`coordination_pattern_learning()`**: Learn and adapt coordination patterns from outcomes

#### **Knowledge Management Tools**
- **`hive_mind_query(query, domain)`**: Semantic search across collective knowledge using FTS5 full-text search
- **`knowledge_synthesis(sources, confidence_threshold)`**: Synthesize information from multiple knowledge sources
- **`pattern_recognition(data_type, similarity_threshold)`**: Identify patterns in coordination data
- **`collective_learning_update(knowledge_domain, content)`**: Update hive mind with new learnings

#### **Memory Management Tools**
- **`memory_store(key, data, persistence_level)`**: Store persistent cross-session information in SQLite database
- **`memory_retrieve(key, similarity_search)`**: Retrieve stored memories with vector similarity matching
- **`memory_consolidation(time_window)`**: Consolidate and optimize memory storage
- **`session_state_management(action, session_data)`**: Manage cross-session state persistence

#### **Performance & Monitoring Tools**
- **`performance_analysis(metric_type, time_range)`**: Analyze swarm coordination performance metrics
- **`bottleneck_identification(system_component)`**: Identify performance bottlenecks using swarm intelligence
- **`optimization_recommendations(target_metric)`**: Generate optimization recommendations based on patterns
- **`health_monitoring(component)`**: Monitor system health and coordination effectiveness

### MCP Server Configuration

```json
{
  "tools": [
    {
      "name": "agent_assignment",
      "description": "Assign tasks to optimal agents using swarm intelligence algorithms"
    },
    {
      "name": "hive_mind_query", 
      "description": "Query collective knowledge base with semantic search"
    },
    {
      "name": "swarm_consensus",
      "description": "Reach consensus on decisions using swarm algorithms"
    },
    {
      "name": "adaptive_coordination",
      "description": "Dynamically coordinate multiple agents with adaptive strategies"
    },
    {
      "name": "memory_store",
      "description": "Store persistent cross-session information"
    },
    {
      "name": "memory_retrieve", 
      "description": "Retrieve stored memories with vector similarity"
    }
  ],
  "resources": [
    {
      "name": "agent_config_manager",
      "description": "Manage agent configuration files"
    },
    {
      "name": "copilot_instructions_manager",
      "description": "Manage copilot instructions with MCP server integration"
    },
    {
      "name": "hive_mind_query",
      "description": "Query collective knowledge"
    },
    {
      "name": "dynamic_coordination",
      "description": "Dynamic task coordination"
    }
    {
      "name": "agent_assignment",
      "description": "Assign tasks to optimal agents using swarm intelligence algorithms"
    },
    {
      "name": "hive_mind_query", 
      "description": "Query collective knowledge base with semantic search"
    },
    {
      "name": "swarm_consensus",
      "description": "Reach consensus on decisions using swarm algorithms"
    },
    {
      "name": "adaptive_coordination",
      "description": "Dynamically coordinate multiple agents with adaptive strategies"
    },
    {
      "name": "memory_store",
      "description": "Store persistent cross-session information"
    },
    {
      "name": "memory_retrieve", 
      "description": "Retrieve stored memories with vector similarity"
    }
  ],
  "resources": [
    {
      "name": "agent_configs",
      "description": "Agent configuration resources and templates"
    },
    {
      "name": "knowledge_base",
      "description": "Collective knowledge and learning resources"
    },
    {
      "name": "memory_snapshots",
      "description": "Persistent memory state snapshots"
    },
    {
      "name": "performance_metrics",
      "description": "Real-time coordination performance data"
    }
  ],
  "server_info": {
    "name": "MCP Swarm Intelligence Server",
    "version": "1.0.0"
  }
}
```

### Agent-Config Integration

The MCP server integrates seamlessly with the agent-config system:

- **Automatic Tool Discovery**: MCP tools are automatically discovered from 17 agent configurations
- **Orchestrator Coordination**: All MCP tool execution routes through orchestrator workflow management
- **Quality Gates**: Agent-defined quality standards apply to all MCP tool operations
- **Multi-Agent Workflows**: Complex MCP operations coordinate multiple specialist agents

## Agent Configuration Hierarchy & Orchestrator Authority

#### 🎯 **Central Command (ALWAYS START HERE)**
- **`orchestrator.md`** - **MANDATORY FIRST STOP** - Central coordination, agent routing, workflow management, project context

#### Core Technical Specialists (Orchestrator-Routed)
- **`agent-hooks.md`** - Enhanced Agent Hooks system with memory-backed lifecycle management and swarm coordination
- **`debug.md`** - Diagnostic and problem resolution specialist for MCP ecosystem debugging and root cause analysis
- **`security_reviewer.md`** - Security specialist for vulnerability identification and secure coding practices for MCP servers

#### Domain Specialists (Orchestrator-Coordinated)
- **`specialists/mcp_specialist.md`** - **MODEL CONTEXT PROTOCOL SPECIALIST** focusing on MCP protocol compliance, tool registration, and resource management
- **`specialists/python_specialist.md`** - **PYTHON DEVELOPMENT SPECIALIST** for Python-specific MCP implementation and best practices
- **`specialists/swarm_intelligence_specialist.md`** - **SWARM INTELLIGENCE SPECIALIST** implementing ACO/PSO algorithms and queen-led coordination
- **`specialists/memory_management_specialist.md`** - **MEMORY MANAGEMENT SPECIALIST** for persistent memory systems and SQLite database optimization
- **`specialists/hive_mind_specialist.md`** - **HIVE MIND SPECIALIST** for collective knowledge management and cross-session learning
- **`specialists/performance_engineering_specialist.md`** - **PERFORMANCE SPECIALIST** for optimization and bottleneck identification
- **`specialists/test_utilities_specialist.md`** - Test infrastructure specialist for comprehensive MCP testing frameworks
- **`specialists/documentation_writer.md`** - Documentation specialist for technical writing and user guides
- **`specialists/devops_infrastructure_specialist.md`** - **DEVOPS SPECIALIST** for CI/CD pipelines and deployment automation
- **`specialists/code.md`** - General project coordination and implementation specialist
- **`specialists/truth_validator.md`** - Truth validation specialist ensuring accurate project status reporting

## Multi-Agent Coordination Patterns (Orchestrator-Managed)

The orchestrator manages several coordination patterns for different task types:

#### **Single-Agent Tasks (Orchestrator Oversight)**
```
Simple MCP tasks → Primary specialist + orchestrator coordination
Tool implementation → mcp_specialist.md + python_specialist.md
Protocol compliance → mcp_specialist.md + security_reviewer.md review
Documentation → documentation_writer.md if user-facing
```

#### **Multi-Agent Collaboration (Orchestrator Coordination)**
```
Complex MCP features → Primary + Secondary specialists + orchestrator management
Swarm intelligence → swarm_intelligence_specialist.md + memory_management_specialist.md
Knowledge systems → hive_mind_specialist.md + memory_management_specialist.md
Performance optimization → performance_engineering_specialist.md + multiple domain specialists
Cross-domain tasks → Multiple specialists + orchestrator coordination
Critical changes → Full review chain + architect + security + orchestrator validation
```

#### **Emergency Response (Orchestrator Escalation)**
```
Critical MCP server issues → debug.md + orchestrator resource coordination
Security vulnerabilities → security_reviewer.md + immediate escalation
Performance problems → performance_engineering_specialist.md + swarm coordination
```

## MCP Server Architecture & Technology Stack

### Core Architecture
```
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

### Technology Stack
- **Python 3.11+**: Async/await architecture for concurrent operations
- **FastMCP**: Official MCP server framework for protocol implementation
- **SQLite 3.40+**: Embedded database with WAL mode, JSON1, and FTS5 extensions
- **NumPy/SciPy**: Numerical computations for swarm intelligence algorithms
- **AsyncIO**: Event loop management and coroutine coordination
- **Docker**: Containerized deployment with optimized configuration

### Database Schema
The persistent memory system uses SQLite with optimized configuration:
- **agents table**: Agent registration, capabilities, and performance tracking
- **tasks table**: Task assignment history and completion tracking  
- **hive_knowledge FTS5**: Full-text searchable collective knowledge base
- **consensus_decisions**: Democratic decision-making outcomes and confidence scores
- **performance_metrics**: Real-time coordination performance data

## Development Workflow Rules - Orchestrator-Driven

1. **🎯 ALWAYS START WITH ORCHESTRATOR** - Read `orchestrator.md` first for every task
2. **Follow orchestrator routing** - Use the orchestrator's agent selection matrix
3. **Maintain orchestrator coordination** - Keep orchestrator informed of progress and handoffs
4. **Respect agent intersections** - Follow established collaboration patterns between agents
5. **Use quality gates** - Apply orchestrator-defined validation requirements
6. **Execute user requests exactly** - Within the orchestrator's workflow framework
7. **Leverage MCP tools** - Use available swarm intelligence tools for coordination
8. **Maintain persistent memory** - Store learnings and patterns in the hive mind database
9. **Follow swarm patterns** - Apply ACO/PSO algorithms for optimal coordination
10. **Stop when complete** - When orchestrator-defined success criteria are met

## Docker Deployment & Usage

### Quick Start
```bash
# Build the Docker image
cd Docker
docker build -t mcp-swarm-server .

# Run the server
docker run -d \
  --name mcp-swarm \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  mcp-swarm-server
```

### Environment Configuration
```bash
# Database configuration
SWARM_DB_PATH=/app/data/memory.db          # SQLite database path
SWARM_DB_ENCRYPTION_KEY=your_key_here      # Database encryption (optional)

# Security configuration  
SWARM_ADMIN_TOKEN=your_admin_token         # Admin access token (optional)

# Logging configuration
PYTHONUNBUFFERED=1                         # Unbuffered Python output
```

### MCP Client Integration

#### VS Code Integration
```json
{
  "mcpServers": {
    "mcp-swarm-server": {
      "command": "docker",
      "args": ["exec", "-i", "mcp-swarm", "python", "mcp_swarm_intelligence_server.py"],
      "env": {}
    }
  }
}
```

#### Claude Desktop Integration
```json
{
  "mcpServers": {
    "mcp-swarm-intelligence": {
      "command": "docker",
      "args": ["exec", "-i", "mcp-swarm", "python", "mcp_swarm_intelligence_server.py"]
    }
  }
}
```

## Current Development Phase

**🎯 Development Phase**: Production-ready MCP server with swarm intelligence capabilities
- **Status**: **COMPLETE & PRODUCTION READY**
- **Orchestrator Routing**: As defined in orchestrator.md workflow matrix
- **Goal**: Fully operational containerized MCP server with collective intelligence
- **Key Features**: Agent coordination, hive mind knowledge, persistent memory, swarm algorithms
- **Deployment**: Docker-ready with optimized SQLite configuration

## Performance & Scalability

### Database Optimization
- **WAL Mode**: Concurrent read/write operations
- **40MB Cache**: Improved query performance  
- **FTS5 Search**: Full-text search capabilities
- **Vector Embeddings**: Semantic similarity matching

### Swarm Coordination
- **ACO/PSO Algorithms**: Optimized task assignment and consensus building
- **Async Operations**: Concurrent agent coordination
- **Performance Monitoring**: Real-time metrics and optimization
- **Persistent Learning**: Cross-session pattern recognition

### Security Features
- **Container Security**: Non-root execution, minimal attack surface
- **Data Protection**: Optional database encryption, secure defaults
- **Network Security**: Container isolation, configurable access control

## When to Stop - Orchestrator-Defined Criteria

- Task completed successfully according to orchestrator quality gates
- User request fulfilled within orchestrator workflow context
- MCP tools properly utilized for coordination and knowledge management
- Persistent memory updated with learnings and patterns
- No further action required as determined by orchestrator coordination
- Clear completion criteria from orchestrator workflow met
- Production deployment standards maintained

## Agent Intersection Examples

### Complex MCP Development Tasks
- **MCP Protocol Implementation**: mcp_specialist.md + python_specialist.md + test_utilities_specialist.md
- **Swarm Intelligence Features**: swarm_intelligence_specialist.md + memory_management_specialist.md + performance_engineering_specialist.md  
- **Knowledge Management**: hive_mind_specialist.md + memory_management_specialist.md + mcp_specialist.md
- **Performance Optimization**: performance_engineering_specialist.md + debug.md + swarm_intelligence_specialist.md
- **Security Review**: security_reviewer.md + mcp_specialist.md + python_specialist.md

### Quality Assurance Workflows
- **Code Review**: code.md + security_reviewer.md + test_utilities_specialist.md
- **Documentation**: documentation_writer.md + mcp_specialist.md + truth_validator.md
- **Deployment**: devops_infrastructure_specialist.md + security_reviewer.md + performance_engineering_specialist.md
