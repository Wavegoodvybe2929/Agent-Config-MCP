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

### Available MCP Tools - Complete Catalog (29 Tools)

The MCP Swarm Intelligence Server provides 29 comprehensive tools for collective intelligence, multi-agent coordination, and automated workflow management. **All tools route through the orchestrator** for proper agent selection and quality gates.

#### **🎯 Core Swarm Intelligence Tools** (Orchestrator-Routed to Swarm Specialist)

**`agent_assignment(task_description="", priority="0.5")`**
- **Purpose**: Optimal task assignment using Ant Colony Optimization (ACO) algorithms
- **Agent Routing**: `swarm_intelligence_specialist.md` (primary) + `orchestrator.md` (coordination)
- **Use Cases**: Multi-agent task distribution, workload optimization, capability matching
- **Example**: Assign complex MCP development tasks to most suitable agent combination

**`swarm_consensus(decision_options="", min_confidence="0.7")`**
- **Purpose**: Democratic decision-making using Particle Swarm Optimization (PSO)
- **Agent Routing**: `swarm_intelligence_specialist.md` (primary) + `orchestrator.md` (validation)
- **Use Cases**: Complex technical decisions, architecture choices, quality gate decisions
- **Example**: Choose between multiple implementation approaches with confidence scoring

**`adaptive_coordination(coordination_mode="auto", agents_involved="")`**
- **Purpose**: Dynamic multi-agent coordination with adaptive strategies (hierarchical, democratic, expert, round_robin)
- **Agent Routing**: `orchestrator.md` (primary) - **ALWAYS routes through central command**
- **Use Cases**: Complex multi-agent workflows, escalation management, workflow optimization
- **Example**: Automatically select optimal coordination pattern based on task complexity

**`coordination_strategies(strategy_type="optimal", context_factors="")`**
- **Purpose**: Select optimal coordination strategies based on context analysis
- **Agent Routing**: `swarm_intelligence_specialist.md` (primary) + `orchestrator.md` (approval)
- **Use Cases**: Strategy optimization, workflow design, coordination pattern selection
- **Example**: Choose hierarchical vs democratic coordination based on project phase

#### **🧠 Knowledge & Hive Mind Tools** (Orchestrator-Routed to Hive Mind Specialist)

**`hive_mind_query(query="", domain="")`**
- **Purpose**: Semantic search across collective knowledge using FTS5 full-text search
- **Agent Routing**: `hive_mind_specialist.md` (primary) + `memory_management_specialist.md` (secondary)
- **Use Cases**: Knowledge discovery, pattern recognition, historical context retrieval
- **Example**: Query past successful coordination patterns for similar tasks

**`knowledge_contribution(domain="", content="", confidence="0.8")`**
- **Purpose**: Contribute knowledge to the hive mind collective database
- **Agent Routing**: `hive_mind_specialist.md` (primary) + validation through quality gates
- **Use Cases**: Learning storage, pattern recording, knowledge sharing across sessions
- **Example**: Store successful MCP implementation patterns for future reuse

**`semantic_search(query="", limit="10")`**
- **Purpose**: Advanced semantic search capabilities across all knowledge sources
- **Agent Routing**: `hive_mind_specialist.md` (primary) + `mcp_specialist.md` (protocol compliance)
- **Use Cases**: Multi-source information retrieval, pattern matching, contextual searches
- **Example**: Find similar past issues and their solutions across all project knowledge

**`knowledge_synthesis_tool(sources="", confidence_threshold="0.7")`**
- **Purpose**: Synthesize knowledge from multiple sources with confidence weighting
- **Agent Routing**: `hive_mind_specialist.md` (primary) + `truth_validator.md` (validation)
- **Use Cases**: Multi-source analysis, comprehensive reports, decision support
- **Example**: Synthesize information from multiple agents for complex technical decisions

#### **💾 Memory & Persistence Tools** (Orchestrator-Routed to Memory Specialist)

**`memory_management(operation="store", key="", data="", namespace="general")`**
- **Purpose**: Core persistent memory operations with namespace organization
- **Agent Routing**: `memory_management_specialist.md` (primary) + `orchestrator.md` (lifecycle)
- **Use Cases**: Cross-session data persistence, agent state management, workflow continuity
- **Example**: Store project state between development sessions

**`persistent_memory_manager(operation="status", memory_type="all", retention_policy="intelligent")`**
- **Purpose**: Advanced persistent memory management with intelligent retention policies
- **Agent Routing**: `memory_management_specialist.md` (primary) + `performance_engineering_specialist.md` (optimization)
- **Use Cases**: Memory optimization, storage cleanup, performance tuning
- **Example**: Optimize memory usage and implement intelligent data retention

**`pattern_analysis_tool(data_type="coordination", similarity_threshold="0.8")`**
- **Purpose**: Advanced pattern recognition and analysis across memory systems
- **Agent Routing**: `memory_management_specialist.md` (primary) + `swarm_intelligence_specialist.md` (pattern learning)
- **Use Cases**: Trend analysis, behavior prediction, optimization opportunities
- **Example**: Identify recurring coordination patterns for optimization

#### **🔧 Configuration & Agent Management Tools** (Orchestrator-Routed to MCP Specialist)

**`agent_config_manager(action="list", agent_id="", config_data="")`**
- **Purpose**: Manage agent configuration files and capabilities registration
- **Agent Routing**: `mcp_specialist.md` (primary) + `orchestrator.md` (coordination) + `agent-hooks.md` (lifecycle)
- **Use Cases**: Agent registration, capability updates, configuration management
- **Example**: Register new specialized agents or update existing capabilities

**`copilot_instructions_manager(action="create", instruction_type="full", output_path=".github/copilot-instructions.md")`**
- **Purpose**: Manage copilot instructions with comprehensive MCP server integration
- **Agent Routing**: `mcp_specialist.md` (primary) + `documentation_writer.md` (content) + `truth_validator.md` (accuracy)
- **Use Cases**: Documentation generation, instruction updates, MCP integration guides
- **Example**: Generate updated copilot instructions with all available MCP tools

**`ecosystem_management(action="health_check", target="")`**
- **Purpose**: Monitor and manage the complete agent ecosystem health
- **Agent Routing**: `orchestrator.md` (primary) - **Central command authority**
- **Use Cases**: System health monitoring, ecosystem optimization, agent lifecycle management
- **Example**: Check overall system health and identify optimization opportunities

**`agent_hooks(hook_type="lifecycle", event="", payload="")`**
- **Purpose**: Execute agent lifecycle hooks for automated workflow management
- **Agent Routing**: `agent-hooks.md` (primary) + `orchestrator.md` (coordination)
- **Use Cases**: Agent lifecycle management, event handling, automated workflow triggers
- **Example**: Trigger agent activation/deactivation hooks during workflow changes

#### **📊 Performance & Monitoring Tools** (Orchestrator-Routed to Performance Specialist)

**`performance_metrics(metric_type="overview", agent_id="")`**
- **Purpose**: Comprehensive performance metrics and system monitoring
- **Agent Routing**: `performance_engineering_specialist.md` (primary) + `orchestrator.md` (reporting)
- **Use Cases**: Performance analysis, bottleneck identification, optimization tracking
- **Example**: Monitor swarm coordination efficiency and identify performance bottlenecks

**`self_monitoring_tool(monitoring_scope="full", auto_remediation="true")`**
- **Purpose**: Comprehensive self-monitoring and automated optimization
- **Agent Routing**: `performance_engineering_specialist.md` (primary) + `debug.md` (troubleshooting)
- **Use Cases**: System health monitoring, automated optimization, predictive maintenance
- **Example**: Continuously monitor and automatically optimize system performance

**`predictive_maintenance_tool(prediction_type="system_health", time_horizon="7_days")`**
- **Purpose**: Predictive system maintenance and optimization recommendations
- **Agent Routing**: `performance_engineering_specialist.md` (primary) + `swarm_intelligence_specialist.md` (prediction)
- **Use Cases**: Proactive maintenance, failure prevention, resource planning
- **Example**: Predict system bottlenecks and recommend preventive actions

**`resource_optimization_tool(optimization_target="efficiency", resource_type="all")`**
- **Purpose**: Comprehensive resource allocation and optimization
- **Agent Routing**: `performance_engineering_specialist.md` (primary) + `orchestrator.md` (approval)
- **Use Cases**: Resource allocation, cost optimization, performance tuning
- **Example**: Optimize agent resource allocation for maximum efficiency

#### **🔍 Quality Assurance & Validation Tools** (Orchestrator-Routed to QA Specialists)

**`quality_assurance_tool(validation_scope="comprehensive", auto_fix="true")`**
- **Purpose**: Automated quality assurance and validation across all systems
- **Agent Routing**: `test_utilities_specialist.md` (primary) + `security_reviewer.md` + `truth_validator.md`
- **Use Cases**: Automated testing, quality gates, compliance validation
- **Example**: Validate MCP server implementation against quality standards

**`compliance_validator_tool(compliance_type="mcp_protocol", validation_level="strict")`**
- **Purpose**: Validate system compliance against MCP protocol and other standards
- **Agent Routing**: `mcp_specialist.md` (primary) + `security_reviewer.md` (security) + `test_utilities_specialist.md` (validation)
- **Use Cases**: Protocol compliance, security validation, standards adherence
- **Example**: Ensure MCP server implementation meets protocol specifications

**`anomaly_detection_tool(detection_scope="system_wide", sensitivity="medium")`**
- **Purpose**: Comprehensive anomaly detection across all system components
- **Agent Routing**: `debug.md` (primary) + `performance_engineering_specialist.md` (analysis)
- **Use Cases**: Issue detection, performance anomalies, security threats
- **Example**: Detect unusual coordination patterns or performance degradation

**`knowledge_quality_validator(knowledge_id="", validation_criteria="comprehensive", auto_improve="false")`**
- **Purpose**: Validate and improve knowledge quality with automated enhancement
- **Agent Routing**: `truth_validator.md` (primary) + `hive_mind_specialist.md` (knowledge)
- **Use Cases**: Knowledge validation, accuracy verification, content improvement
- **Example**: Validate technical documentation accuracy and completeness

#### **🚀 Advanced Intelligence & Learning Tools** (Orchestrator-Routed to Learning Systems)

**`adaptive_learning_tool(operation="get_status", parameters="", optimization_target="task_success")`**
- **Purpose**: Comprehensive adaptive learning and system evolution
- **Agent Routing**: `swarm_intelligence_specialist.md` (primary) + `memory_management_specialist.md` (persistence)
- **Use Cases**: System learning, performance evolution, adaptive optimization
- **Example**: Learn from coordination outcomes to improve future task assignments

**`workflow_automation_tool(automation_type="full_pipeline", workflow_config="", trigger_conditions="")`**
- **Purpose**: Automated workflow management and orchestration
- **Agent Routing**: `orchestrator.md` (primary) - **Central workflow control**
- **Use Cases**: Process automation, workflow optimization, task sequencing
- **Example**: Automate complete MCP development workflows with quality gates

#### **🔧 Utility & Support Tools** (Available to All Agents)

**`directory_manager(action="list", path="", options="")`**
- **Purpose**: Manage project directories and file system operations
- **Agent Routing**: Available to all agents through `orchestrator.md` coordination
- **Use Cases**: File management, project structure, organizational tasks
- **Example**: Organize project files and maintain directory structure

**`confidence_aggregation(confidence_values="", method="weighted_average")`**
- **Purpose**: Aggregate confidence scores across multiple agents and decisions
- **Agent Routing**: `orchestrator.md` (coordination) + any agent requiring confidence analysis
- **Use Cases**: Multi-agent decision making, confidence scoring, validation
- **Example**: Combine confidence scores from multiple agents for final decision

**`consensus_algorithms(algorithm="majority_vote", options="", agent_preferences="")`**
- **Purpose**: Apply various consensus mechanisms for group decision making
- **Agent Routing**: `swarm_intelligence_specialist.md` (primary) + `orchestrator.md` (coordination)
- **Use Cases**: Democratic decisions, conflict resolution, group consensus
- **Example**: Resolve conflicts between agents using democratic voting algorithms

**`decision_audit(decision_id="", audit_depth="comprehensive")`**
- **Purpose**: Comprehensive auditing and analysis of decision-making processes
- **Agent Routing**: `orchestrator.md` (primary) + `truth_validator.md` (validation)
- **Use Cases**: Decision tracking, process improvement, accountability
- **Example**: Audit complex technical decisions for lessons learned

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
    },
    {
      "name": "agent_config_manager",
      "description": "Manage agent configuration files"
    },
    {
      "name": "copilot_instructions_manager",
      "description": "Manage copilot instructions with MCP server integration"
    },
    {
      "name": "knowledge_contribution",
      "description": "Contribute knowledge to hive mind database"
    },
    {
      "name": "ecosystem_management",
      "description": "Monitor and manage agent ecosystem health"
    },
    {
      "name": "semantic_search",
      "description": "Advanced semantic search across knowledge base"
    },
    {
      "name": "performance_metrics",
      "description": "Comprehensive performance metrics and monitoring"
    },
    {
      "name": "self_monitoring_tool",
      "description": "System self-monitoring and optimization"
    },
    {
      "name": "predictive_maintenance_tool",
      "description": "Predictive system maintenance and optimization"
    },
    {
      "name": "quality_assurance_tool",
      "description": "Automated quality assurance and validation"
    },
    {
      "name": "compliance_validator_tool",
      "description": "Validate system compliance against standards"
    },
    {
      "name": "anomaly_detection_tool",
      "description": "Comprehensive system anomaly detection"
    },
    {
      "name": "knowledge_quality_validator",
      "description": "Validate knowledge quality and accuracy"
    },
    {
      "name": "adaptive_learning_tool",
      "description": "Adaptive learning and system evolution"
    },
    {
      "name": "workflow_automation_tool",
      "description": "Automated workflow management and orchestration"
    },
    {
      "name": "directory_manager",
      "description": "Project directory and file management"
    },
    {
      "name": "confidence_aggregation",
      "description": "Aggregate confidence scores across agents"
    },
    {
      "name": "consensus_algorithms",
      "description": "Apply various consensus mechanisms"
    },
    {
      "name": "coordination_strategies",
      "description": "Select optimal coordination strategies"
    },
    {
      "name": "decision_audit",
      "description": "Audit and analyze decision-making processes"
    },
    {
      "name": "agent_hooks",
      "description": "Execute agent lifecycle hooks"
    },
    {
      "name": "knowledge_synthesis_tool",
      "description": "Synthesize knowledge from multiple sources"
    },
    {
      "name": "pattern_analysis_tool",
      "description": "Advanced pattern recognition and analysis"
    },
    {
      "name": "resource_optimization_tool",
      "description": "Resource allocation and optimization"
    },
    {
      "name": "persistent_memory_manager",
      "description": "Advanced persistent memory management"
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

### 🎯 MCP Tool Usage Guidelines & Best Practices

#### **When to Use Each Tool Category**

**🐝 Swarm Intelligence Tools** - Use when you need:
- Optimal task assignment for multi-agent workflows
- Democratic consensus on complex technical decisions
- Dynamic coordination strategy selection
- Performance optimization through swarm algorithms

**🧠 Knowledge & Hive Mind Tools** - Use when you need:
- Access to historical patterns and successful solutions
- Cross-session knowledge persistence and learning
- Multi-source information synthesis and validation
- Semantic search across all project knowledge

**💾 Memory & Persistence Tools** - Use when you need:
- Cross-session state management and continuity
- Pattern recognition from historical data
- Memory optimization and intelligent retention
- Long-term learning and adaptation

**⚙️ Configuration & Management Tools** - Use when you need:
- Agent ecosystem health monitoring
- Configuration file management and updates
- Lifecycle event handling and automation
- Documentation generation and maintenance

**📊 Performance & Monitoring Tools** - Use when you need:
- Real-time system performance analysis
- Predictive maintenance and issue prevention
- Resource optimization and bottleneck identification
- Automated system health monitoring

**🔍 Quality & Validation Tools** - Use when you need:
- Automated quality assurance and compliance checking
- Knowledge accuracy validation and improvement
- System anomaly detection and analysis
- Standards compliance verification

#### **Orchestrator Routing Patterns**

**Simple Single-Tool Operations:**
```
User Request → Orchestrator Routing → Primary Agent + Tool → Result
```

**Complex Multi-Tool Workflows:**
```
User Request → Orchestrator Analysis → Primary Agent Selection → 
Tool Coordination → Secondary Agent Consultation → 
Quality Gates → Result Synthesis → Final Output
```

**Emergency Response Patterns:**
```
Critical Issue → Orchestrator Escalation → debug.md + 
Emergency Tools → Immediate Resolution → 
Post-incident Analysis → Learning Storage
```

#### **Tool Combination Patterns**

**For Complex Development Tasks:**
1. Start with `adaptive_coordination` to select optimal strategy
2. Use `agent_assignment` for task distribution
3. Apply `hive_mind_query` for historical context
4. Execute with specialist tools based on domain
5. Validate with `quality_assurance_tool`
6. Store learnings with `knowledge_contribution`

**For Performance Optimization:**
1. Begin with `performance_metrics` for baseline analysis
2. Use `anomaly_detection_tool` to identify issues
3. Apply `resource_optimization_tool` for improvements
4. Validate with `self_monitoring_tool`
5. Schedule with `predictive_maintenance_tool`

**For Knowledge Management:**
1. Query existing knowledge with `hive_mind_query`
2. Synthesize information with `knowledge_synthesis_tool`
3. Validate quality with `knowledge_quality_validator`
4. Contribute new insights with `knowledge_contribution`
5. Analyze patterns with `pattern_analysis_tool`
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
