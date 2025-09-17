# MCP Swarm Intelligence Server Copilot Instructions

## Project Overview

MCP Swarm Intelligence Server is a high-performance implementation of collective intelligence for multi-agent coordination, featuring agent ecosystem management, hive mind knowledge bases, persistent memory systems, and automated workflow orchestration. The project follows the **Orchestrator-Driven Multi-Agent Workflow** enhanced with claude-flow inspired swarm intelligence patterns.

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

### Agent Configuration Hierarchy & Orchestrator Authority

#### 🎯 **Central Command (ALWAYS START HERE)**
- **`orchestrator.md`** - **MANDATORY FIRST STOP** - Central coordination, agent routing, workflow management, project context

#### Core Technical Specialists (Orchestrator-Routed)
- **`architect.md`** - System architecture and design (intersects with: research, security, code)
- **`code.md`** - Primary development and implementation (intersects with: python_specialist, debug, test_utilities)
- **`debug.md`** - Problem resolution and troubleshooting (intersects with: code, error_handling, test_utilities)
- **`python_specialist.md`** - Python development and MCP implementation (intersects with: mcp_specialist, code, api_development)
- **`mcp_specialist.md`** - Model Context Protocol expertise (intersects with: python_specialist, api_development, code)
- **`swarm_intelligence_specialist.md`** - Swarm algorithms and coordination (intersects with: hive_mind_specialist, performance_engineering, code)
- **`hive_mind_specialist.md`** - Collective knowledge management (intersects with: memory_management, swarm_intelligence, code)
- **`memory_management_specialist.md`** - Persistent memory systems (intersects with: hive_mind_specialist, performance_engineering, code)
- **`performance_engineering_specialist.md`** - Optimization and acceleration (intersects with: swarm_intelligence, memory_management, architect)
- **`test_utilities_specialist.md`** - Testing infrastructure and validation (intersects with: error_handling, debug, truth_validator)
- **`error_handling_specialist.md`** - Error management and resilience (intersects with: debug, python_specialist, test_utilities)

#### Quality & Coordination Specialists
- **`truth_validator.md`** - Quality assurance and validation (intersects with: ALL agents for quality gates)
- **`security_reviewer.md`** - Security and safety analysis (intersects with: rust_best_practices, architect, error_handling)
- **`documentation_writer.md`** - Technical documentation (intersects with: ask, api_development, ALL specialists)
- **`ask.md`** - User interaction and requirements (intersects with: documentation_writer, customer_success, ui_ux)

#### Project Management (Orchestrator-Coordinated)
- **`development_phase_tracker.md`** - Timeline and milestone tracking (intersects with: comprehensive_todo_manager, orchestrator, truth_validator)
- **`comprehensive_todo_manager.md`** - Roadmap management (intersects with: development_phase_tracker, ALL specialists)
- **`publishing_expert.md`** - Release management (intersects with: truth_validator, documentation_writer, devops_infrastructure)

#### Commercial & Business Specialists
- **`saas_platform_architect.md`** - SaaS platform design (intersects with: api_development, devops_infrastructure, security_reviewer)
- **`api_development_specialist.md`** - API development (intersects with: saas_platform, inference_engine, documentation_writer)
- **`business_intelligence_specialist.md`** - Business analytics (intersects with: customer_success, performance_engineering, api_development)
- **`customer_success_specialist.md`** - Customer success (intersects with: business_intelligence, ask, ui_ux)
- **`devops_infrastructure_specialist.md`** - DevOps and infrastructure (intersects with: saas_platform, publishing_expert, security_reviewer)
- **`ui_ux_development_specialist.md`** - Frontend and UX (intersects with: customer_success, api_development, ask)

#### Support & Configuration
- **`agent-hooks.md`** - Agent coordination system (intersects with: ALL agents, orchestrator, truth_validator)
- **`project_commands_config.md`** - Build systems and commands (intersects with: devops_infrastructure, code, test_utilities)
- **`project_rules_config.md`** - Standards and guidelines (intersects with: rust_best_practices, variable_matcher, ALL specialists)
- **`variable_matcher.md`** - Code consistency (intersects with: rust_best_practices, code, project_rules_config)
- **`project_research.md`** - Research and innovation (intersects with: architect, performance_engineering, security_reviewer)

### Multi-Agent Coordination Patterns (Orchestrator-Managed)

The orchestrator manages several coordination patterns for different task types:

#### **Single-Agent Tasks (Orchestrator Oversight)**
```
Simple tasks → Primary specialist + orchestrator coordination
Quality validation → truth_validator.md review
Documentation → documentation_writer.md if user-facing
```

#### **Multi-Agent Collaboration (Orchestrator Coordination)**
```
Complex features → Primary + Secondary specialists + orchestrator management
Cross-domain tasks → Multiple specialists + daily sync + orchestrator coordination
Critical changes → Full review chain + architect + security + orchestrator validation
```

#### **Emergency Response (Orchestrator Escalation)**
```
Critical bugs → debug.md + immediate escalation + orchestrator resource coordination
Performance issues → debug.md + performance_engineering + orchestrator timeline management
Security incidents → security_reviewer + architect + orchestrator incident management
```

### Workflow Decision Framework (Orchestrator-Defined)

**The orchestrator uses this decision matrix for task routing:**

| Task Type | Primary Agent | Secondary Agents | Quality Gates |
|-----------|---------------|------------------|---------------|
| Feature Development | `code.md` | `python_specialist.md`, `test_utilities_specialist.md` | Code quality + test coverage |
| Debugging | `debug.md` | `code.md`, `error_handling_specialist.md` | Root cause + fix validation |
| Architecture | `architect.md` | `project_research.md`, `security_reviewer.md` | Design review + security validation |
| Performance | `performance_engineering_specialist.md` | `swarm_intelligence_specialist.md`, `code.md` | Benchmark validation + optimization review |
| MCP Server Development | `mcp_specialist.md` | `python_specialist.md`, `api_development_specialist.md` | Protocol compliance + API usability |
| Swarm Intelligence | `swarm_intelligence_specialist.md` | `hive_mind_specialist.md`, `performance_engineering_specialist.md` | Algorithm efficiency + coordination quality |
| Memory Management | `memory_management_specialist.md` | `hive_mind_specialist.md`, `performance_engineering_specialist.md` | Memory efficiency + persistence validation |
| Documentation | `documentation_writer.md` | `ask.md`, domain specialists | User testing + accuracy validation |
| Testing | `test_utilities_specialist.md` | `error_handling_specialist.md`, `debug.md` | Coverage + edge case validation |
| Security | `security_reviewer.md` | `python_specialist.md`, `architect.md` | Vulnerability assessment + safe patterns |
| Release | `publishing_expert.md` | `truth_validator.md`, `documentation_writer.md` | Comprehensive validation + documentation |

### Agent Intersection Understanding (Orchestrator-Defined)

**Every agent understands their intersections with other agents:**

- **Code Development** intersects with Python Specialist (quality), Debug (fixes), Test Utilities (validation)
- **Debug** intersects with Code (implementation), Error Handling (resilience), Test Utilities (reproduction)
- **MCP Specialist** intersects with Python Specialist (implementation), API Development (interfaces), Code (implementation)
- **Swarm Intelligence** intersects with Hive Mind (coordination), Performance Engineering (optimization), Code (implementation)
- **Performance Engineering** intersects with Swarm Intelligence (algorithm speed), Memory Management (efficiency), Architect (system design)
- **Security Reviewer** intersects with Python Specialist (safety), Architect (security design), Error Handling (resilience)

**And many more intersections explicitly defined in each agent config.**

### Current Priority (Week 1) - Orchestrator-Managed

**🎯 Phase 1**: Enhanced Foundation Setup for MCP Swarm Intelligence Server
- **Orchestrator Routing**: `orchestrator.md` → `code.md` + `python_specialist.md` + `memory_management_specialist.md`
- **Goal**: Complete automated project scaffolding with memory/swarm components
- **Key Tasks**: Project structure, agent config deployment, CI/CD pipeline
- **Effort**: 30 hours across Phase 1
- **Next**: Phase 2 - MCP Tools Implementation (Week 2)

## Workflow Rules - Orchestrator-Driven

1. **🎯 ALWAYS START WITH ORCHESTRATOR** - Read `orchestrator.md` first for every task
2. **Follow orchestrator routing** - Use the orchestrator's agent selection matrix
3. **Maintain orchestrator coordination** - Keep orchestrator informed of progress and handoffs
4. **Respect agent intersections** - Follow established collaboration patterns between agents
5. **Use quality gates** - Apply orchestrator-defined validation requirements
6. **Follow current phase** - Align with COMPREHENSIVE_TODO.md priorities as managed by orchestrator
7. **Execute user requests exactly** - Within the orchestrator's workflow framework
8. **Stop when complete** - When orchestrator-defined success criteria are met
9. **Be direct and clear** - Provide straightforward responses following orchestrator guidance
10. **Use available tools** - Leverage tools efficiently within orchestrator's workflow framework

## Orchestrator-Coordinated Workflow

1. **🎯 START WITH ORCHESTRATOR** - Always read `orchestrator.md` first
2. **Get orchestrator routing** - Use agent selection matrix for appropriate specialists
3. **Follow agent intersections** - Consult routed specialists with understanding of their collaboration patterns
4. **Execute with coordination** - Perform work following orchestrator workflow management
5. **Report through orchestrator** - Confirm completion and coordinate next steps through orchestrator

## When to Stop - Orchestrator-Defined Criteria

- Task completed successfully according to orchestrator quality gates
- User request fulfilled within orchestrator workflow context
- No further action required as determined by orchestrator coordination
- Clear completion criteria from orchestrator workflow met
- Current phase priorities defined by orchestrator respected

## Project Context Usage - Orchestrator-Managed

- **Current Status**: Foundation setup phase, building MCP Swarm Intelligence Server with enhanced memory and coordination capabilities
- **Active Roadmap**: MCP_COMPREHENSIVE_TODO.md managed through orchestrator and specialist coordination
- **Agent Coordination**: ALL coordination managed through orchestrator workflow with swarm intelligence enhancement
- **Quality Gates**: Orchestrator-defined excellence standards while implementing MCP server development phases
- **Workflow Management**: ALL development activities coordinated through orchestrator's enhanced multi-agent management system with persistent memory


