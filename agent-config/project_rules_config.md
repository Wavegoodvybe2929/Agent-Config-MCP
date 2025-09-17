# MCP Swarm Intelligence Server Project Rules & Guidelines

## Project Overview

MCP Swarm Intelligence Server is a Model Context Protocol server implementation featuring collective intelligence for multi-agent coordination, hive mind knowledge bases, persistent memory systems, and automated workflow orchestration.

## Development Phases - Current Status (September 17, 2025)

- **Phase 1**: Enhanced Foundation Setup (**CURRENT IMPLEMENTATION** 🎯)
- **Current Achievement**: Starting fresh with proven orchestrator patterns adapted for MCP
- **Phase 2-5**: MCP Tools & Advanced Features (**PLANNED** - Dependencies being established)

**Current Status**: 🎯 **PHASE 1 FOUNDATION SETUP** - Enhanced Project Structure with Swarm Intelligence & Persistent Memory

### Latest Development Status (September 17, 2025) - FOUNDATION SETUP PHASE 🎯

**FOUNDATION ESTABLISHMENT**: Project implementing proven BitNet-Rust orchestrator patterns for MCP development

#### 🎯 FOUNDATION SETUP STATUS:
- **Agent Configuration System**: ✅ MCP-specific orchestrator and specialist agents deployed
- **Project Structure**: 🎯 Python-based MCP server with SQLite memory database (IN PROGRESS)
- **Core Technologies**: 🎯 Python 3.11+, MCP SDK, asyncio, SQLite, swarm algorithms (IMPLEMENTATION)
- **CI/CD Pipeline**: 🎯 GitHub Actions with automated testing and quality gates (PLANNED)
- **Enhancement Features**: 🎯 Claude-flow inspired swarm intelligence and persistent memory (IMPLEMENTATION)

#### 🎯 CURRENT FOCUS AREAS (Foundation Setup):
- **Project Scaffolding**: Python package structure with virtual environment and dependencies
- **MCP Server Foundation**: Core MCP server implementation with protocol compliance
- **Swarm Intelligence Core**: ACO/PSO algorithms for agent coordination and task assignment
- **Memory Management**: SQLite database with persistent cross-session memory
- **Testing Infrastructure**: Comprehensive test coverage with pytest and automated validation

## Core Development Standards

### Python Development Standards
- **Python Version**: 3.11+ (leverage latest async improvements and performance)
- **Code Style**: Black formatting, flake8 linting, isort import organization
- **Type Safety**: Comprehensive type hints with mypy validation
- **Documentation**: Detailed docstrings for all public APIs and functions

### MCP Protocol Standards
- **Protocol Compliance**: Full adherence to MCP v1.x specification
- **Tool Registration**: Dynamic tool discovery with type-safe parameter validation
- **Resource Management**: Comprehensive MIME type support with proper schemas
- **Error Handling**: MCP standard error codes with detailed diagnostic information

### Swarm Intelligence Standards
- **Algorithm Performance**: Real-time coordination decisions (< 1 second response)
- **Optimization Quality**: 95%+ optimal solutions for task assignment
- **Scalability**: Support 100+ agents with linear performance degradation
- **Pattern Learning**: Continuous improvement from coordination outcomes

### Database and Memory Standards
- **SQLite Configuration**: WAL mode with optimized PRAGMA settings for concurrency
- **Data Integrity**: ACID compliance with proper transaction management
- **Performance**: Sub-10ms query response for indexed operations
- **Backup**: Automated backup procedures with integrity validation

## Code Quality Requirements

### Testing Standards
- **Test Coverage**: Minimum 95% code coverage across all modules
- **Test Types**: Unit tests, integration tests, performance tests, and property-based tests
- **Async Testing**: Comprehensive async/await testing with pytest-asyncio
- **MCP Testing**: Protocol compliance testing with mock clients and servers

### Performance Standards
- **MCP Response Time**: Sub-100ms for tool operations and resource access
- **Swarm Coordination**: Sub-1 second for coordination decisions and consensus
- **Database Operations**: Sub-10ms for indexed queries and memory operations
- **Memory Usage**: Optimal memory usage with proper cleanup and garbage collection

### Security Standards
- **Input Validation**: Comprehensive validation for all external inputs and parameters
- **Resource Access**: Secure resource access with proper permission controls
- **Error Handling**: No sensitive information leaked in error messages
- **Dependency Security**: Regular security scanning with safety and bandit

## Architecture Principles

### Orchestrator-Driven Development
- **Central Coordination**: All development activities coordinated through orchestrator.md
- **Agent Specialization**: Clear separation of concerns with specialist agents
- **Workflow Management**: Standardized workflows with agent hooks integration
- **Quality Gates**: Automated validation at every development stage

### Swarm Intelligence Design
- **Queen-Led Coordination**: Hierarchical coordination with orchestrator as master coordinator
- **Collective Decision Making**: Democratic voting and consensus algorithms
- **Pattern Recognition**: Learning from successful coordination patterns
- **Adaptive Optimization**: Continuous improvement through experience

### Memory and Persistence
- **Cross-Session Memory**: Persistent learning and state across server restarts
- **Knowledge Management**: Semantic search and pattern matching for collective intelligence
- **Data Lifecycle**: Automated cleanup and archival of obsolete data
- **Performance Optimization**: Efficient data structures and access patterns

### Async and Concurrency
- **Async-First Design**: All I/O operations use async/await patterns
- **Concurrent Safety**: Thread-safe operations with proper synchronization
- **Resource Management**: Connection pooling and prepared statement caching
- **Error Propagation**: Proper async exception handling and recovery

## Development Workflow

### Task Assignment Process
1. **Orchestrator Consultation**: Always start with orchestrator.md for task routing
2. **Specialist Selection**: Use orchestrator's agent selection matrix for appropriate agents
3. **Requirements Analysis**: Collaborate with relevant specialists for detailed requirements
4. **Implementation Planning**: Design implementation strategy with architect.md if needed
5. **Code Development**: Implement following Python and MCP standards
6. **Testing and Validation**: Comprehensive testing with test_utilities_specialist.md
7. **Quality Review**: Code review with relevant specialist agents
8. **Integration**: System integration coordinated through orchestrator.md

### Quality Assurance Process
1. **Pre-commit Validation**: Automated code quality checks with pre-commit hooks
2. **Type Checking**: mypy validation for type safety and correctness
3. **Testing**: Automated test execution with coverage reporting
4. **Performance Validation**: Benchmark testing for critical operations
5. **Security Review**: Security analysis for external interfaces and dependencies
6. **Documentation**: API documentation and usage examples

### Agent Coordination Rules
- **Mandatory Orchestrator Routing**: All agents must consult orchestrator.md first
- **Intersection Awareness**: Agents must understand and follow intersection patterns
- **Quality Gate Compliance**: All quality gates must pass before task completion
- **Handoff Procedures**: Proper handoff coordination between specialist agents
- **Status Reporting**: Regular progress updates through orchestrator coordination

## File and Directory Standards

### Project Structure
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
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
└── pyproject.toml          # Project metadata and configuration
```

### Code Organization
- **Package Structure**: Clear module organization with proper __init__.py files
- **Import Management**: Absolute imports preferred, organized with isort
- **Naming Conventions**: PEP 8 compliant naming for classes, functions, and variables
- **Documentation**: Comprehensive docstrings with parameter and return type information

### Configuration Management
- **Environment Variables**: Use environment variables for configuration
- **Settings Files**: Python configuration files for complex settings
- **Schema Validation**: Validate configuration with pydantic or similar
- **Default Values**: Sensible defaults for all configuration options

## Error Handling and Logging

### Error Handling Standards
- **MCP Error Codes**: Use standard MCP error codes for protocol errors
- **Exception Hierarchy**: Custom exception hierarchy for different error types
- **Error Context**: Include relevant context information in error messages
- **Recovery Strategies**: Implement graceful degradation and recovery mechanisms

### Logging Standards
- **Structured Logging**: Use structured logging with JSON format for production
- **Log Levels**: Appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Sensitive Data**: Never log sensitive information or credentials
- **Performance**: Efficient logging that doesn't impact performance

## Dependencies and Environment

### Python Dependencies
- **Core**: mcp (official SDK), asyncio, sqlite3, numpy, scikit-learn
- **Testing**: pytest, pytest-asyncio, hypothesis, coverage
- **Quality**: black, flake8, mypy, isort, pre-commit
- **Optional**: sentence-transformers, networkx, aiofiles

### Development Environment
- **Python Version**: 3.11+ with virtual environment isolation
- **Package Manager**: pip with requirements.txt and requirements-dev.txt
- **Version Control**: Git with conventional commit messages
- **IDE Support**: VSCode configuration with Python extensions

### CI/CD Pipeline
- **Automated Testing**: GitHub Actions with multi-version Python testing
- **Code Quality**: Automated code quality checks and enforcement
- **Security Scanning**: Dependency vulnerability scanning with safety
- **Documentation**: Automated documentation generation and deployment

## Success Metrics

### Development Quality Metrics
- **Test Coverage**: 95%+ coverage across all modules
- **Code Quality**: Zero flake8/mypy violations in production code
- **Performance**: All performance benchmarks within target ranges
- **Security**: Zero high-risk security vulnerabilities

### MCP Compliance Metrics
- **Protocol Compliance**: 100% adherence to MCP specification
- **Tool Functionality**: All MCP tools operational with proper validation
- **Resource Management**: Comprehensive MIME type support and handling
- **Client Compatibility**: Compatible with standard MCP clients

### Swarm Intelligence Metrics
- **Coordination Quality**: 95%+ successful task coordination and assignment
- **Algorithm Performance**: Real-time coordination within 1-second targets
- **Learning Effectiveness**: Continuous improvement from coordination outcomes
- **Scalability**: Linear performance degradation with agent population growth

### System Performance Metrics
- **Response Time**: Sub-100ms for MCP operations, sub-1s for coordination
- **Memory Usage**: Optimal memory usage with proper cleanup and optimization
- **Database Performance**: Sub-10ms for indexed queries and operations
- **Availability**: 99.9% uptime with proper error handling and recovery