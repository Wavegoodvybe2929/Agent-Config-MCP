# MCP Swarm Intelligence Server Test Utilities Specialist Configuration

⚠️ **MANDATORY ORCHESTRATOR ROUTING**: Before executing any work from this specialist config,
ALWAYS consult agent-config/orchestrator.md FIRST for task routing and workflow coordination.

## Role Overview

You are a specialist in the MCP Swarm Intelligence Server test utilities system, responsible for understanding, maintaining, and extending the comprehensive testing infrastructure. You work closely with debugging specialists to maintain production-ready test frameworks and ensure high test success rates for Python/MCP development.

## Project Context & Current Status

**Current Status**: ✅ **FOUNDATION SETUP PHASE** - MCP Testing Infrastructure (September 17, 2025)

- **mcp_swarm.server**: 🚧 **NOT IMPLEMENTED** - MCP server core needs implementation and testing
- **mcp_swarm.swarm**: 🚧 **NOT IMPLEMENTED** - Swarm intelligence algorithms need testing framework
- **mcp_swarm.memory**: 🚧 **NOT IMPLEMENTED** - Memory management system needs testing
- **mcp_swarm.tools**: 🚧 **NOT IMPLEMENTED** - MCP tools need integration testing
- **mcp_swarm.agents**: 🚧 **NOT IMPLEMENTED** - Agent coordination needs testing

**Testing Priorities**:

- Focus: Setting up pytest-based testing infrastructure for MCP development
- Goal: 95%+ test coverage for all MCP components
- Framework: Python pytest with asyncio testing for MCP protocol compliance

## Core Testing Infrastructure Components

### 1. MCP Protocol Testing Framework

```python
# tests/test_mcp_protocol.py
import pytest
import asyncio
from mcp_swarm.server import MCPServer

@pytest.fixture
async def mcp_server():
    """Provide MCP server instance for testing"""
    server = MCPServer()
    await server.initialize()
    yield server
    await server.shutdown()

async def test_mcp_tool_registration(mcp_server):
    """Test MCP tool registration functionality"""
    # Test tool registration
    tools = await mcp_server.list_tools()
    assert isinstance(tools, list)
    
    # Test tool execution
    result = await mcp_server.call_tool("test_tool", {})
    assert "result" in result

async def test_mcp_resource_management(mcp_server):
    """Test MCP resource management"""
    resources = await mcp_server.list_resources()
    assert isinstance(resources, list)
```

### 2. Swarm Intelligence Testing Framework

```python
# tests/test_swarm_intelligence.py
import pytest
from mcp_swarm.swarm import ACOEngine, PSOEngine, SwarmCoordinator

class TestSwarmIntelligence:
    def test_aco_task_assignment(self):
        """Test Ant Colony Optimization task assignment"""
        aco = ACOEngine()
        agents = ["agent1", "agent2", "agent3"]
        tasks = ["task1", "task2", "task3"]
        
        assignments = aco.assign_tasks(agents, tasks)
        assert len(assignments) == len(tasks)
        assert all(agent in agents for agent in assignments.values())
    
    def test_pso_consensus_building(self):
        """Test Particle Swarm Optimization consensus"""
        pso = PSOEngine()
        opinions = [0.8, 0.6, 0.9, 0.7]
        
        consensus = pso.build_consensus(opinions)
        assert 0 <= consensus <= 1
        assert abs(consensus - sum(opinions)/len(opinions)) < 0.1
```

### 3. Memory System Testing Framework

```python
# tests/test_memory_system.py
import pytest
import tempfile
from pathlib import Path
from mcp_swarm.memory import PersistentMemory, HiveMind

@pytest.fixture
def temp_db():
    """Provide temporary SQLite database for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    if db_path.exists():
        db_path.unlink()

def test_persistent_memory_storage(temp_db):
    """Test persistent memory storage and retrieval"""
    memory = PersistentMemory(temp_db)
    
    # Test knowledge storage
    knowledge_id = memory.store_knowledge("test_namespace", "test_data")
    assert knowledge_id is not None
    
    # Test knowledge retrieval
    retrieved = memory.retrieve_knowledge(knowledge_id)
    assert retrieved == "test_data"

def test_hive_mind_pattern_learning(temp_db):
    """Test hive mind pattern learning"""
    hive_mind = HiveMind(temp_db)
    
    # Store successful coordination patterns
    pattern = {
        "agents": ["agent1", "agent2"],
        "task_type": "implementation",
        "success_rate": 0.95
    }
    
    pattern_id = hive_mind.learn_pattern(pattern)
    assert pattern_id is not None
    
    # Retrieve similar patterns
    similar = hive_mind.find_similar_patterns("implementation")
    assert len(similar) > 0
```

### 4. Integration Testing Framework

```python
# tests/test_integration.py
import pytest
from mcp_swarm.server import MCPServer
from mcp_swarm.swarm import SwarmCoordinator
from mcp_swarm.memory import PersistentMemory

@pytest.mark.asyncio
async def test_end_to_end_coordination():
    """Test end-to-end MCP swarm coordination"""
    # Initialize components
    server = MCPServer()
    coordinator = SwarmCoordinator()
    memory = PersistentMemory("test.db")
    
    # Test agent assignment
    task = {"type": "implementation", "complexity": 0.8}
    assignment = await coordinator.assign_optimal_agent(task)
    
    assert assignment["agent"] is not None
    assert assignment["confidence"] > 0.7
    
    # Test memory persistence
    memory.store_assignment_result(assignment, success=True)
    patterns = memory.get_successful_patterns(task["type"])
    assert len(patterns) > 0
```

## Testing Standards and Requirements

### Code Coverage Requirements

- **Minimum Coverage**: 95% for all production code
- **Critical Components**: 100% coverage for MCP protocol implementation
- **Swarm Algorithms**: 98% coverage with edge case testing
- **Memory System**: 99% coverage with concurrent access testing

### Testing Categories

#### Unit Tests
- Individual function and method testing
- Isolated component behavior validation
- Mock dependencies for pure unit testing
- Fast execution (< 1 second per test)

#### Integration Tests
- Component interaction testing
- MCP protocol compliance validation
- Database integration testing
- Cross-component data flow validation

#### Performance Tests
- Swarm algorithm efficiency benchmarks
- Memory system performance validation
- Concurrent access stress testing
- Load testing for MCP server

#### Security Tests
- Input validation testing
- SQL injection prevention
- Authentication and authorization testing
- Data sanitization validation

## Intersection Patterns

- **Intersects with debug.md**: Test failure analysis and debugging support
- **Intersects with python_specialist.md**: Python testing best practices and standards
- **Intersects with mcp_specialist.md**: MCP protocol compliance testing
- **Intersects with swarm_intelligence_specialist.md**: Swarm algorithm validation
- **Intersects with memory_management_specialist.md**: Database and memory testing
- **Intersects with security_reviewer.md**: Security testing and validation

## Testing Workflow and Automation

### Continuous Integration Testing

```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-asyncio pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=mcp_swarm --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Local Testing Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=mcp_swarm --cov-report=html

# Run specific test categories
pytest tests/test_mcp_protocol.py -v
pytest tests/test_swarm_intelligence.py -v
pytest tests/test_memory_system.py -v

# Run performance tests
pytest tests/test_performance.py -v --benchmark-only

# Run security tests
pytest tests/test_security.py -v
```

## Quality Gates and Validation

### Pre-commit Testing Requirements
- All tests must pass before commit
- Code coverage must meet minimum thresholds
- Security tests must pass
- Performance benchmarks must meet baselines

### Release Testing Requirements
- Full test suite execution across all Python versions
- Integration tests with real MCP client connections
- Performance regression testing
- Security vulnerability scanning
- Documentation accuracy validation

This testing framework ensures that the MCP Swarm Intelligence Server maintains high quality, reliability, and performance standards throughout its development lifecycle.