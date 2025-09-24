"""
MCP Swarm Intelligence Server - Automation Module

This module provides complete automation capabilities for the MCP Swarm Intelligence Server,
including end-to-end workflow orchestration, multi-agent coordination, parallel execution,
and automated error recovery systems.

Key Components:
- CompleteWorkflowOrchestrator: End-to-end workflow orchestration
- MultiAgentCoordinator: Multi-agent coordination with zero manual handoffs
- ParallelExecutionEngine: Optimal parallel task execution
- AutomatedErrorRecovery: Automated error recovery with alternative paths
"""

from .workflow_orchestrator import CompleteWorkflowOrchestrator
from .multi_agent_coordinator import MultiAgentCoordinator
from .parallel_engine import ParallelExecutionEngine
from .error_recovery import AutomatedErrorRecovery

__all__ = [
    "CompleteWorkflowOrchestrator",
    "MultiAgentCoordinator", 
    "ParallelExecutionEngine",
    "AutomatedErrorRecovery"
]