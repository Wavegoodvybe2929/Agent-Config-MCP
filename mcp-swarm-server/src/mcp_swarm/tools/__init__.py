"""
MCP Tools Implementation

This module provides MCP tool implementations for swarm intelligence operations
including agent assignment, consensus building, resource coordination, and 
agent configuration management.
"""

from .agent_assignment import AgentAssignmentTool
from .mcda import MCDAAnalyzer, Alternative, Criterion
from .load_balancer import LoadBalancer, AgentLoad, LoadStatus
from .fuzzy_matcher import FuzzyCapabilityMatcher, CapabilityMatch, FuzzySet
from .explanation import AssignmentExplainer, AssignmentReason, AssignmentExplanation
from .agent_config_manager import (
    AgentConfigManagerTool,
    agent_config_manager_tool,
    agent_config_directory_setup_tool,
    MCP_TOOLS as AGENT_CONFIG_TOOLS,
    TOOL_REGISTRY as AGENT_CONFIG_TOOL_REGISTRY
)
from .copilot_instructions_manager import (
    CopilotInstructionsManager,
    handle_copilot_instructions_manager_tool,
    copilot_instructions_manager_schema,
)

__all__ = [
    "AgentAssignmentTool",
    "MCDAAnalyzer", 
    "Alternative", 
    "Criterion",
    "LoadBalancer", 
    "AgentLoad", 
    "LoadStatus",
    "FuzzyCapabilityMatcher", 
    "CapabilityMatch", 
    "FuzzySet",
    "AssignmentExplainer", 
    "AssignmentReason", 
    "AssignmentExplanation",
    "AgentConfigManagerTool",
    "agent_config_manager_tool",
    "agent_config_directory_setup_tool",
    "AGENT_CONFIG_TOOLS",
    "AGENT_CONFIG_TOOL_REGISTRY",
    "CopilotInstructionsManager",
    "handle_copilot_instructions_manager_tool",
    "copilot_instructions_manager_schema",
]