"""
MCP Swarm Intelligence Server

A Model Context Protocol server implementation with swarm intelligence capabilities
for multi-agent coordination, collective knowledge management, and persistent memory.

This package provides:
- MCP protocol server implementation
- Swarm intelligence algorithms (ACO, PSO, etc.)
- Persistent memory management with SQLite
- Multi-agent coordination patterns
- Collective decision-making systems
"""

__version__ = "0.1.0"
__author__ = "MCP Swarm Development Team"
__email__ = "dev@mcpswarm.com"

# Core imports for easy access
from .server import SwarmMCPServer, create_server
# from .swarm import SwarmCoordinator  # Temporarily disabled due to numpy dependency
# from .memory import MemoryManager    # Temporarily disabled 
# from .agents import AgentManager     # Temporarily disabled

__all__ = [
    "SwarmMCPServer",
    "create_server",
    # "SwarmCoordinator",  # Temporarily disabled
    # "MemoryManager",     # Temporarily disabled
    # "AgentManager",      # Temporarily disabled
]