"""
Factory functions for creating server instances.
"""

from .base import SwarmMCPServer


async def create_server(name: str = "swarm-intelligence-server") -> SwarmMCPServer:
    """Create and initialize swarm MCP server.
    
    Args:
        name: Server name identifier
        
    Returns:
        Initialized SwarmMCPServer instance
    """
    server = SwarmMCPServer(name)
    await server.initialize()
    return server