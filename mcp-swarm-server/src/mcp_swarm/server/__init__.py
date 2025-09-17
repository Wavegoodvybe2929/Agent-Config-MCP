"""
MCP Server Implementation

This module provides the core MCP server implementation with swarm intelligence
integration for multi-agent coordination and collective decision-making.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional, Dict, Any
from pathlib import Path

from .base import SwarmMCPServer, ToolInfo, ResourceInfo
from .tools import ToolRegistry, ToolMetadata
from .resources import ResourceManager
from .messages import MessageHandler

__all__ = ["SwarmMCPServer", "create_server", "run_server", "ToolInfo", "ResourceInfo"]


async def create_server(
    name: str = "swarm-intelligence-server",
    version: str = "1.0.0",
    resource_path: Optional[str] = None,
    cache_size: int = 100
) -> SwarmMCPServer:
    """Create and initialize swarm MCP server.
    
    Args:
        name: Server name
        version: Server version
        resource_path: Path for resource storage
        cache_size: Resource cache size
        
    Returns:
        Initialized SwarmMCPServer instance
    """
    # Create server instance
    server = SwarmMCPServer(name=name, version=version)
    
    # Initialize tool registry
    tool_registry = ToolRegistry()
    
    # Initialize resource manager
    if resource_path is None:
        resource_path = "data/resources"
    resource_manager = ResourceManager(base_path=resource_path, cache_size=cache_size)
    
    # Initialize message handler
    message_handler = MessageHandler(server)
    
    # Set up server components
    server._tool_registry = tool_registry
    server._resource_manager = resource_manager
    server._message_handler = message_handler
    
    # Initialize server
    await server.initialize()
    
    # Discover and register tools
    discovered_tools = await tool_registry.discover_tools()
    for tool_metadata in discovered_tools:
        await tool_registry.register_tool(tool_metadata)
        
    # Initialize resource manager
    await resource_manager.initialize()
    
    logging.info("Created and initialized %s v%s", name, version)
    return server


async def run_server(server: Optional[SwarmMCPServer] = None, **kwargs) -> None:
    """Run the MCP server with proper lifecycle management.
    
    Args:
        server: Optional pre-created server instance
        **kwargs: Arguments for create_server if server not provided
    """
    if server is None:
        server = await create_server(**kwargs)
        
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("mcp.swarm.runner")
    
    # Set up signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        shutdown_event.set()
        
    # Register signal handlers
    if sys.platform != 'win32':
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
    
    try:
        logger.info("Starting %s v%s", server.name, server.version)
        
        # Create main server loop
        server_task = asyncio.create_task(server_loop(server, shutdown_event))
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
        logger.info("Shutting down server...")
        
        # Cancel server task
        server_task.cancel()
        
        try:
            await server_task
        except asyncio.CancelledError:
            pass
            
        # Shutdown server gracefully
        await server.shutdown()
        
        logger.info("Server shutdown complete")
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        await server.shutdown()
    except Exception as e:
        logger.error("Server error: %s", str(e))
        await server.shutdown()
        raise


async def server_loop(server: SwarmMCPServer, shutdown_event: asyncio.Event) -> None:
    """Main server loop for handling requests.
    
    Args:
        server: Server instance
        shutdown_event: Event to signal shutdown
    """
    logger = logging.getLogger("mcp.swarm.loop")
    
    # This is a simplified loop - in a real implementation, this would
    # handle stdio/websocket communication with MCP clients
    
    try:
        while not shutdown_event.is_set():
            # In a real implementation, this would:
            # 1. Read messages from stdin or websocket
            # 2. Parse JSON-RPC messages
            # 3. Route to message handler
            # 4. Send responses back to client
            
            # For now, just run periodic maintenance
            await asyncio.sleep(1.0)
            
            # Cleanup expired cache periodically
            if (hasattr(server, '_resource_manager') and 
                server._resource_manager is not None and
                hasattr(server._resource_manager, 'cleanup_expired_cache')):
                await server._resource_manager.cleanup_expired_cache()
                
    except asyncio.CancelledError:
        logger.info("Server loop cancelled")
        raise
    except Exception as e:
        logger.error("Server loop error: %s", str(e))
        raise


async def create_stdio_server(**kwargs) -> SwarmMCPServer:
    """Create server configured for stdio transport.
    
    Args:
        **kwargs: Arguments for create_server
        
    Returns:
        Configured server instance
    """
    server = await create_server(**kwargs)
    
    # Configure for stdio transport
    # This would set up stdin/stdout handling in a real implementation
    
    return server


async def create_websocket_server(host: str = "localhost", port: int = 8765, **kwargs) -> SwarmMCPServer:
    """Create server configured for WebSocket transport.
    
    Args:
        host: WebSocket host
        port: WebSocket port
        **kwargs: Arguments for create_server
        
    Returns:
        Configured server instance
    """
    server = await create_server(**kwargs)
    
    # Configure for WebSocket transport
    # This would set up WebSocket handling in a real implementation
    logger = logging.getLogger("mcp.swarm.websocket")
    logger.info("WebSocket server would listen on %s:%s", host, port)
    
    return server


def register_example_tools(server: SwarmMCPServer) -> None:
    """Register example tools for demonstration.
    
    Args:
        server: Server instance
    """
    async def example_tool(message: str = "Hello, World!") -> str:
        """Example tool that echoes a message."""
        return f"Echo: {message}"
        
    async def math_add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
        
    async def math_multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
        
    # Register tools
    if (server._tool_registry is not None and
        hasattr(server._tool_registry, 'register_tool_function')):
        asyncio.create_task(server._tool_registry.register_tool_function(
            "echo",
            example_tool,
            "Echo a message back",
            {
                "message": {
                    "type": "string",
                    "description": "Message to echo",
                    "required": False
                }
            }
        ))
    
    if (server._tool_registry is not None and
        hasattr(server._tool_registry, 'register_tool_function')):
        asyncio.create_task(server._tool_registry.register_tool_function(
            "add",
            math_add,
            "Add two numbers",
            {
                "a": {
                    "type": "number",
                    "description": "First number",
                    "required": True
                },
                "b": {
                    "type": "number", 
                    "description": "Second number",
                    "required": True
                }
            }
        ))
    
    if (server._tool_registry is not None and
        hasattr(server._tool_registry, 'register_tool_function')):
        asyncio.create_task(server._tool_registry.register_tool_function(
            "multiply",
            math_multiply,
            "Multiply two numbers",
            {
                "a": {
                    "type": "number",
                "description": "First number", 
                "required": True
            },
            "b": {
                "type": "number",
                "description": "Second number",
                "required": True
            }
        }
    ))


async def register_example_resources(server: SwarmMCPServer) -> None:
    """Register example resources for demonstration.
    
    Args:
        server: Server instance
    """
    # Create example text resource
    if (server._resource_manager is not None and
        hasattr(server._resource_manager, 'create_resource')):
        await server._resource_manager.create_resource(
            uri="example://hello.txt",
            content="Hello, this is an example text resource!",
            name="Hello Text",
            description="A simple text resource for demonstration",
            tags=["example", "text"]
        )
    
    # Create example JSON resource
    import json
    json_content = json.dumps({
        "message": "Hello from JSON",
        "type": "example",
        "data": [1, 2, 3, 4, 5]
    }, indent=2)
    
    if (server._resource_manager is not None and
        hasattr(server._resource_manager, 'create_resource')):
        await server._resource_manager.create_resource(
            uri="example://data.json", 
            content=json_content,
            name="Example Data",
            description="Example JSON data resource",
            tags=["example", "json", "data"]
        )


# Convenience function for quick server startup
async def main() -> None:
    """Main entry point for server execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Swarm Intelligence Server")
    parser.add_argument("--name", default="swarm-intelligence-server", help="Server name")
    parser.add_argument("--version", default="1.0.0", help="Server version")
    parser.add_argument("--transport", choices=["stdio", "websocket"], default="stdio", help="Transport type")
    parser.add_argument("--host", default="localhost", help="WebSocket host (for websocket transport)")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port (for websocket transport)")
    parser.add_argument("--resource-path", help="Resource storage path")
    parser.add_argument("--cache-size", type=int, default=100, help="Resource cache size")
    parser.add_argument("--examples", action="store_true", help="Register example tools and resources")
    
    args = parser.parse_args()
    
    # Create server based on transport type
    if args.transport == "stdio":
        server = await create_stdio_server(
            name=args.name,
            version=args.version,
            resource_path=args.resource_path,
            cache_size=args.cache_size
        )
    elif args.transport == "websocket":
        server = await create_websocket_server(
            host=args.host,
            port=args.port,
            name=args.name,
            version=args.version,
            resource_path=args.resource_path,
            cache_size=args.cache_size
        )
    else:
        raise ValueError(f"Unknown transport type: {args.transport}")
        
    # Register examples if requested
    if args.examples:
        register_example_tools(server)
        await register_example_resources(server)
        
    # Run server
    await run_server(server)


if __name__ == "__main__":
    asyncio.run(main())