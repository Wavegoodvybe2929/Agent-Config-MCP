"""
MCP Swarm Intelligence Server CLI

Command-line interface for the MCP Swarm Intelligence Server.
Provides commands to start, configure, and manage the server.
"""

import asyncio
import logging
import sys
import signal
from pathlib import Path
from typing import Optional, Dict, Any
import click
import uvicorn
from dotenv import load_dotenv

from .server import create_server, SwarmMCPServer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global server instance for signal handling
_server_instance: Optional[SwarmMCPServer] = None


def signal_handler(sig, _frame):
    """Handle shutdown signals gracefully."""
    logger.info("Received signal %s, shutting down...", sig)
    if _server_instance:
        # Trigger graceful shutdown
        asyncio.create_task(_server_instance.shutdown())
    sys.exit(0)


@click.group()
@click.version_option(version="0.1.0", prog_name="mcp-swarm")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool):
    """MCP Swarm Intelligence Server - Collective intelligence for multi-agent coordination."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")


@cli.command()
@click.option("--name", default="swarm-intelligence-server", help="Server name")
@click.option("--version", default="1.0.0", help="Server version")  
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", type=int, default=8080, help="Port to bind to")
@click.option("--resource-path", help="Path for resource storage")
@click.option("--cache-size", type=int, default=100, help="Resource cache size")
@click.option("--stdio", is_flag=True, help="Use stdio transport (for MCP clients)")
def start(
    name: str,
    version: str,
    host: str,
    port: int,
    resource_path: Optional[str],
    cache_size: int,
    stdio: bool
):
    """Start the MCP Swarm Intelligence Server."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting MCP Swarm Server '%s' v%s", name, version)
    
    if stdio:
        # Run with stdio transport (for MCP clients like Claude Desktop)
        asyncio.run(start_stdio_server(name, version, resource_path, cache_size))
    else:
        # Run with HTTP transport (for web-based access)
        asyncio.run(start_http_server(name, version, host, port, resource_path, cache_size))


async def start_stdio_server(
    name: str, 
    version: str, 
    resource_path: Optional[str], 
    cache_size: int
):
    """Start server with stdio transport for MCP clients."""
    global _server_instance
    
    try:
        # Create server instance
        _server_instance = await create_server(
            name=name,
            version=version,
            resource_path=resource_path,
            cache_size=cache_size
        )
        
        logger.info("MCP Server started with stdio transport")
        logger.info("Server ready to accept MCP protocol connections")
        
        # Run the server (this will handle MCP protocol over stdio)
        await _server_instance.run_stdio()
        
    except Exception as e:
        logger.error("Failed to start stdio server: %s", str(e))
        sys.exit(1)


async def start_http_server(
    name: str,
    version: str, 
    host: str,
    port: int,
    resource_path: Optional[str],
    cache_size: int
):
    """Start server with HTTP transport for web access."""
    global _server_instance
    
    try:
        # Create server instance
        _server_instance = await create_server(
            name=name,
            version=version,
            resource_path=resource_path,
            cache_size=cache_size
        )
        
        logger.info("MCP Server starting on http://%s:%s", host, port)
        
        # Create FastAPI application wrapper
        from fastapi import FastAPI
        app = FastAPI(
            title=name,
            version=version,
            description="MCP Swarm Intelligence Server - Collective intelligence for multi-agent coordination"
        )
        
        # Add health check endpoint
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "server": name, "version": version}
        
        # Add MCP endpoint (basic implementation)
        @app.post("/mcp")
        async def mcp_endpoint(request: Dict[str, Any]):
            """Handle MCP protocol requests over HTTP."""
            try:
                # This would need proper MCP protocol handling
                return {"jsonrpc": "2.0", "result": {"status": "not implemented"}, "id": request.get("id")}
            except Exception as e:
                return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}, "id": request.get("id")}
        
        # Run with uvicorn
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except Exception as e:
        logger.error("Failed to start HTTP server: %s", str(e))
        sys.exit(1)


@cli.command()
@click.option("--config-path", default=".agent-config", help="Path to agent configuration directory")
def validate(config_path: str):
    """Validate MCP server configuration and agent setup."""
    logger.info("Validating MCP server configuration...")
    
    config_dir = Path(config_path)
    if not config_dir.exists():
        logger.error(f"Agent configuration directory not found: {config_path}")
        sys.exit(1)
    
    # Check for required agent configurations
    required_agents = [
        "orchestrator.md",
        "specialists/mcp_specialist.md", 
        "specialists/python_specialist.md",
        "specialists/swarm_intelligence_specialist.md"
    ]
    
    missing_agents = []
    for agent in required_agents:
        agent_path = config_dir / agent
        if not agent_path.exists():
            missing_agents.append(agent)
        else:
            logger.info(f"✅ Found agent config: {agent}")
    
    if missing_agents:
        logger.error("❌ Missing required agent configurations:")
        for agent in missing_agents:
            logger.error(f"   - {agent}")
        sys.exit(1)
    
    logger.info("✅ All required agent configurations found")
    logger.info("MCP server validation completed successfully")


@cli.command()
def info():
    """Display server information and capabilities."""
    logger.info("MCP Swarm Intelligence Server Information")
    logger.info("=" * 50)
    logger.info(f"Version: 0.1.0")
    logger.info(f"Python: {sys.version}")
    logger.info(f"MCP Protocol: Supported")
    logger.info(f"Transport: stdio, HTTP")
    logger.info(f"Capabilities:")
    logger.info(f"  - Swarm Intelligence Coordination")
    logger.info(f"  - Multi-Agent Task Assignment") 
    logger.info(f"  - Persistent Memory Management")
    logger.info(f"  - Collective Decision Making")
    logger.info(f"  - Knowledge Base Management")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()