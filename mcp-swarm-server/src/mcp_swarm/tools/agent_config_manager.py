"""
Agent Configuration Manager MCP Tool

This module provides MCP tool interfaces for managing agent configuration files,
including creating, reading, updating, and validating agent configurations with
proper orchestrator routing patterns and YAML frontmatter support.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# MCP availability check
try:
    import mcp  # noqa: F401
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from ..agents.config_operations import AgentConfigOperations

logger = logging.getLogger(__name__)


def mcp_tool(name: str):
    """Decorator for MCP tools (placeholder implementation)."""
    def decorator(func):
        func.mcp_tool_name = name
        return func
    return decorator


class AgentConfigManagerTool:
    """
    MCP tool implementation for agent configuration management.
    
    Provides comprehensive agent configuration management capabilities through MCP interface.
    """
    
    def __init__(self, config_directory: Optional[Path] = None):
        if config_directory is None:
            config_directory = Path(".agent-config")
        
        self.config_operations = AgentConfigOperations(config_directory)
        
    async def initialize(self):
        """Initialize the agent config manager tool"""
        try:
            # Ensure config directory exists
            await self.config_operations.create_agent_config_directory()
            logger.info("Agent config manager tool initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize agent config manager tool: %s", e)
            raise


@mcp_tool("agent_config_manager")
async def agent_config_manager_tool(
    action: str,
    agent_name: Optional[str] = None,
    config_data: Optional[Dict[str, Any]] = None,
    config_path: str = ".agent-config"
) -> Dict[str, Any]:
    """
    MCP tool for managing agent configuration files.
    
    Args:
        action: Action to perform (create/read/update/list/validate)
        agent_name: Name of the agent to manage
        config_data: Configuration data for create/update operations
        config_path: Path to agent configuration directory
        
    Returns:
        Agent configuration management results
    """
    try:
        # Initialize config operations
        config_ops = AgentConfigOperations(Path(config_path))
        
        # Ensure directory exists for all operations
        await config_ops.create_agent_config_directory()
        
        if action == "create":
            if not agent_name or not config_data:
                return {
                    "success": False,
                    "error": "agent_name and config_data are required for create action",
                    "action": action
                }
            
            try:
                file_path = await config_ops.create_agent_config_file(agent_name, config_data)
                return {
                    "success": True,
                    "action": action,
                    "agent_name": agent_name,
                    "file_path": str(file_path),
                    "message": f"Successfully created agent config for {agent_name}"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to create agent config: {str(e)}",
                    "action": action,
                    "agent_name": agent_name
                }
        
        elif action == "read":
            if not agent_name:
                return {
                    "success": False,
                    "error": "agent_name is required for read action",
                    "action": action
                }
            
            config_data = await config_ops.read_agent_config_file(agent_name)
            if config_data:
                return {
                    "success": True,
                    "action": action,
                    "agent_name": agent_name,
                    "config": config_data
                }
            else:
                return {
                    "success": False,
                    "error": f"Agent config not found for {agent_name}",
                    "action": action,
                    "agent_name": agent_name
                }
        
        elif action == "update":
            if not agent_name or not config_data:
                return {
                    "success": False,
                    "error": "agent_name and config_data are required for update action",
                    "action": action
                }
            
            success = await config_ops.update_agent_config_file(agent_name, config_data)
            if success:
                return {
                    "success": True,
                    "action": action,
                    "agent_name": agent_name,
                    "message": f"Successfully updated agent config for {agent_name}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to update agent config for {agent_name}",
                    "action": action,
                    "agent_name": agent_name
                }
        
        elif action == "list":
            configs = await config_ops.list_agent_config_files()
            return {
                "success": True,
                "action": action,
                "count": len(configs),
                "configs": configs,
                "message": f"Found {len(configs)} agent configuration files"
            }
        
        elif action == "validate":
            if not agent_name:
                return {
                    "success": False,
                    "error": "agent_name is required for validate action",
                    "action": action
                }
            
            validation_result = await config_ops.validate_agent_config_file(agent_name)
            return {
                "success": True,
                "action": action,
                "agent_name": agent_name,
                "validation": validation_result
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}. Supported actions: create, read, update, list, validate",
                "action": action
            }
    
    except Exception as e:
        logger.error("Agent config manager tool error: %s", e)
        return {
            "success": False,
            "error": f"Agent config manager tool error: {str(e)}",
            "action": action
        }


@mcp_tool("agent_config_directory_setup")
async def agent_config_directory_setup_tool(
    config_path: str = ".agent-config",
    create_examples: bool = False
) -> Dict[str, Any]:
    """
    MCP tool for setting up agent configuration directory structure.
    
    Args:
        config_path: Path to agent configuration directory
        create_examples: Whether to create example configuration files
        
    Returns:
        Directory setup results
    """
    try:
        config_ops = AgentConfigOperations(Path(config_path))
        
        # Create directory structure
        success = await config_ops.create_agent_config_directory()
        
        result = {
            "success": success,
            "config_path": config_path,
            "directories_created": []
        }
        
        if success:
            result["directories_created"] = [
                str(config_ops.config_dir),
                str(config_ops.specialists_dir)
            ]
            result["message"] = f"Successfully created agent config directory structure at {config_path}"
            
            if create_examples:
                # Create example specialist configuration
                example_specialist = {
                    "agent_type": "specialist",
                    "domain": "example_domain",
                    "capabilities": ["example_capability"],
                    "intersections": ["code"],
                    "memory_enabled": True,
                    "coordination_style": "standard",
                    "content": """# Example Specialist

## Role Overview

This is an example specialist configuration showing the expected structure and format.

## Expertise Areas

- Example expertise area 1
- Example expertise area 2

## Responsibilities

- Example responsibility 1
- Example responsibility 2
"""
                }
                
                try:
                    example_path = await config_ops.create_agent_config_file("example_specialist", example_specialist)
                    result["example_created"] = str(example_path)
                    result["message"] += f". Example specialist created at {example_path}"
                except Exception as e:
                    result["example_error"] = f"Failed to create example: {str(e)}"
        else:
            result["error"] = f"Failed to create agent config directory structure at {config_path}"
        
        return result
    
    except Exception as e:
        logger.error("Agent config directory setup error: %s", e)
        return {
            "success": False,
            "error": f"Directory setup error: {str(e)}",
            "config_path": config_path
        }


# Export MCP tools for registration
MCP_TOOLS = [
    agent_config_manager_tool,
    agent_config_directory_setup_tool
]


# Tool registry for easy access
TOOL_REGISTRY = {
    "agent_config_manager": agent_config_manager_tool,
    "agent_config_directory_setup": agent_config_directory_setup_tool
}