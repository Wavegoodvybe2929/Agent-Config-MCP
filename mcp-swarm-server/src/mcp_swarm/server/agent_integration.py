"""
Agent-Config Integration Engine for MCP Swarm Intelligence Server.

This module provides the integration layer between the MCP server and the
agent-config system, enabling automatic tool discovery, registration, and
orchestrator workflow coordination.
"""

from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import yaml
import re

from ..agents.config_scanner import AgentConfigScanner
from ..agents.capability_matrix import CapabilityMatrixGenerator
from ..server.tools import ToolRegistry, ToolMetadata
from ..memory.manager import MemoryManager


@dataclass
class AgentToolDefinition:
    """Definition of a tool extracted from agent configuration."""
    name: str
    description: str
    agent_source: str
    capabilities: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_consensus: bool = False
    orchestrator_routing: bool = True


@dataclass
class IntegrationStatus:
    """Status of agent-config integration."""
    agents_discovered: int
    tools_extracted: int
    tools_registered: int
    last_scan: Optional[datetime] = None
    integration_errors: List[str] = field(default_factory=list)


class AgentConfigIntegration:
    """
    Integrate MCP server with agent-config system for automatic tool discovery.
    
    This engine provides the bridge between the agent configuration system and
    the MCP server, enabling automatic discovery and registration of tools from
    agent configurations while maintaining orchestrator workflow patterns.
    """
    
    def __init__(
        self, 
        config_directory: Optional[Path] = None,
        tool_registry: Optional[ToolRegistry] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        """Initialize the agent-config integration engine."""
        self.config_directory = config_directory or Path(".agent-config")
        self.tool_registry = tool_registry
        self.memory_manager = memory_manager
        
        # Initialize scanner and capability analyzer
        self.config_scanner = AgentConfigScanner(self.config_directory)
        self.capability_matrix = CapabilityMatrixGenerator()
        
        # Integration state
        self._status = IntegrationStatus(agents_discovered=0, tools_extracted=0, tools_registered=0)
        self._discovered_tools: Dict[str, AgentToolDefinition] = {}
        self._agent_mappings: Dict[str, Dict[str, Any]] = {}
        
        self._logger = logging.getLogger("mcp.swarm.integration")
        
    async def initialize_integration(self) -> Dict[str, Any]:
        """Initialize the agent-config integration system."""
        try:
            self._logger.info("Initializing agent-config integration engine")
            
            # Verify config directory exists
            if not self.config_directory.exists():
                self._logger.warning("Agent config directory does not exist: %s", self.config_directory)
                return {
                    "success": False,
                    "error": f"Agent config directory not found: {self.config_directory}",
                    "recommendation": "Create .agent-config directory with agent configurations"
                }
            
            # Perform initial discovery
            discovery_result = await self.discover_and_integrate_agents()
            
            if discovery_result["success"]:
                self._logger.info("Agent-config integration initialized successfully")
                return {
                    "success": True,
                    "message": "Agent-config integration initialized successfully",
                    "status": self._get_status_dict(),
                    "discovery_result": discovery_result
                }
            else:
                return discovery_result
                
        except Exception as e:
            error_msg = f"Failed to initialize agent-config integration: {str(e)}"
            self._logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
    
    async def discover_and_integrate_agents(self) -> Dict[str, Any]:
        """Discover agents and integrate their tools with MCP server."""
        try:
            self._logger.info("Starting agent discovery and integration")
            
            # Clear previous state
            self._discovered_tools.clear()
            self._agent_mappings.clear()
            self._status.integration_errors.clear()
            
            # Scan for agent configurations
            agent_configs = await self.config_scanner.scan_agent_configurations()
            self._status.agents_discovered = len(agent_configs)
            
            # Process each agent configuration
            tools_extracted = 0
            tools_registered = 0
            
            for config_path in agent_configs:
                try:
                    config_path_obj = Path(config_path) if isinstance(config_path, str) else config_path
                    agent_data = await self._parse_agent_config(config_path_obj)
                    if agent_data:
                        # Extract tools from agent configuration
                        extracted_tools = await self._extract_tools_from_agent(config_path_obj, agent_data)
                        tools_extracted += len(extracted_tools)
                        
                        # Register tools with MCP server
                        if self.tool_registry:
                            for tool_def in extracted_tools:
                                registration_result = await self._register_tool_with_mcp(tool_def)
                                if registration_result:
                                    tools_registered += 1
                                    self._discovered_tools[tool_def.name] = tool_def
                        
                        # Store agent mapping
                        self._agent_mappings[str(config_path)] = agent_data
                        
                except Exception as e:
                    error_msg = f"Error processing agent config {config_path}: {str(e)}"
                    self._logger.error(error_msg)
                    self._status.integration_errors.append(error_msg)
            
            # Update status
            self._status.tools_extracted = tools_extracted
            self._status.tools_registered = tools_registered
            self._status.last_scan = datetime.now()
            
            success_rate = (tools_registered / tools_extracted) if tools_extracted > 0 else 1.0
            
            return {
                "success": True,
                "message": f"Agent discovery completed - {len(agent_configs)} agents, {tools_extracted} tools extracted, {tools_registered} registered",
                "agents_discovered": len(agent_configs),
                "tools_extracted": tools_extracted,
                "tools_registered": tools_registered,
                "success_rate": success_rate,
                "errors": self._status.integration_errors
            }
            
        except Exception as e:
            error_msg = f"Failed to discover and integrate agents: {str(e)}"
            self._logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
    
    async def register_tools_with_mcp_server(self) -> Dict[str, Any]:
        """Register discovered tools with the MCP server."""
        try:
            if not self.tool_registry:
                return {
                    "success": False,
                    "error": "Tool registry not available"
                }
            
            registered_count = 0
            failed_registrations = []
            
            for tool_name, tool_def in self._discovered_tools.items():
                try:
                    success = await self._register_tool_with_mcp(tool_def)
                    if success:
                        registered_count += 1
                    else:
                        failed_registrations.append(tool_name)
                except Exception as e:
                    failed_registrations.append(f"{tool_name}: {str(e)}")
            
            return {
                "success": True,
                "registered_tools": registered_count,
                "total_tools": len(self._discovered_tools),
                "failed_registrations": failed_registrations
            }
            
        except Exception as e:
            self._logger.error("Failed to register tools with MCP server: %s", str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status and statistics."""
        return {
            "success": True,
            "status": self._get_status_dict(),
            "discovered_tools": {
                name: {
                    "agent_source": tool.agent_source,
                    "capabilities": tool.capabilities,
                    "requires_consensus": tool.requires_consensus,
                    "orchestrator_routing": tool.orchestrator_routing
                }
                for name, tool in self._discovered_tools.items()
            },
            "agent_mappings": list(self._agent_mappings.keys())
        }
    
    async def refresh_integration(self) -> Dict[str, Any]:
        """Refresh agent discovery and tool integration."""
        self._logger.info("Refreshing agent-config integration")
        return await self.discover_and_integrate_agents()
    
    async def _parse_agent_config(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """Parse agent configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract YAML frontmatter if present
            frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
            if frontmatter_match:
                yaml_content = frontmatter_match.group(1)
                frontmatter = yaml.safe_load(yaml_content)
                return frontmatter
            
            # If no frontmatter, try to parse the entire file as YAML
            try:
                return yaml.safe_load(content)
            except yaml.YAMLError:
                # Not YAML, treat as markdown with embedded metadata
                return self._extract_metadata_from_markdown(content)
                
        except Exception as e:
            self._logger.error("Failed to parse agent config %s: %s", config_path, str(e))
            return None
    
    def _extract_metadata_from_markdown(self, content: str) -> Dict[str, Any]:
        """Extract metadata from markdown content."""
        metadata = {}
        
        # Extract agent type
        if "specialist" in content.lower():
            metadata["agent_type"] = "specialist"
        elif "orchestrator" in content.lower():
            metadata["agent_type"] = "orchestrator"
        else:
            metadata["agent_type"] = "general"
        
        # Extract capabilities from content
        capabilities = []
        capability_patterns = [
            r"capabilities?[:\s]+\[(.*?)\]",
            r"expertise[:\s]+(.*)",
            r"responsibilities?[:\s]+(.*)"
        ]
        
        for pattern in capability_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str):
                    # Clean and split capabilities
                    caps = [cap.strip().strip('"\'') for cap in match.split(',')]
                    capabilities.extend(caps)
        
        metadata["capabilities"] = capabilities
        return metadata
    
    async def _extract_tools_from_agent(
        self, 
        config_path: Path, 
        agent_data: Dict[str, Any]
    ) -> List[AgentToolDefinition]:
        """Extract tool definitions from agent configuration."""
        tools = []
        agent_name = config_path.stem
        
        # Get agent capabilities
        capabilities = agent_data.get("capabilities", [])
        agent_type = agent_data.get("agent_type", "general")
        
        # Define common tool patterns for different agent types
        if agent_type == "specialist":
            # Specialist agents typically provide domain-specific tools
            tools.append(AgentToolDefinition(
                name=f"{agent_name}_consultation",
                description=f"Consult with {agent_name} for specialized expertise",
                agent_source=agent_name,
                capabilities=capabilities,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Consultation query"},
                        "context": {"type": "string", "description": "Additional context"}
                    },
                    "required": ["query"]
                },
                requires_consensus=agent_data.get("requires_consensus", False),
                orchestrator_routing=agent_data.get("orchestrator_routing", True)
            ))
        
        elif agent_type == "orchestrator":
            # Orchestrator provides coordination tools
            tools.append(AgentToolDefinition(
                name="orchestrator_coordination",
                description="Coordinate multi-agent workflows and task assignments",
                agent_source=agent_name,
                capabilities=capabilities,
                parameters={
                    "type": "object", 
                    "properties": {
                        "task": {"type": "string", "description": "Task to coordinate"},
                        "agents": {"type": "array", "description": "Agents to coordinate"},
                        "priority": {"type": "string", "description": "Task priority"}
                    },
                    "required": ["task"]
                },
                requires_consensus=False,
                orchestrator_routing=False  # Orchestrator doesn't route to itself
            ))
        
        # Add memory-based tools if agent has memory capability
        if agent_data.get("memory_enabled", False):
            tools.append(AgentToolDefinition(
                name=f"{agent_name}_memory_query",
                description=f"Query {agent_name}'s persistent memory",
                agent_source=agent_name,
                capabilities=capabilities + ["memory_management"],
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Memory query"},
                        "context": {"type": "string", "description": "Query context"}
                    },
                    "required": ["query"]
                }
            ))
        
        return tools
    
    async def _register_tool_with_mcp(self, tool_def: AgentToolDefinition) -> bool:
        """Register a tool definition with the MCP server."""
        try:
            if not self.tool_registry:
                return False
            
            # Create tool metadata for MCP registration
            tool_metadata = ToolMetadata(
                name=tool_def.name,
                description=tool_def.description,
                parameters=tool_def.parameters,
                handler=self._create_tool_handler(tool_def),
                requires_consensus=tool_def.requires_consensus,
                agent_assignment=tool_def.agent_source
            )
            
            # Register with tool registry
            await self.tool_registry.register_tool(tool_metadata)
            
            self._logger.debug("Registered tool %s from agent %s", 
                             tool_def.name, tool_def.agent_source)
            return True
            
        except Exception as e:
            self._logger.error("Failed to register tool %s: %s", tool_def.name, str(e))
            return False
    
    def _create_tool_handler(self, tool_def: AgentToolDefinition):
        """Create a tool handler function for the given tool definition."""
        async def tool_handler(**kwargs):
            """Dynamic tool handler for agent-discovered tools."""
            # This would route the tool execution to the appropriate agent
            # For now, return a placeholder response
            return {
                "success": True,
                "agent_source": tool_def.agent_source,
                "capabilities_used": tool_def.capabilities,
                "message": f"Tool {tool_def.name} executed via agent {tool_def.agent_source}",
                "parameters": kwargs
            }
        
        return tool_handler
    
    def _get_status_dict(self) -> Dict[str, Any]:
        """Get integration status as dictionary."""
        return {
            "agents_discovered": self._status.agents_discovered,
            "tools_extracted": self._status.tools_extracted,
            "tools_registered": self._status.tools_registered,
            "last_scan": self._status.last_scan.isoformat() if self._status.last_scan else None,
            "integration_errors": self._status.integration_errors,
            "config_directory": str(self.config_directory),
            "discovered_tools_count": len(self._discovered_tools)
        }


# MCP Tool for Agent-Config Integration Management
async def agent_config_integration_tool(
    action: str,
    config_directory: Optional[str] = None
) -> Dict[str, Any]:
    """
    MCP tool for managing agent-config integration with MCP server.
    
    This tool provides management interface for the agent-config integration
    system, enabling discovery, registration, and monitoring of agent-based
    tools within the MCP server framework.
    
    Args:
        action: Action to perform (initialize/discover/register/status/refresh)
        config_directory: Optional path to agent configuration directory
        refresh_cache: Force refresh of cached data
        
    Returns:
        Dictionary containing integration results and status information
        
    Examples:
        # Initialize integration
        await agent_config_integration_tool("initialize")
        
        # Discover and integrate agents
        await agent_config_integration_tool("discover")
        
        # Get integration status
        await agent_config_integration_tool("status")
    """
    
    # Initialize integration engine
    config_dir = Path(config_directory) if config_directory else Path(".agent-config")
    integration_engine = AgentConfigIntegration(config_directory=config_dir)
    
    # Execute requested action
    if action == "initialize":
        return await integration_engine.initialize_integration()
    elif action == "discover":
        return await integration_engine.discover_and_integrate_agents()
    elif action == "register":
        return await integration_engine.register_tools_with_mcp_server()
    elif action == "status":
        return await integration_engine.get_integration_status()
    elif action == "refresh":
        return await integration_engine.refresh_integration()
    else:
        return {
            "success": False,
            "error": f"Unknown action: {action}",
            "valid_actions": ["initialize", "discover", "register", "status", "refresh"]
        }


# Tool metadata for MCP registration
AGENT_CONFIG_INTEGRATION_TOOL = {
    "name": "agent_config_integration",
    "description": "Integrate MCP server with agent-config system for automatic tool discovery",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["initialize", "discover", "register", "status", "refresh"],
                "description": "Action to perform on agent-config integration"
            },
            "config_directory": {
                "type": "string",
                "description": "Path to agent configuration directory",
                "default": ".agent-config"
            },
            "refresh_cache": {
                "type": "boolean",
                "default": False,
                "description": "Force refresh of cached discovery data"
            }
        },
        "required": ["action"]
    }
}