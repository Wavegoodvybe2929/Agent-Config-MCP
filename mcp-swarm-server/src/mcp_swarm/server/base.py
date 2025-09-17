"""
Base MCP Server implementation for swarm intelligence.
"""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import asyncio
import logging

if TYPE_CHECKING:
    from .tools import ToolRegistry
    from .resources import ResourceManager  
    from .messages import MessageHandler

try:
    from mcp.types import (
        ServerCapabilities, 
        ToolsCapability, 
        ResourcesCapability
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Fallback for development/testing - use Any for compatibility
    ServerCapabilities = type("ServerCapabilities", (), {"__init__": lambda self, **kwargs: setattr(self, "__dict__", kwargs)})
    ToolsCapability = type("ToolsCapability", (), {"__init__": lambda self, **kwargs: setattr(self, "__dict__", kwargs)})
    ResourcesCapability = type("ResourcesCapability", (), {"__init__": lambda self, **kwargs: setattr(self, "__dict__", kwargs)})
    # Fallback for development/testing
    class MCPServerCapabilities:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class MCPTool:
        pass
    
    class MCPResource:
        pass
    
    class MCPToolsCapability:
        def __init__(self):
            pass
    
    class MCPResourcesCapability:
        def __init__(self):
            pass


class ToolInfo:
    """Tool information for registration."""
    def __init__(self, name: str, description: str = "", parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}


class ResourceInfo:
    """Resource information for registration."""
    def __init__(self, uri: str, name: str = "", description: str = "", mimeType: str = "text/plain"):
        self.uri = uri
        self.name = name or uri
        self.description = description
        self.mimeType = mimeType


class SwarmMCPServer:
    """Enhanced MCP Server with swarm intelligence capabilities."""
    
    def __init__(self, name: str = "swarm-intelligence-server", version: str = "1.0.0"):
        """Initialize the swarm MCP server.
        
        Args:
            name: Server name identifier
            version: Server version
        """
        self.name = name
        self.version = version
        self._initialized = False
        self._tools_registry: Dict[str, Any] = {}
        self._resources_cache: Dict[str, Any] = {}
        self._memory_manager: Optional[Any] = None
        self._swarm_coordinator: Optional[Any] = None
        self._message_handlers: Dict[str, Any] = {}
        self._logger = logging.getLogger(f"mcp.swarm.{name}")
        
        # Component references for integration
        self._tool_registry: Optional["ToolRegistry"] = None
        self._resource_manager: Optional["ResourceManager"] = None
        self._message_handler: Optional["MessageHandler"] = None
        
        # Initialize capabilities
        if MCP_AVAILABLE:
            self._capabilities = ServerCapabilities(  # type: ignore
                tools=ToolsCapability(),  # type: ignore
                resources=ResourcesCapability(),  # type: ignore
                experimental={
                    "swarm_coordination": {"version": "1.0"},
                    "memory_persistence": {"version": "1.0"}
                }
            )
        else:
            self._capabilities = ServerCapabilities(  # type: ignore
                tools=ToolsCapability(),  # type: ignore
                resources=ResourcesCapability(),  # type: ignore
                experimental={
                    "swarm_coordination": {"version": "1.0"},
                    "memory_persistence": {"version": "1.0"}
                }
            )
        
    async def initialize(self) -> None:
        """Initialize server with swarm components."""
        if self._initialized:
            return
            
        self._logger.info("Initializing %s v%s", self.name, self.version)
        
        # Initialize memory manager if available
        try:
            from ..memory.manager import MemoryManager
            self._memory_manager = MemoryManager()
            if hasattr(self._memory_manager, 'initialize'):
                await self._memory_manager.initialize()
            self._logger.info("Memory manager initialized")
        except ImportError:
            self._logger.warning("Memory manager not available")
        except Exception as e:
            self._logger.warning("Memory manager initialization failed: %s", e)
            
        # Initialize swarm coordinator if available
        try:
            from ..swarm.coordinator import SwarmCoordinator
            self._swarm_coordinator = SwarmCoordinator()
            if hasattr(self._swarm_coordinator, 'initialize'):
                await self._swarm_coordinator.initialize()
            self._logger.info("Swarm coordinator initialized")
        except ImportError:
            self._logger.warning("Swarm coordinator not available")
        except Exception as e:
            self._logger.warning("Swarm coordinator initialization failed: %s", e)
            
        # Setup message handlers
        self._setup_message_handlers()
        
        self._initialized = True
        self._logger.info("Server initialization complete")
        
    def _setup_message_handlers(self) -> None:
        """Setup MCP message handlers."""
        # Setup basic handlers for now
        self._message_handlers.update({
            "initialize": self._handle_initialize,
            "tools/list": self._handle_list_tools,
            "tools/call": self._handle_call_tool,
            "resources/list": self._handle_list_resources,
            "resources/read": self._handle_read_resource,
        })
        
    async def _handle_initialize(self, request: Any) -> Dict[str, Any]:  # pylint: disable=unused-argument
        """Handle initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": vars(self._capabilities) if hasattr(self._capabilities, '__dict__') else {},
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }
        
    async def _handle_list_tools(self, request: Any) -> Dict[str, Any]:  # pylint: disable=unused-argument
        """Handle list tools request."""
        tools = await self.get_tools()
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.parameters
                }
                for tool in tools
            ]
        }
        
    async def _handle_call_tool(self, request: Any) -> Dict[str, Any]:
        """Handle call tool request."""
        params = getattr(request, 'params', {})
        name = params.get('name')
        arguments = params.get('arguments', {})
        
        if not name:
            raise ValueError("Tool name is required")
            
        result = await self.call_tool(name, arguments)
        return {
            "content": [
                {
                    "type": "text", 
                    "text": str(result)
                }
            ]
        }
        
    async def _handle_list_resources(self, request: Any) -> Dict[str, Any]:  # pylint: disable=unused-argument
        """Handle list resources request."""
        resources = await self.list_resources()
        return {
            "resources": [
                {
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mimeType": resource.mimeType
                }
                for resource in resources
            ]
        }
        
    async def _handle_read_resource(self, request: Any) -> Dict[str, Any]:
        """Handle read resource request."""
        params = getattr(request, 'params', {})
        uri = params.get('uri')
        
        if not uri:
            raise ValueError("Resource URI is required")
            
        resource = await self.get_resource(uri)
        if not resource:
            raise ValueError(f"Resource not found: {uri}")
            
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": getattr(resource, 'mime_type', 'text/plain'),
                    "text": getattr(resource, 'content', str(resource))
                }
            ]
        }
        
    async def handle_request(self, request: Any) -> Any:
        """Handle incoming MCP requests with swarm coordination.
        
        Args:
            request: MCP request message
            
        Returns:
            MCP response message
        """
        if not self._initialized:
            await self.initialize()
            
        method = getattr(request, 'method', None)
        if not method:
            raise ValueError("Request missing method")
            
        handler = self._message_handlers.get(method)
        if not handler:
            raise ValueError(f"Unknown method: {method}")
            
        try:
            # Log request for swarm coordination
            if (self._memory_manager and 
                hasattr(self._memory_manager, 'log_request')):
                await self._memory_manager.log_request({"method": method, "request": request})
                
            # Execute request handler
            result = await handler(request)
            
            # Log successful response
            if (self._memory_manager and 
                hasattr(self._memory_manager, 'log_response')):
                await self._memory_manager.log_response({"method": method, "result": result, "success": True})
                
            return result
            
        except Exception as e:
            # Log error
            if (self._memory_manager and 
                hasattr(self._memory_manager, 'log_response')):
                await self._memory_manager.log_response({"method": method, "error": str(e), "success": False})
            raise
        
    async def register_tool(self, tool: Union[Any, Dict[str, Any]]) -> None:
        """Register a new tool with automatic discovery.
        
        Args:
            tool: Tool instance or tool definition dict
        """
        if isinstance(tool, dict):
            # Convert dict to ToolInfo
            tool_info = ToolInfo(
                name=tool['name'],
                description=tool.get('description', ''),
                parameters=tool.get('parameters', {})
            )
        else:
            # Assume it's a Tool-like object
            tool_info = ToolInfo(
                name=getattr(tool, 'name', str(tool)),
                description=getattr(tool, 'description', ''),
                parameters=getattr(tool, 'parameters', {})
            )
            
        self._tools_registry[tool_info.name] = tool
        self._logger.info("Registered tool: %s", tool_info.name)
        
        # Notify swarm coordinator of new tool
        if (self._swarm_coordinator and 
            hasattr(self._swarm_coordinator, 'register_tool')):
            await self._swarm_coordinator.register_tool(tool_info.name, tool)
            
    async def get_tools(self) -> List[ToolInfo]:
        """Get list of available tools.
        
        Returns:
            List of tool information
        """
        tools = []
        for name, tool in self._tools_registry.items():
            if isinstance(tool, dict):
                tool_info = ToolInfo(
                    name=tool['name'],
                    description=tool.get('description', ''),
                    parameters=tool.get('parameters', {})
                )
            else:
                tool_info = ToolInfo(
                    name=getattr(tool, 'name', name),
                    description=getattr(tool, 'description', ''),
                    parameters=getattr(tool, 'parameters', {})
                )
            tools.append(tool_info)
        return tools
        
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with swarm coordination.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if name not in self._tools_registry:
            raise ValueError(f"Tool not found: {name}")
            
        tool = self._tools_registry[name]
        
        # Get swarm coordination if needed
        if (self._swarm_coordinator and 
            hasattr(self._swarm_coordinator, 'coordinate_tool_call') and
            hasattr(tool, 'requires_consensus') and 
            getattr(tool, 'requires_consensus', False)):
            arguments = await self._swarm_coordinator.coordinate_tool_call(name, arguments)
                
        # Execute tool
        if hasattr(tool, 'execute'):
            result = tool.execute(**arguments)
            return await result if asyncio.iscoroutine(result) else result
        elif callable(tool):
            result = tool(**arguments)
            return await result if asyncio.iscoroutine(result) else result
        elif isinstance(tool, dict) and 'handler' in tool:
            handler = tool['handler']
            result = handler(**arguments)
            return await result if asyncio.iscoroutine(result) else result
        else:
            raise ValueError(f"Tool {name} is not executable")
            
    async def register_resource(self, resource: Union[Any, Dict[str, Any]]) -> None:
        """Register a new resource.
        
        Args:
            resource: Resource to register
        """
        if isinstance(resource, dict):
            uri = resource['uri']
        else:
            uri = getattr(resource, 'uri', str(resource))
            
        self._resources_cache[uri] = resource
        self._logger.info("Registered resource: %s", uri)
        
    async def get_resource(self, uri: str) -> Optional[Any]:
        """Get a resource by URI.
        
        Args:
            uri: Resource URI
            
        Returns:
            Resource if found, None otherwise
        """
        return self._resources_cache.get(uri)
        
    async def list_resources(self) -> List[ResourceInfo]:
        """List available resources.
        
        Returns:
            List of resource information
        """
        resources = []
        for uri, resource in self._resources_cache.items():
            if isinstance(resource, dict):
                resource_info = ResourceInfo(
                    uri=uri,
                    name=resource.get('name', uri),
                    description=resource.get('description', ''),
                    mimeType=resource.get('mimeType', 'text/plain')
                )
            else:
                resource_info = ResourceInfo(
                    uri=uri,
                    name=getattr(resource, 'name', uri),
                    description=getattr(resource, 'description', ''),
                    mimeType=getattr(resource, 'mime_type', 'text/plain')
                )
            resources.append(resource_info)
        return resources
        
    async def shutdown(self) -> None:
        """Shutdown the server gracefully."""
        if not self._initialized:
            return
            
        self._logger.info("Shutting down server")
        
        # Shutdown swarm coordinator
        if (self._swarm_coordinator and 
            hasattr(self._swarm_coordinator, 'shutdown')):
            await self._swarm_coordinator.shutdown()
            
        # Shutdown memory manager
        if (self._memory_manager and 
            hasattr(self._memory_manager, 'shutdown')):
            await self._memory_manager.shutdown()
            
        self._initialized = False
        self._logger.info("Server shutdown complete")
        
    @property
    def capabilities(self) -> Any:
        """Get server capabilities.
        
        Returns:
            Server capabilities
        """
        return self._capabilities