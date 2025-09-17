"""
Tool Registration System for MCP Swarm Server

Provides dynamic tool discovery and registration with metadata validation
and automatic discovery capabilities for swarm intelligence coordination.
"""

from typing import Dict, Any, Callable, List, Optional, Union
from dataclasses import dataclass, field
import asyncio
import inspect
import logging
from datetime import datetime


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    requires_consensus: bool = False
    agent_assignment: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    last_used: Optional[datetime] = None
    success_rate: float = 1.0


@dataclass
class ToolExecutionResult:
    """Result of tool execution with metadata."""
    success: bool
    result: Any
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Dynamic tool discovery and registration system."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, ToolMetadata] = {}
        self._logger = logging.getLogger("mcp.swarm.tools")
        
    async def discover_tools(self) -> List[ToolMetadata]:
        """Automatically discover available tools.
        
        Returns:
            List of discovered tool metadata
        """
        discovered_tools = []
        
        # Look for tools in common locations
        tool_locations = [
            "mcp_swarm.tools",
            "mcp_swarm.agents.tools",
            "mcp_swarm.swarm.tools",
        ]
        
        for location in tool_locations:
            try:
                tools = await self._discover_tools_in_module(location)
                discovered_tools.extend(tools)
            except ImportError:
                self._logger.debug("Tool location not found: %s", location)
            except Exception as e:
                self._logger.warning("Error discovering tools in %s: %s", location, str(e))
                
        self._logger.info("Discovered %d tools", len(discovered_tools))
        return discovered_tools
        
    async def _discover_tools_in_module(self, module_name: str) -> List[ToolMetadata]:
        """Discover tools in a specific module.
        
        Args:
            module_name: Name of the module to search
            
        Returns:
            List of discovered tools
        """
        try:
            import importlib
            module = importlib.import_module(module_name)
            tools = []
            
            for name in dir(module):
                obj = getattr(module, name)
                
                # Check if it's a tool function
                if (callable(obj) and 
                    hasattr(obj, '__annotations__') and
                    not name.startswith('_')):
                    
                    tool_metadata = self._extract_tool_metadata(name, obj)
                    if tool_metadata:
                        tools.append(tool_metadata)
                        
            return tools
            
        except Exception as e:
            self._logger.error("Failed to discover tools in %s: %s", module_name, str(e))
            return []
            
    def _extract_tool_metadata(self, name: str, func: Callable) -> Optional[ToolMetadata]:
        """Extract metadata from a function to create tool metadata.
        
        Args:
            name: Function name
            func: Function object
            
        Returns:
            ToolMetadata if valid tool, None otherwise
        """
        try:
            # Get function signature
            sig = inspect.signature(func)
            parameters = {}
            
            # Extract parameter information
            for param_name, param in sig.parameters.items():
                param_info = {
                    "type": self._get_type_name(param.annotation),
                    "required": param.default == inspect.Parameter.empty
                }
                
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                    
                parameters[param_name] = param_info
                
            # Get description from docstring
            description = func.__doc__ or f"Tool: {name}"
            
            # Check for consensus requirement
            requires_consensus = getattr(func, '_requires_consensus', False)
            
            # Check for agent assignment
            agent_assignment = getattr(func, '_agent_assignment', None)
            
            return ToolMetadata(
                name=name,
                description=description.strip(),
                parameters=parameters,
                handler=func,
                requires_consensus=requires_consensus,
                agent_assignment=agent_assignment
            )
            
        except Exception as e:
            self._logger.warning("Failed to extract metadata for %s: %s", name, str(e))
            return None
            
    def _get_type_name(self, annotation: Any) -> str:
        """Get string representation of type annotation.
        
        Args:
            annotation: Type annotation
            
        Returns:
            String name of the type
        """
        if annotation == inspect.Parameter.empty:
            return "any"
        elif hasattr(annotation, '__name__'):
            return annotation.__name__
        else:
            return str(annotation)
            
    async def register_tool(self, metadata: ToolMetadata) -> None:
        """Register a new tool with validation.
        
        Args:
            metadata: Tool metadata to register
            
        Raises:
            ValueError: If tool validation fails
        """
        # Validate tool metadata
        await self._validate_tool_metadata(metadata)
        
        # Register the tool
        self._tools[metadata.name] = metadata
        self._logger.info("Registered tool: %s", metadata.name)
        
    async def register_tool_function(
        self, 
        name: str, 
        handler: Callable, 
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        requires_consensus: bool = False,
        agent_assignment: Optional[str] = None
    ) -> None:
        """Register a tool from a function.
        
        Args:
            name: Tool name
            handler: Function to handle tool execution
            description: Tool description
            parameters: Parameter schema
            requires_consensus: Whether tool requires swarm consensus
            agent_assignment: Specific agent assignment
        """
        if parameters is None:
            # Extract parameters from function signature
            metadata = self._extract_tool_metadata(name, handler)
            if metadata:
                parameters = metadata.parameters
            else:
                parameters = {}
                
        tool_metadata = ToolMetadata(
            name=name,
            description=description or f"Tool: {name}",
            parameters=parameters,
            handler=handler,
            requires_consensus=requires_consensus,
            agent_assignment=agent_assignment
        )
        
        await self.register_tool(tool_metadata)
        
    async def _validate_tool_metadata(self, metadata: ToolMetadata) -> None:
        """Validate tool metadata.
        
        Args:
            metadata: Tool metadata to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not metadata.name:
            raise ValueError("Tool name is required")
            
        if not callable(metadata.handler):
            raise ValueError("Tool handler must be callable")
            
        if metadata.name in self._tools:
            raise ValueError(f"Tool {metadata.name} is already registered")
            
        # Validate parameters schema
        if not isinstance(metadata.parameters, dict):
            raise ValueError("Tool parameters must be a dictionary")
            
    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """Execute tool with swarm coordination.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found or execution fails
        """
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
            
        tool = self._tools[name]
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Validate arguments
            await self._validate_arguments(tool, arguments)
            
            # Execute the tool
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                result = tool.handler(**arguments)
                
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Update tool statistics
            await self._update_tool_statistics(tool, success=True, execution_time=execution_time)
            
            return ToolExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={
                    "tool_name": name,
                    "arguments": arguments,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Update tool statistics
            await self._update_tool_statistics(tool, success=False, execution_time=execution_time)
            
            self._logger.error("Tool execution failed for %s: %s", name, str(e))
            
            return ToolExecutionResult(
                success=False,
                result=None,
                execution_time=execution_time,
                error=str(e),
                metadata={
                    "tool_name": name,
                    "arguments": arguments,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
    async def _validate_arguments(self, tool: ToolMetadata, arguments: Dict[str, Any]) -> None:
        """Validate tool arguments against parameter schema.
        
        Args:
            tool: Tool metadata
            arguments: Arguments to validate
            
        Raises:
            ValueError: If validation fails
        """
        for param_name, param_info in tool.parameters.items():
            if param_info.get("required", False) and param_name not in arguments:
                raise ValueError(f"Required parameter {param_name} missing for tool {tool.name}")
                
        # Check for unexpected arguments
        for arg_name in arguments:
            if arg_name not in tool.parameters:
                self._logger.warning("Unexpected argument %s for tool %s", arg_name, tool.name)
                
    async def _update_tool_statistics(self, tool: ToolMetadata, success: bool, execution_time: float) -> None:
        """Update tool usage statistics.
        
        Args:
            tool: Tool metadata
            success: Whether execution was successful
            execution_time: Time taken for execution
        """
        tool.usage_count += 1
        tool.last_used = datetime.now()
        
        # Update success rate using exponential moving average
        alpha = 0.1  # Smoothing factor
        if success:
            tool.success_rate = tool.success_rate * (1 - alpha) + alpha
        else:
            tool.success_rate = tool.success_rate * (1 - alpha)
            
    async def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool metadata if found, None otherwise
        """
        return self._tools.get(name)
        
    async def list_tools(self) -> List[ToolMetadata]:
        """List all registered tools.
        
        Returns:
            List of all tool metadata
        """
        return list(self._tools.values())
        
    async def list_tools_by_agent(self, agent_id: str) -> List[ToolMetadata]:
        """List tools assigned to a specific agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of tools assigned to the agent
        """
        return [
            tool for tool in self._tools.values()
            if tool.agent_assignment == agent_id
        ]
        
    async def get_tool_statistics(self) -> Dict[str, Any]:
        """Get overall tool usage statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self._tools:
            return {
                "total_tools": 0,
                "total_executions": 0,
                "average_success_rate": 0.0
            }
            
        total_executions = sum(tool.usage_count for tool in self._tools.values())
        average_success_rate = sum(tool.success_rate for tool in self._tools.values()) / len(self._tools)
        
        return {
            "total_tools": len(self._tools),
            "total_executions": total_executions,
            "average_success_rate": average_success_rate,
            "tools": [
                {
                    "name": tool.name,
                    "usage_count": tool.usage_count,
                    "success_rate": tool.success_rate,
                    "last_used": tool.last_used.isoformat() if tool.last_used else None
                }
                for tool in self._tools.values()
            ]
        }
        
    async def unregister_tool(self, name: str) -> bool:
        """Unregister a tool.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            self._logger.info("Unregistered tool: %s", name)
            return True
        return False


# Decorator for marking functions as tools
def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    requires_consensus: bool = False,
    agent_assignment: Optional[str] = None
):
    """Decorator to mark a function as a tool.
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        requires_consensus: Whether tool requires swarm consensus
        agent_assignment: Specific agent assignment
    """
    def decorator(func: Callable) -> Callable:
        func._tool_name = name or func.__name__  # type: ignore
        func._tool_description = description or func.__doc__  # type: ignore
        func._requires_consensus = requires_consensus  # type: ignore
        func._agent_assignment = agent_assignment  # type: ignore
        func._is_tool = True  # type: ignore
        return func
    
    return decorator


# Decorator for marking consensus-required tools
def consensus_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    agent_assignment: Optional[str] = None
):
    """Decorator to mark a function as requiring consensus.
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        agent_assignment: Specific agent assignment
    """
    return tool(
        name=name,
        description=description,
        requires_consensus=True,
        agent_assignment=agent_assignment
    )