"""
Message Handler System for MCP Swarm Server

Implements MCP message handling with JSON-RPC 2.0 compliance
and proper error propagation for swarm intelligence coordination.
"""

from typing import Any, Dict, Optional
import asyncio
import logging
import json
from datetime import datetime
from dataclasses import dataclass

try:
    # Check if MCP is available but don't import to avoid conflicts
    import importlib.util
    mcp_spec = importlib.util.find_spec('mcp')
    MCP_AVAILABLE = mcp_spec is not None
except ImportError:
    MCP_AVAILABLE = False

# Define our own JSONRPCError that inherits from Exception
class JSONRPCError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


# JSON-RPC Error Codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


@dataclass
class MessageMetrics:
    """Metrics for message handling."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    methods_called: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        if self.methods_called is None:
            self.methods_called = {}


class MessageHandler:
    """Handle MCP messages with error propagation."""
    
    def __init__(self, server):
        """Initialize message handler.
        
        Args:
            server: Reference to the SwarmMCPServer instance
        """
        self.server = server
        self._logger = logging.getLogger("mcp.swarm.messages")
        self._metrics = MessageMetrics()
        self._register_handlers()
        
    def _register_handlers(self) -> None:
        """Register message handlers."""
        # This will be called by the server setup
        
    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming message with proper error handling.
        
        Args:
            message: Incoming JSON-RPC message
            
        Returns:
            Response message
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Validate JSON-RPC structure
            self._validate_jsonrpc_message(message)
            
            method = message.get('method')
            request_id = message.get('id')
            
            if not method:
                raise JSONRPCError(INVALID_REQUEST, "Missing method")
                
            # Route to appropriate handler
            response = await self._route_message(method, message)
            
            # Add request ID to response
            if request_id is not None:
                response['id'] = request_id
                
            # Update metrics
            execution_time = asyncio.get_event_loop().time() - start_time
            await self._update_metrics(method, success=True, execution_time=execution_time)
            
            return response
            
        except JSONRPCError as e:
            # Update metrics
            execution_time = asyncio.get_event_loop().time() - start_time
            method = message.get('method', 'unknown')
            await self._update_metrics(method, success=False, execution_time=execution_time)
            
            # Return JSON-RPC error response
            error_response = {
                "jsonrpc": "2.0",
                "error": {
                    "code": e.code,
                    "message": e.message
                }
            }
            
            if hasattr(e, 'data') and e.data:
                error_response["error"]["data"] = e.data
                
            if message.get('id') is not None:
                error_response['id'] = message['id']
                
            return error_response
            
        except Exception as e:
            # Update metrics
            execution_time = asyncio.get_event_loop().time() - start_time
            method = message.get('method', 'unknown')
            await self._update_metrics(method, success=False, execution_time=execution_time)
            
            # Return internal error
            self._logger.error("Internal error handling message: %s", str(e))
            
            error_response = {
                "jsonrpc": "2.0",
                "error": {
                    "code": INTERNAL_ERROR,
                    "message": "Internal error",
                    "data": str(e)
                }
            }
            
            if message.get('id') is not None:
                error_response['id'] = message['id']
                
            return error_response
            
    def _validate_jsonrpc_message(self, message: Dict[str, Any]) -> None:
        """Validate JSON-RPC message structure.
        
        Args:
            message: Message to validate
            
        Raises:
            JSONRPCError: If message is invalid
        """
        if not isinstance(message, dict):
            raise JSONRPCError(INVALID_REQUEST, "Message must be an object")
            
        if message.get('jsonrpc') != '2.0':
            raise JSONRPCError(INVALID_REQUEST, "Invalid JSON-RPC version")
            
        if 'method' not in message:
            raise JSONRPCError(INVALID_REQUEST, "Missing method")
            
        if not isinstance(message['method'], str):
            raise JSONRPCError(INVALID_REQUEST, "Method must be a string")
            
    async def _route_message(self, method: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Route message to appropriate handler.
        
        Args:
            method: Message method
            message: Full message
            
        Returns:
            Response message
            
        Raises:
            JSONRPCError: If method not found or execution fails
        """
        # Standard MCP methods
        if method == 'initialize':
            return await self.handle_initialize(message)
        elif method == 'tools/list':
            return await self.handle_list_tools(message)
        elif method == 'tools/call':
            return await self.handle_call_tool(message)
        elif method == 'resources/list':
            return await self.handle_list_resources(message)
        elif method == 'resources/read':
            return await self.handle_read_resource(message)
        elif method == 'ping':
            return await self.handle_ping(message)
        else:
            raise JSONRPCError(METHOD_NOT_FOUND, f"Unknown method: {method}")
            
    async def handle_initialize(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle client initialization with capability negotiation.
        
        Args:
            message: Initialize message
            
        Returns:
            Initialize result
        """
        params = message.get('params', {})
        
        # Extract client info
        client_info = params.get('clientInfo', {})
        self._logger.info("Client connecting: %s v%s", 
                          client_info.get('name', 'unknown'),
                          client_info.get('version', 'unknown'))
        
        # Get server capabilities
        capabilities = self.server.capabilities
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": capabilities.__dict__ if hasattr(capabilities, '__dict__') else {},
                "serverInfo": {
                    "name": self.server.name,
                    "version": self.server.version
                }
            }
        }
        
    async def handle_list_tools(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """List available tools with dynamic discovery.
        
        Args:
            message: List tools message
            
        Returns:
            List tools result
        """
        try:
            tools = await self.server.get_tools()
            
            tool_list = []
            for tool in tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {
                        "type": "object",
                        "properties": tool.parameters,
                        "required": [
                            name for name, param in tool.parameters.items()
                            if param.get("required", False)
                        ]
                    }
                }
                tool_list.append(tool_info)
                
            return {
                "jsonrpc": "2.0",
                "result": {
                    "tools": tool_list
                }
            }
            
        except Exception as e:
            raise JSONRPCError(INTERNAL_ERROR, f"Failed to list tools: {e}")
            
    async def handle_call_tool(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool with swarm coordination.
        
        Args:
            message: Call tool message
            
        Returns:
            Call tool result
        """
        params = message.get('params', {})
        
        if 'name' not in params:
            raise JSONRPCError(INVALID_PARAMS, "Missing tool name")
            
        tool_name = params['name']
        arguments = params.get('arguments', {})
        
        try:
            result = await self.server.call_tool(tool_name, arguments)
            
            # Format result as content
            content = []
            if result is not None:
                if isinstance(result, str):
                    content.append({
                        "type": "text",
                        "text": result
                    })
                elif isinstance(result, dict):
                    content.append({
                        "type": "text", 
                        "text": json.dumps(result, indent=2)
                    })
                else:
                    content.append({
                        "type": "text",
                        "text": str(result)
                    })
            else:
                content.append({
                    "type": "text",
                    "text": "Tool executed successfully"
                })
                
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": content
                }
            }
            
        except ValueError as e:
            raise JSONRPCError(INVALID_PARAMS, str(e))
        except Exception as e:
            raise JSONRPCError(INTERNAL_ERROR, f"Tool execution failed: {e}")
            
    async def handle_list_resources(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """List available resources.
        
        Args:
            message: List resources message
            
        Returns:
            List resources result
        """
        try:
            resources = await self.server.list_resources()
            
            resource_list = []
            for resource in resources:
                resource_info = {
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mimeType": resource.mimeType
                }
                resource_list.append(resource_info)
                
            return {
                "jsonrpc": "2.0",
                "result": {
                    "resources": resource_list
                }
            }
            
        except Exception as e:
            raise JSONRPCError(INTERNAL_ERROR, f"Failed to list resources: {e}")
            
    async def handle_read_resource(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Read resource content.
        
        Args:
            message: Read resource message
            
        Returns:
            Read resource result
        """
        params = message.get('params', {})
        
        if 'uri' not in params:
            raise JSONRPCError(INVALID_PARAMS, "Missing resource URI")
            
        uri = params['uri']
        
        try:
            resource = await self.server.get_resource(uri)
            if not resource:
                raise JSONRPCError(INVALID_PARAMS, f"Resource not found: {uri}")
                
            # Convert resource to content format
            contents = []
            
            if hasattr(resource, 'type'):
                if resource.type == 'text':
                    contents.append({
                        "uri": uri,
                        "mimeType": "text/plain",
                        "text": getattr(resource, 'text', str(resource))
                    })
                elif resource.type == 'image':
                    contents.append({
                        "uri": uri,
                        "mimeType": getattr(resource, 'mimeType', 'image/png'),
                        "blob": getattr(resource, 'data', '')
                    })
                elif resource.type == 'audio':
                    contents.append({
                        "uri": uri,
                        "mimeType": getattr(resource, 'mimeType', 'audio/wav'),
                        "blob": getattr(resource, 'data', '')
                    })
                else:
                    contents.append({
                        "uri": uri,
                        "mimeType": "text/plain",
                        "text": str(resource)
                    })
            else:
                contents.append({
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": str(resource)
                })
                
            return {
                "jsonrpc": "2.0",
                "result": {
                    "contents": contents
                }
            }
            
        except ValueError as e:
            raise JSONRPCError(INVALID_PARAMS, str(e))
        except Exception as e:
            raise JSONRPCError(INTERNAL_ERROR, f"Failed to read resource: {e}")
            
    async def handle_ping(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping request.
        
        Args:
            message: Ping message
            
        Returns:
            Ping response
        """
        return {
            "jsonrpc": "2.0",
            "result": {}
        }
        
    async def _update_metrics(self, method: str, success: bool, execution_time: float) -> None:
        """Update message handling metrics.
        
        Args:
            method: Method name
            success: Whether request was successful
            execution_time: Time taken to process request
        """
        self._metrics.total_requests += 1
        
        if success:
            self._metrics.successful_requests += 1
        else:
            self._metrics.failed_requests += 1
            
        # Update method counts
        if self._metrics.methods_called is None:
            self._metrics.methods_called = {}
        if method not in self._metrics.methods_called:
            self._metrics.methods_called[method] = 0
        self._metrics.methods_called[method] += 1
        
        # Update average response time (exponential moving average)
        alpha = 0.1
        self._metrics.average_response_time = (
            self._metrics.average_response_time * (1 - alpha) + 
            execution_time * alpha
        )
        
    async def get_metrics(self) -> Dict[str, Any]:
        """Get message handling metrics.
        
        Returns:
            Dictionary of metrics
        """
        success_rate = 0.0
        if self._metrics.total_requests > 0:
            success_rate = (self._metrics.successful_requests / 
                           self._metrics.total_requests * 100)
            
        return {
            "total_requests": self._metrics.total_requests,
            "successful_requests": self._metrics.successful_requests,
            "failed_requests": self._metrics.failed_requests,
            "success_rate": success_rate,
            "average_response_time": self._metrics.average_response_time,
            "methods_called": dict(self._metrics.methods_called) if self._metrics.methods_called else {},
            "timestamp": datetime.now().isoformat()
        }
        
    async def reset_metrics(self) -> None:
        """Reset message handling metrics."""
        self._metrics = MessageMetrics()
        self._logger.info("Message handling metrics reset")


# Utility functions for message validation
def validate_tool_call_params(params: Dict[str, Any]) -> None:
    """Validate tool call parameters.
    
    Args:
        params: Parameters to validate
        
    Raises:
        JSONRPCError: If validation fails
    """
    if 'name' not in params:
        raise JSONRPCError(INVALID_PARAMS, "Missing 'name' parameter")
        
    if not isinstance(params['name'], str):
        raise JSONRPCError(INVALID_PARAMS, "'name' must be a string")
        
    if 'arguments' in params and not isinstance(params['arguments'], dict):
        raise JSONRPCError(INVALID_PARAMS, "'arguments' must be an object")


def validate_resource_read_params(params: Dict[str, Any]) -> None:
    """Validate resource read parameters.
    
    Args:
        params: Parameters to validate
        
    Raises:
        JSONRPCError: If validation fails
    """
    if 'uri' not in params:
        raise JSONRPCError(INVALID_PARAMS, "Missing 'uri' parameter")
        
    if not isinstance(params['uri'], str):
        raise JSONRPCError(INVALID_PARAMS, "'uri' must be a string")


def create_error_response(request_id: Optional[Any], error_code: int, error_message: str, error_data: Any = None) -> Dict[str, Any]:
    """Create a JSON-RPC error response.
    
    Args:
        request_id: Request ID
        error_code: Error code
        error_message: Error message
        error_data: Optional error data
        
    Returns:
        Error response dictionary
    """
    response = {
        "jsonrpc": "2.0",
        "error": {
            "code": error_code,
            "message": error_message
        }
    }
    
    if error_data is not None:
        response["error"]["data"] = error_data
        
    if request_id is not None:
        response["id"] = request_id
        
    return response


def create_success_response(request_id: Optional[Any], result: Any) -> Dict[str, Any]:
    """Create a JSON-RPC success response.
    
    Args:
        request_id: Request ID
        result: Result data
        
    Returns:
        Success response dictionary
    """
    response = {
        "jsonrpc": "2.0",
        "result": result
    }
    
    if request_id is not None:
        response["id"] = request_id
        
    return response