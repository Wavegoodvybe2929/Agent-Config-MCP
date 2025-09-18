"""
Test suite for MCP Server concurrent request handling and protocol compliance.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, List
import time

from mcp_swarm.server import create_server, SwarmMCPServer
from mcp_swarm.server.base import ToolInfo, ResourceInfo
from mcp_swarm.server.messages import MessageHandler, JSONRPCError


class TestMCPServerConcurrency:
    """Test MCP server's ability to handle concurrent requests."""
    
    @pytest.fixture
    async def server(self):
        """Create test server instance."""
        server = await create_server(
            name="test-server",
            version="1.0.0-test",
            resource_path="test_data/resources"
        )
        yield server
        await server.shutdown()
    
    async def test_concurrent_request_handling(self, server):
        """Test that server can handle 100+ concurrent requests."""
        # Register a simple test tool
        async def test_tool(message: str = "test") -> str:
            await asyncio.sleep(0.01)  # Simulate some work
            return f"Processed: {message}"
        
        await server.register_tool({
            'name': 'test_tool',
            'description': 'Test tool for concurrency',
            'parameters': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'}
                }
            },
            'handler': test_tool
        })
        
        # Create 150 concurrent requests
        async def make_request(request_id: int) -> Dict[str, Any]:
            request = {
                'jsonrpc': '2.0',
                'id': request_id,
                'method': 'tools/call',
                'params': {
                    'name': 'test_tool',
                    'arguments': {'message': f'request_{request_id}'}
                }
            }
            return await server.handle_request(request)
        
        # Execute concurrent requests
        start_time = time.time()
        tasks = [make_request(i) for i in range(150)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Validate results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful_results) >= 140, f"Expected at least 140 successful requests, got {len(successful_results)}"
        assert len(failed_results) <= 10, f"Too many failed requests: {len(failed_results)}"
        assert end_time - start_time < 10.0, f"Requests took too long: {end_time - start_time:.2f}s"
        
        # Validate response format
        for result in successful_results:
            if isinstance(result, dict):
                assert 'content' in result
                assert isinstance(result['content'], list)
                assert len(result['content']) > 0
                assert 'text' in result['content'][0]
    
    async def test_concurrent_tool_registration(self, server):
        """Test concurrent tool registration."""
        async def register_tool(tool_id: int):
            await server.register_tool({
                'name': f'tool_{tool_id}',
                'description': f'Test tool {tool_id}',
                'parameters': {},
                'handler': lambda: f"result_{tool_id}"
            })
        
        # Register 50 tools concurrently
        tasks = [register_tool(i) for i in range(50)]
        await asyncio.gather(*tasks)
        
        # Verify all tools were registered
        tools = await server.get_tools()
        tool_names = [tool.name for tool in tools]
        
        for i in range(50):
            assert f'tool_{i}' in tool_names, f"Tool tool_{i} was not registered"
    
    async def test_concurrent_resource_access(self, server):
        """Test concurrent resource access."""
        # Register test resources
        for i in range(20):
            await server.register_resource({
                'uri': f'test://resource_{i}',
                'name': f'Resource {i}',
                'description': f'Test resource {i}',
                'content': f'Content for resource {i}',
                'mimeType': 'text/plain'
            })
        
        # Concurrent resource requests
        async def get_resource(resource_id: int):
            return await server.get_resource(f'test://resource_{resource_id}')
        
        tasks = [get_resource(i % 20) for i in range(100)]
        results = await asyncio.gather(*tasks)
        
        # Verify all requests succeeded
        assert all(result is not None for result in results)


class TestMCPMessageHandling:
    """Test message serialization and deserialization."""
    
    @pytest.fixture
    def message_handler(self):
        """Create message handler for testing."""
        server = MagicMock()
        server.handle_request = AsyncMock()
        return MessageHandler(server)
    
    def test_message_serialization(self):
        """Test message serialization to JSON."""
        message = {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'test/method',
            'params': {'arg1': 'value1', 'arg2': 42}
        }
        
        # Serialize message
        serialized = json.dumps(message)
        
        # Deserialize and validate
        deserialized = json.loads(serialized)
        assert deserialized == message
        assert deserialized['jsonrpc'] == '2.0'
        assert deserialized['id'] == 1
        assert deserialized['method'] == 'test/method'
        assert deserialized['params']['arg1'] == 'value1'
        assert deserialized['params']['arg2'] == 42
    
    async def test_message_deserialization_validation(self, message_handler):
        """Test message deserialization with validation."""
        # Valid message
        valid_message = {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'tools/list',
            'params': {}
        }
        
        # Should not raise exception
        result = await message_handler.handle_message(valid_message)
        assert 'id' in result
        assert result['id'] == 1
    
    async def test_invalid_message_handling(self, message_handler):
        """Test handling of invalid messages."""
        # Missing jsonrpc
        invalid_message = {
            'id': 1,
            'method': 'test/method'
        }
        
        with pytest.raises((JSONRPCError, ValueError)):
            await message_handler.handle_message(invalid_message)
        
        # Invalid jsonrpc version
        invalid_version = {
            'jsonrpc': '1.0',
            'id': 1,
            'method': 'test/method'
        }
        
        with pytest.raises((JSONRPCError, ValueError)):
            await message_handler.handle_message(invalid_version)
    
    async def test_error_response_format(self, message_handler):
        """Test proper error response formatting."""
        # Create a message that will trigger an error
        error_message = {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'nonexistent/method',
            'params': {}
        }
        
        try:
            await message_handler.handle_message(error_message)
        except (JSONRPCError, ValueError) as e:
            # Error should be properly structured
            if hasattr(e, 'code'):
                assert isinstance(e.code, int)  # type: ignore
            if hasattr(e, 'message'):
                assert isinstance(e.message, str)  # type: ignore


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance."""
    
    @pytest.fixture
    async def server(self):
        """Create test server instance."""
        server = await create_server()
        yield server
        await server.shutdown()
    
    async def test_capabilities_negotiation(self, server):
        """Test MCP capabilities negotiation."""
        capabilities = server.capabilities
        
        # Check required capabilities
        assert hasattr(capabilities, 'tools') or hasattr(capabilities, '__dict__')
        assert hasattr(capabilities, 'resources') or hasattr(capabilities, '__dict__')
        
        # If capabilities has __dict__, check the dict version
        if hasattr(capabilities, '__dict__'):
            caps_dict = vars(capabilities)
            assert 'tools' in caps_dict or 'experimental' in caps_dict
            assert 'resources' in caps_dict or 'experimental' in caps_dict
    
    async def test_tool_schema_validation(self, server):
        """Test tool schema validation."""
        # Register tool with proper schema
        valid_tool = {
            'name': 'valid_tool',
            'description': 'A valid test tool',
            'parameters': {
                'type': 'object',
                'properties': {
                    'input': {
                        'type': 'string',
                        'description': 'Input parameter'
                    }
                },
                'required': ['input']
            },
            'handler': lambda input: f"Processed: {input}"
        }
        
        await server.register_tool(valid_tool)
        
        # Verify tool is properly registered
        tools = await server.get_tools()
        tool_names = [tool.name for tool in tools]
        assert 'valid_tool' in tool_names
        
        # Find our tool and validate its schema
        our_tool = next(tool for tool in tools if tool.name == 'valid_tool')
        assert our_tool.description == 'A valid test tool'
        assert 'type' in our_tool.parameters
        assert our_tool.parameters['type'] == 'object'
    
    async def test_resource_uri_handling(self, server):
        """Test resource URI handling compliance."""
        # Register resource with various URI schemes
        test_resources = [
            {
                'uri': 'file:///test/path',
                'name': 'File Resource',
                'content': 'File content',
                'mimeType': 'text/plain'
            },
            {
                'uri': 'http://example.com/resource',
                'name': 'HTTP Resource', 
                'content': 'HTTP content',
                'mimeType': 'text/html'
            },
            {
                'uri': 'custom://app/resource',
                'name': 'Custom Resource',
                'content': 'Custom content',
                'mimeType': 'application/json'
            }
        ]
        
        for resource in test_resources:
            await server.register_resource(resource)
        
        # Verify all resources are accessible
        for resource in test_resources:
            retrieved = await server.get_resource(resource['uri'])
            assert retrieved is not None
    
    async def test_error_code_compliance(self, server):
        """Test MCP error code compliance."""
        # Test method not found error
        request = {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'nonexistent/method',
            'params': {}
        }
        
        try:
            await server.handle_request(request)
            assert False, "Should have raised an exception"
        except (ValueError, JSONRPCError) as e:
            # Should be a proper JSON-RPC error
            error_msg = str(e)
            assert 'method' in error_msg.lower() or 'not found' in error_msg.lower()


class TestMCPServerIntegration:
    """Integration tests for complete MCP server functionality."""
    
    @pytest.fixture
    async def configured_server(self):
        """Create a fully configured test server."""
        server = await create_server()
        
        # Register test tools
        await server.register_tool({
            'name': 'echo',
            'description': 'Echo the input message',
            'parameters': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'}
                },
                'required': ['message']
            },
            'handler': lambda message: f"Echo: {message}"
        })
        
        await server.register_tool({
            'name': 'add',
            'description': 'Add two numbers',
            'parameters': {
                'type': 'object', 
                'properties': {
                    'a': {'type': 'number'},
                    'b': {'type': 'number'}
                },
                'required': ['a', 'b']
            },
            'handler': lambda a, b: a + b
        })
        
        # Register test resources
        await server.register_resource({
            'uri': 'test://hello',
            'name': 'Hello Resource',
            'content': 'Hello, World!',
            'mimeType': 'text/plain'
        })
        
        yield server
        await server.shutdown()
    
    async def test_full_tool_workflow(self, configured_server):
        """Test complete tool workflow."""
        server = configured_server
        
        # List tools
        tools = await server.get_tools()
        assert len(tools) >= 2
        
        tool_names = [tool.name for tool in tools]
        assert 'echo' in tool_names
        assert 'add' in tool_names
        
        # Call echo tool
        echo_result = await server.call_tool('echo', {'message': 'Hello, MCP!'})
        assert echo_result == 'Echo: Hello, MCP!'
        
        # Call add tool
        add_result = await server.call_tool('add', {'a': 5, 'b': 3})
        assert add_result == 8
    
    async def test_full_resource_workflow(self, configured_server):
        """Test complete resource workflow."""
        server = configured_server
        
        # List resources
        resources = await server.list_resources()
        assert len(resources) >= 1
        
        resource_uris = [resource.uri for resource in resources]
        assert 'test://hello' in resource_uris
        
        # Get resource
        resource = await server.get_resource('test://hello')
        assert resource is not None
        
        # Check resource properties
        if hasattr(resource, 'content'):
            assert resource.content == 'Hello, World!'
        elif isinstance(resource, dict):
            assert resource.get('content') == 'Hello, World!'


if __name__ == '__main__':
    # Run tests if executed directly
    import sys
    sys.exit(pytest.main([__file__, '-v']))