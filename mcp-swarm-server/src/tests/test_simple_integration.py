"""
Simple integration tests for MCP server functionality.
"""

import asyncio
from mcp_swarm.server import create_server

def test_basic_server_creation():
    """Test that we can create a server instance."""
    server = None
    async def create_test_server():
        nonlocal server
        server = await create_server()
        assert server is not None
        assert server.name == "swarm-intelligence-server"
        await server.shutdown()
    
    asyncio.run(create_test_server())

def test_tool_registration():
    """Test tool registration functionality."""
    async def test_tools():
        server = await create_server()
        
        # Register a test tool
        test_tool = {
            'name': 'test_echo',
            'description': 'Echo input',
            'parameters': {'type': 'object'},
            'handler': lambda message: f"Echo: {message}"
        }
        
        await server.register_tool(test_tool)
        
        # Verify tool is registered
        tools = await server.get_tools()
        tool_names = [tool.name for tool in tools]
        assert 'test_echo' in tool_names
        
        # Test tool execution
        result = await server.call_tool('test_echo', {'message': 'Hello'})
        assert result == 'Echo: Hello'
        
        await server.shutdown()
    
    asyncio.run(test_tools())

def test_resource_registration():
    """Test resource registration functionality."""
    async def test_resources():
        server = await create_server()
        
        # Register a test resource
        test_resource = {
            'uri': 'test://sample',
            'name': 'Sample Resource',
            'description': 'A test resource',
            'content': 'Sample content',
            'mimeType': 'text/plain'
        }
        
        await server.register_resource(test_resource)
        
        # Verify resource is registered
        resources = await server.list_resources()
        resource_uris = [resource.uri for resource in resources]
        assert 'test://sample' in resource_uris
        
        # Test resource retrieval
        resource = await server.get_resource('test://sample')
        assert resource is not None
        
        await server.shutdown()
    
    asyncio.run(test_resources())

def test_concurrent_tool_calls():
    """Test concurrent tool execution."""
    async def test_concurrency():
        server = await create_server()
        
        # Register a test tool
        await server.register_tool({
            'name': 'concurrent_test',
            'description': 'Test concurrent execution',
            'parameters': {'type': 'object'},
            'handler': lambda x: f"Result {x}"
        })
        
        # Execute 50 concurrent calls
        tasks = []
        for i in range(50):
            task = server.call_tool('concurrent_test', {'x': i})
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Verify all calls succeeded
        assert len(results) == 50
        for i, result in enumerate(results):
            assert result == f"Result {i}"
        
        await server.shutdown()
    
    asyncio.run(test_concurrency())

if __name__ == '__main__':
    test_basic_server_creation()
    test_tool_registration()
    test_resource_registration() 
    test_concurrent_tool_calls()
    print("All tests passed!")