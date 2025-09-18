"""
Test to validate MCP server can handle 100+ concurrent requests
"""

import asyncio
import time
from mcp_swarm.server import create_server

async def test_concurrent_requests():
    """Test that server can handle 100+ concurrent requests."""
    server = await create_server()
    
    # Register a test tool that simulates some work
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
    async def make_request(request_id: int) -> str:
        try:
            result = await server.call_tool('test_tool', {'message': f'request_{request_id}'})
            return str(result)
        except Exception as e:
            return f"ERROR: {e}"
    
    # Execute concurrent requests
    print("Starting 150 concurrent requests...")
    start_time = time.time()
    tasks = [make_request(i) for i in range(150)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()
    
    # Analyze results
    successful_results = [r for r in results if not isinstance(r, Exception) and not str(r).startswith("ERROR")]
    failed_results = [r for r in results if isinstance(r, Exception) or str(r).startswith("ERROR")]
    
    print(f"Total requests: 150")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    print(f"Total time: {end_time - start_time:.2f}s")
    print(f"Requests per second: {150 / (end_time - start_time):.2f}")
    
    # Validate acceptance criteria
    assert len(successful_results) >= 140, f"Expected at least 140 successful requests, got {len(successful_results)}"
    assert len(failed_results) <= 10, f"Too many failed requests: {len(failed_results)}"
    assert end_time - start_time < 10.0, f"Requests took too long: {end_time - start_time:.2f}s"
    
    print("✅ SERVER CAN HANDLE 100+ CONCURRENT REQUESTS - ACCEPTANCE CRITERIA MET")
    
    await server.shutdown()

if __name__ == '__main__':
    asyncio.run(test_concurrent_requests())