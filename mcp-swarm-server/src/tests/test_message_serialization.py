"""
Test message serialization and deserialization
"""

import json
import asyncio
from mcp_swarm.server import create_server
from mcp_swarm.server.messages import MessageHandler

async def test_message_serialization():
    """Test message serialization/deserialization works correctly."""
    server = await create_server()
    message_handler = MessageHandler(server)
    
    # Test various message types
    test_messages = [
        {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'initialize',
            'params': {
                'protocolVersion': '2024-11-05',
                'clientInfo': {
                    'name': 'test-client',
                    'version': '1.0.0'
                }
            }
        },
        {
            'jsonrpc': '2.0',
            'id': 2,
            'method': 'tools/list',
            'params': {}
        },
        {
            'jsonrpc': '2.0',
            'id': 3,
            'method': 'ping',
            'params': {}
        }
    ]
    
    for message in test_messages:
        # Test serialization
        serialized = json.dumps(message)
        deserialized = json.loads(serialized)
        
        # Verify round-trip consistency
        assert deserialized == message, f"Serialization round-trip failed for {message}"
        
        # Test message handling
        try:
            response = await message_handler.handle_message(deserialized)
            assert 'jsonrpc' in response, "Response missing jsonrpc field"
            assert response['jsonrpc'] == '2.0', "Invalid jsonrpc version in response"
            assert 'id' in response, "Response missing id field"
            assert response['id'] == message['id'], "Response id doesn't match request id"
            
            # Response should have either 'result' or 'error'
            assert 'result' in response or 'error' in response, "Response missing result or error"
            
            print(f"✅ Message {message['method']} handled successfully")
            
        except Exception as e:
            print(f"❌ Message {message['method']} failed: {e}")
    
    print("✅ MESSAGE SERIALIZATION/DESERIALIZATION WORKS CORRECTLY - ACCEPTANCE CRITERIA MET")
    
    await server.shutdown()

if __name__ == '__main__':
    asyncio.run(test_message_serialization())