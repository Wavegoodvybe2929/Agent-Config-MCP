"""
Simple validation test for MCP Server Manager components.

This test validates the core functionality without complex imports.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_mcp_server_manager_import():
    """Test that we can import the MCP server manager tool."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from mcp_swarm.tools.mcp_server_manager import MCPServerConfig, MCPServerManager
        
        # Test basic configuration
        config = MCPServerConfig(
            name="Test Server",
            version="1.0.0",
            transport="stdio"
        )
        
        # Test manager creation
        manager = MCPServerManager(config)
        
        logger.info("✅ MCP Server Manager import and basic setup - PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ MCP Server Manager import test - FAILED: {str(e)}")
        return False


async def test_agent_config_integration_import():
    """Test that we can import the agent config integration."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from mcp_swarm.server.agent_integration import AgentConfigIntegration, AgentToolDefinition
        
        # Test basic tool definition
        tool_def = AgentToolDefinition(
            name="test_tool",
            description="Test tool description",
            agent_source="test_agent",
            capabilities=["testing"]
        )
        
        # Test integration creation
        integration = AgentConfigIntegration(
            config_directory=Path(".agent-config")
        )
        
        logger.info("✅ Agent Config Integration import and basic setup - PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Agent Config Integration import test - FAILED: {str(e)}")
        return False


async def test_tool_functionality():
    """Test basic tool functionality."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from mcp_swarm.tools.mcp_server_manager import mcp_server_manager_tool
        
        # Test status action (should work without complex dependencies)
        result = await mcp_server_manager_tool(action="status")
        
        # Should return a dictionary with success field
        if isinstance(result, dict) and "success" in result:
            logger.info("✅ MCP Server Manager tool basic functionality - PASSED")
            return True
        else:
            logger.error("❌ MCP Server Manager tool returned unexpected format")
            return False
            
    except Exception as e:
        logger.error(f"❌ MCP Server Manager tool functionality test - FAILED: {str(e)}")
        return False


async def main():
    """Run simple validation tests."""
    logger.info("Starting MCP Server Configuration Integration Validation")
    
    tests = [
        test_mcp_server_manager_import,
        test_agent_config_integration_import,
        test_tool_functionality
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed_tests += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} threw exception: {str(e)}")
    
    success_rate = (passed_tests / total_tests) * 100
    
    print("\n" + "="*50)
    print("MCP SERVER CONFIGURATION VALIDATION RESULTS")
    print("="*50)
    
    if passed_tests == total_tests:
        print("✅ ALL VALIDATION TESTS PASSED")
    else:
        print("❌ SOME VALIDATION TESTS FAILED")
    
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)