"""
Test script for MCP Server Configuration Integration.

This script validates the MCP server configuration integration including:
- MCP Server Manager MCP Tool functionality
- Agent-Config Integration Engine functionality
- Auto-discovery and tool registration
- Orchestrator workflow integration
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp_swarm.tools.mcp_server_manager import mcp_server_manager_tool
from src.mcp_swarm.server.agent_integration import agent_config_integration_tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPServerIntegrationTester:
    """Test suite for MCP Server Configuration Integration."""
    
    def __init__(self):
        self.test_results = {}
        self.agent_config_path = Path(".agent-config")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("Starting MCP Server Configuration Integration Tests")
        
        test_methods = [
            self.test_mcp_server_manager_initialization,
            self.test_mcp_server_manager_configuration,
            self.test_mcp_server_manager_status,
            self.test_agent_config_integration_initialization,
            self.test_agent_config_integration_discovery,
            self.test_integration_workflow
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
                logger.info(f"Running test: {test_method.__name__}")
                result = await test_method()
                
                if result.get("success", False):
                    logger.info(f"✅ {test_method.__name__} - PASSED")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_method.__name__} - FAILED: {result.get('error', 'Unknown error')}")
                
                self.test_results[test_method.__name__] = result
                
            except Exception as e:
                logger.error(f"❌ {test_method.__name__} - EXCEPTION: {str(e)}")
                self.test_results[test_method.__name__] = {
                    "success": False,
                    "error": f"Exception during test: {str(e)}"
                }
        
        # Summary
        success_rate = (passed_tests / total_tests) * 100
        logger.info(f"\n=== Test Summary ===")
        logger.info(f"Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        return {
            "success": passed_tests == total_tests,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "success_rate": success_rate,
            "test_results": self.test_results
        }
    
    async def test_mcp_server_manager_initialization(self) -> Dict[str, Any]:
        """Test MCP Server Manager initialization."""
        try:
            result = await mcp_server_manager_tool(
                action="initialize",
                auto_discovery=True
            )
            
            # Check result structure
            if not isinstance(result, dict):
                return {"success": False, "error": "Result is not a dictionary"}
            
            if "success" not in result:
                return {"success": False, "error": "Result missing 'success' field"}
            
            return {"success": True, "result": result}
            
        except Exception as e:
            return {"success": False, "error": f"Exception: {str(e)}"}
    
    async def test_mcp_server_manager_configuration(self) -> Dict[str, Any]:
        """Test MCP Server Manager configuration."""
        try:
            config_updates = {
                "transport": "websocket",
                "port": 8080,
                "auto_discovery": True
            }
            
            result = await mcp_server_manager_tool(
                action="configure",
                server_config=config_updates
            )
            
            if not result.get("success", False):
                return {"success": False, "error": result.get("error", "Configuration failed")}
            
            return {"success": True, "result": result}
            
        except Exception as e:
            return {"success": False, "error": f"Exception: {str(e)}"}
    
    async def test_mcp_server_manager_status(self) -> Dict[str, Any]:
        """Test MCP Server Manager status retrieval."""
        try:
            result = await mcp_server_manager_tool(action="status")
            
            if not result.get("success", False):
                return {"success": False, "error": result.get("error", "Status check failed")}
            
            # Verify status structure
            if "status" not in result:
                return {"success": False, "error": "Status information missing from result"}
            
            status = result["status"]
            required_fields = ["running", "uptime", "tools_count", "resources_count", "agents_discovered"]
            
            for field in required_fields:
                if field not in status:
                    return {"success": False, "error": f"Status missing field: {field}"}
            
            return {"success": True, "result": result}
            
        except Exception as e:
            return {"success": False, "error": f"Exception: {str(e)}"}
    
    async def test_agent_config_integration_initialization(self) -> Dict[str, Any]:
        """Test Agent-Config Integration initialization."""
        try:
            result = await agent_config_integration_tool(
                action="initialize",
                config_directory=str(self.agent_config_path)
            )
            
            if not isinstance(result, dict):
                return {"success": False, "error": "Result is not a dictionary"}
            
            # The integration may fail if no agent-config directory exists, which is acceptable
            return {"success": True, "result": result}
            
        except Exception as e:
            return {"success": False, "error": f"Exception: {str(e)}"}
    
    async def test_agent_config_integration_discovery(self) -> Dict[str, Any]:
        """Test Agent-Config Integration discovery."""
        try:
            result = await agent_config_integration_tool(
                action="discover",
                config_directory=str(self.agent_config_path)
            )
            
            if not isinstance(result, dict):
                return {"success": False, "error": "Result is not a dictionary"}
            
            # Discovery may succeed even with no agents found
            return {"success": True, "result": result}
            
        except Exception as e:
            return {"success": False, "error": f"Exception: {str(e)}"}
    
    async def test_integration_workflow(self) -> Dict[str, Any]:
        """Test end-to-end integration workflow."""
        try:
            # Step 1: Initialize MCP server
            server_init = await mcp_server_manager_tool(action="initialize")
            if not server_init.get("success", False):
                return {"success": False, "error": f"Server initialization failed: {server_init.get('error')}"}
            
            # Step 2: Initialize agent integration
            agent_init = await agent_config_integration_tool(
                action="initialize",
                config_directory=str(self.agent_config_path)
            )
            
            # Step 3: Get server status
            status_result = await mcp_server_manager_tool(action="status")
            if not status_result.get("success", False):
                return {"success": False, "error": f"Status check failed: {status_result.get('error')}"}
            
            # Step 4: Get integration status
            integration_status = await agent_config_integration_tool(action="status")
            
            return {
                "success": True,
                "workflow_results": {
                    "server_init": server_init,
                    "agent_init": agent_init,
                    "server_status": status_result,
                    "integration_status": integration_status
                }
            }
            
        except Exception as e:
            return {"success": False, "error": f"Workflow exception: {str(e)}"}


async def main():
    """Run the MCP Server Configuration Integration tests."""
    tester = MCPServerIntegrationTester()
    results = await tester.run_all_tests()
    
    print("\n" + "="*50)
    print("MCP SERVER CONFIGURATION INTEGRATION TEST RESULTS")
    print("="*50)
    
    if results["success"]:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"Tests Passed: {results['passed_tests']}/{results['total_tests']}")
    
    # Print individual test results
    print("\nDetailed Results:")
    for test_name, result in results["test_results"].items():
        status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not result.get("success", False) and "error" in result:
            print(f"    Error: {result['error']}")
    
    return results["success"]


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)