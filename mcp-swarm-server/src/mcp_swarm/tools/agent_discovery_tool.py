"""
Agent Discovery MCP Tool for MCP Swarm Intelligence Server

This module provides MCP tool interfaces for agent discovery, configuration 
management, and capability analysis.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

# MCP availability check
try:
    import mcp  # noqa: F401
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from ..agents.config_scanner import AgentConfigScanner
from ..agents.capability_matrix import CapabilityMatrixGenerator
from ..agents.config_reloader import DynamicConfigurationReloader, AgentRegistry

logger = logging.getLogger(__name__)


class AgentDiscoveryTool:
    """
    MCP tool implementation for agent discovery and configuration management.
    
    Provides comprehensive agent discovery capabilities through MCP interface.
    """
    
    def __init__(self, config_directory: Optional[Path] = None):
        if config_directory is None:
            config_directory = Path("agent-config")
        
        self.config_directory = config_directory
        self.scanner = AgentConfigScanner(config_directory)
        self.matrix_generator = CapabilityMatrixGenerator()
        self.agent_registry = AgentRegistry()
        self.reloader = DynamicConfigurationReloader(self.agent_registry, config_directory)
        
        # Cache for performance
        self._discovery_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 300  # 5 minutes
        
    def _is_cache_valid(self) -> bool:
        """Check if discovery cache is still valid"""
        if not self._cache_timestamp or not self._discovery_cache:
            return False
        
        cache_age = datetime.utcnow() - self._cache_timestamp
        return cache_age.total_seconds() < self._cache_ttl_seconds
    
    def _invalidate_cache(self):
        """Invalidate discovery cache"""
        self._discovery_cache = None
        self._cache_timestamp = None
    
    async def discover_agents(
        self,
        rescan: bool = False,
        validate_all: bool = True,
        include_metadata: bool = True,
        include_capability_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Discover and analyze agent configurations.
        
        Args:
            rescan: Force rescan of configuration directory
            validate_all: Validate all discovered configurations
            include_metadata: Include detailed metadata in results
            include_capability_analysis: Include capability matrix analysis
            
        Returns:
            Complete agent discovery results with analysis
        """
        try:
            # Check cache first
            if not rescan and self._is_cache_valid() and self._discovery_cache:
                logger.debug("Returning cached discovery results")
                return self._discovery_cache
            
            logger.info("Starting agent discovery process")
            start_time = datetime.utcnow()
            
            # Scan for agent configurations
            discovered_agents = await self.scanner.scan_agent_configurations(
                force_rescan=rescan
            )
            
            logger.info("Discovered %d agent configurations", len(discovered_agents))
            
            # Validate configurations if requested
            validation_results = {}
            if validate_all:
                for agent_name, config in discovered_agents.items():
                    validation_results[agent_name] = self.scanner.validate_agent_configuration(config)
            
            # Generate capability matrix if requested
            capability_matrix = None
            if include_capability_analysis and discovered_agents:
                capability_matrix = await self.matrix_generator.generate_capability_matrix(
                    discovered_agents
                )
            
            # Calculate discovery statistics
            discovery_stats = await self.scanner.get_discovery_statistics()
            
            # Build result
            result = {
                "discovery_summary": {
                    "total_agents": len(discovered_agents),
                    "valid_agents": sum(1 for v in validation_results.values() if v.is_valid),
                    "invalid_agents": sum(1 for v in validation_results.values() if not v.is_valid),
                    "scan_duration_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000),
                    "discovery_timestamp": start_time.isoformat(),
                    "config_directory": str(self.config_directory)
                },
                "agent_list": list(discovered_agents.keys()),
                "discovery_statistics": discovery_stats
            }
            
            # Add validation results
            if validation_results:
                result["validation_summary"] = {
                    "total_validated": len(validation_results),
                    "validation_passed": sum(1 for v in validation_results.values() if v.is_valid),
                    "total_errors": sum(len(v.errors) for v in validation_results.values()),
                    "total_warnings": sum(len(v.warnings) for v in validation_results.values())
                }
                
                if include_metadata:
                    result["validation_details"] = {
                        name: result.to_dict() for name, result in validation_results.items()
                    }
            
            # Add capability analysis
            if capability_matrix:
                result["capability_analysis"] = {
                    "coverage_score": capability_matrix.coverage_score,
                    "redundancy_score": capability_matrix.redundancy_score,
                    "coordination_efficiency": capability_matrix.coordination_efficiency,
                    "capability_gaps": len(capability_matrix.capability_gaps),
                    "capability_overlaps": len(capability_matrix.capability_overlaps),
                    "agent_clusters": len(capability_matrix.agent_clusters)
                }
                
                if include_metadata:
                    result["capability_matrix"] = capability_matrix.to_dict()
            
            # Add detailed agent metadata if requested
            if include_metadata:
                result["agent_details"] = {
                    name: config.to_dict() for name, config in discovered_agents.items()
                }
            
            # Cache the results
            self._discovery_cache = result
            self._cache_timestamp = datetime.utcnow()
            
            logger.info("Agent discovery completed successfully")
            return result
            
        except Exception as e:  # noqa: BLE001  # Broad exception catching for MCP tool error handling
            logger.error("Error during agent discovery: %s", e)
            return {
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
    
    async def get_agent_details(self, agent_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Detailed agent information
        """
        try:
            # Get latest agent configurations
            discovered_agents = await self.scanner.scan_agent_configurations()
            
            if agent_name not in discovered_agents:
                return {
                    "error": {
                        "message": f"Agent '{agent_name}' not found",
                        "available_agents": list(discovered_agents.keys())
                    }
                }
            
            config = discovered_agents[agent_name]
            validation = self.scanner.validate_agent_configuration(config)
            
            # Get intersecting agents
            intersecting_agents = self.scanner.get_agent_intersections(agent_name)
            
            # Generate capability matrix for analysis
            capability_matrix = await self.matrix_generator.generate_capability_matrix(
                discovered_agents
            )
            
            # Get recommendations
            recommendations = self.matrix_generator.get_capability_recommendations(
                capability_matrix, agent_name
            )
            
            result = {
                "agent_name": agent_name,
                "configuration": config.to_dict(),
                "validation": validation.to_dict(),
                "intersections": {
                    "direct_intersections": config.intersections,
                    "intersecting_agents": [agent.name for agent in intersecting_agents],
                    "intersection_count": len(intersecting_agents)
                },
                "capability_analysis": {
                    "total_capabilities": len(config.capabilities),
                    "capability_list": config.capabilities,
                    "domain": config.domain,
                    "priority": config.priority
                },
                "recommendations": recommendations.get("agent_specific", {}).get(agent_name, {}),
                "metadata": {
                    "file_path": str(config.file_path) if config.file_path else None,
                    "last_modified": config.last_modified.isoformat() if config.last_modified else None,
                    "memory_enabled": config.memory_enabled,
                    "coordination_style": config.coordination_style
                }
            }
            
            return result
            
        except Exception as e:  # noqa: BLE001  # Broad exception catching for MCP tool error handling
            logger.error("Error getting agent details for %s: %s", agent_name, e)
            return {
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                    "agent_name": agent_name
                }
            }
    
    async def analyze_capability_matrix(self) -> Dict[str, Any]:
        """
        Analyze the capability matrix and provide recommendations.
        
        Returns:
            Capability matrix analysis and recommendations
        """
        try:
            # Get current agents
            discovered_agents = await self.scanner.scan_agent_configurations()
            
            if not discovered_agents:
                return {
                    "error": {
                        "message": "No agents discovered for capability analysis"
                    }
                }
            
            # Generate capability matrix
            capability_matrix = await self.matrix_generator.generate_capability_matrix(
                discovered_agents
            )
            
            # Get recommendations
            recommendations = self.matrix_generator.get_capability_recommendations(
                capability_matrix
            )
            
            result = {
                "matrix_summary": {
                    "total_agents": len(capability_matrix.agents),
                    "total_capabilities": len(capability_matrix.capability_map),
                    "coverage_score": capability_matrix.coverage_score,
                    "redundancy_score": capability_matrix.redundancy_score,
                    "coordination_efficiency": capability_matrix.coordination_efficiency,
                    "generation_timestamp": capability_matrix.generation_timestamp.isoformat() if capability_matrix.generation_timestamp else datetime.utcnow().isoformat()
                },
                "capability_gaps": {
                    "total_gaps": len(capability_matrix.capability_gaps),
                    "critical_gaps": len([g for g in capability_matrix.capability_gaps if g.severity == 'critical']),
                    "high_priority_gaps": len([g for g in capability_matrix.capability_gaps if g.severity == 'high']),
                    "gaps_detail": [gap.to_dict() for gap in capability_matrix.capability_gaps]
                },
                "capability_overlaps": {
                    "total_overlaps": len(capability_matrix.capability_overlaps),
                    "excessive_overlaps": len([o for o in capability_matrix.capability_overlaps if o.redundancy_level == 'excessive']),
                    "overlaps_detail": [overlap.to_dict() for overlap in capability_matrix.capability_overlaps]
                },
                "agent_clusters": {
                    "total_clusters": len(capability_matrix.agent_clusters),
                    "clusters_detail": [cluster.to_dict() for cluster in capability_matrix.agent_clusters]
                },
                "domain_distribution": capability_matrix.domain_map,
                "recommendations": recommendations
            }
            
            return result
            
        except Exception as e:  # noqa: BLE001  # Broad exception catching for MCP tool error handling
            logger.error("Error analyzing capability matrix: %s", e)
            return {
                "error": {
                    "message": str(e),
                    "type": type(e).__name__
                }
            }
    
    async def reload_configurations(
        self, 
        agent_name: Optional[str] = None,
        force_reload: bool = False
    ) -> Dict[str, Any]:
        """
        Reload agent configurations.
        
        Args:
            agent_name: Specific agent to reload (None for all)
            force_reload: Force reload even if files haven't changed
            
        Returns:
            Reload results
        """
        try:
            self._invalidate_cache()  # Clear cache after reload
            
            if agent_name:
                # Reload specific agent
                result = await self.reloader.reload_agent_configuration(agent_name)
                return {
                    "reload_type": "single_agent",
                    "agent_name": agent_name,
                    "result": result.to_dict(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                # Reload all agents
                results = await self.reloader.reload_all_configurations(force=force_reload)
                
                # Summarize results
                total_agents = len(results)
                successful_reloads = sum(1 for r in results.values() if r.success)
                total_changes = sum(len(r.changes_applied) for r in results.values())
                
                return {
                    "reload_type": "all_agents",
                    "summary": {
                        "total_agents": total_agents,
                        "successful_reloads": successful_reloads,
                        "failed_reloads": total_agents - successful_reloads,
                        "total_changes": total_changes
                    },
                    "results": {name: result.to_dict() for name, result in results.items()},
                    "timestamp": datetime.utcnow().isoformat()
                }
            
        except Exception as e:  # noqa: BLE001  # Broad exception catching for MCP tool error handling
            logger.error("Error reloading configurations: %s", e)
            return {
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                    "agent_name": agent_name
                }
            }
    
    async def get_discovery_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive discovery statistics.
        
        Returns:
            Discovery statistics and system status
        """
        try:
            # Get scanner statistics
            scanner_stats = await self.scanner.get_discovery_statistics()
            
            # Get reloader statistics
            reloader_stats = self.reloader.get_reload_statistics()
            
            # Get cache information
            cache_info = {
                "cache_enabled": True,
                "cache_valid": self._is_cache_valid(),
                "cache_timestamp": self._cache_timestamp.isoformat() if self._cache_timestamp else None,
                "cache_ttl_seconds": self._cache_ttl_seconds
            }
            
            # System status
            system_status = {
                "config_directory": str(self.config_directory),
                "config_directory_exists": self.config_directory.exists(),
                "scanner_ready": True,
                "matrix_generator_ready": True,
                "reloader_ready": True
            }
            
            result = {
                "scanner_statistics": scanner_stats,
                "reloader_statistics": reloader_stats,
                "cache_information": cache_info,
                "system_status": system_status,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:  # noqa: BLE001  # Broad exception catching for MCP tool error handling
            logger.error("Error getting discovery statistics: %s", e)
            return {
                "error": {
                    "message": str(e),
                    "type": type(e).__name__
                }
            }


# MCP Tool Interface Functions
async def agent_discovery_tool(
    rescan: bool = False,
    validate_all: bool = True,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    MCP tool for discovering and managing agent configurations.
    
    Args:
        rescan: Force rescan of configuration directory
        validate_all: Validate all discovered configurations
        include_metadata: Include detailed metadata in results
        
    Returns:
        Complete agent discovery results with capability analysis
    """
    discovery_tool = AgentDiscoveryTool()
    return await discovery_tool.discover_agents(
        rescan=rescan,
        validate_all=validate_all,
        include_metadata=include_metadata,
        include_capability_analysis=True
    )


async def agent_details_tool(agent_name: str) -> Dict[str, Any]:
    """
    MCP tool for getting detailed information about a specific agent.
    
    Args:
        agent_name: Name of the agent to analyze
        
    Returns:
        Detailed agent information and analysis
    """
    discovery_tool = AgentDiscoveryTool()
    return await discovery_tool.get_agent_details(agent_name)


async def capability_matrix_tool() -> Dict[str, Any]:
    """
    MCP tool for analyzing the capability matrix.
    
    Returns:
        Capability matrix analysis and recommendations
    """
    discovery_tool = AgentDiscoveryTool()
    return await discovery_tool.analyze_capability_matrix()


async def configuration_reload_tool(
    agent_name: Optional[str] = None,
    force_reload: bool = False
) -> Dict[str, Any]:
    """
    MCP tool for reloading agent configurations.
    
    Args:
        agent_name: Specific agent to reload (None for all)
        force_reload: Force reload even if files haven't changed
        
    Returns:
        Reload results
    """
    discovery_tool = AgentDiscoveryTool()
    return await discovery_tool.reload_configurations(agent_name, force_reload)


async def discovery_statistics_tool() -> Dict[str, Any]:
    """
    MCP tool for getting discovery statistics.
    
    Returns:
        Discovery statistics and system status
    """
    discovery_tool = AgentDiscoveryTool()
    return await discovery_tool.get_discovery_statistics()


# Tool Definitions for MCP Server Registration
AGENT_DISCOVERY_TOOL_DEFINITIONS = {
    "agent_discovery": {
        "name": "agent_discovery",
        "description": "Discover and analyze agent configurations with capability matrix",
        "function": agent_discovery_tool,
        "parameters": {
            "rescan": {"type": "boolean", "default": False, "description": "Force rescan of configuration directory"},
            "validate_all": {"type": "boolean", "default": True, "description": "Validate all discovered configurations"},
            "include_metadata": {"type": "boolean", "default": True, "description": "Include detailed metadata in results"}
        }
    },
    "agent_details": {
        "name": "agent_details", 
        "description": "Get detailed information about a specific agent",
        "function": agent_details_tool,
        "parameters": {
            "agent_name": {"type": "string", "required": True, "description": "Name of the agent to analyze"}
        }
    },
    "capability_matrix": {
        "name": "capability_matrix",
        "description": "Analyze the capability matrix and provide recommendations", 
        "function": capability_matrix_tool,
        "parameters": {}
    },
    "configuration_reload": {
        "name": "configuration_reload",
        "description": "Reload agent configurations",
        "function": configuration_reload_tool,
        "parameters": {
            "agent_name": {"type": "string", "description": "Specific agent to reload (optional)"},
            "force_reload": {"type": "boolean", "default": False, "description": "Force reload even if files haven't changed"}
        }
    },
    "discovery_statistics": {
        "name": "discovery_statistics",
        "description": "Get discovery statistics and system status",
        "function": discovery_statistics_tool,
        "parameters": {}
    }
}


async def main():
    """Example usage of Agent Discovery Tools"""
    
    # Initialize discovery tool
    discovery_tool = AgentDiscoveryTool(Path("../../agent-config"))
    
    print("🔍 Running Agent Discovery...")
    
    # Discover agents
    discovery_result = await discovery_tool.discover_agents()
    print("📊 Discovery Summary:")
    print(f"  Total Agents: {discovery_result['discovery_summary']['total_agents']}")
    print(f"  Valid Agents: {discovery_result['discovery_summary']['valid_agents']}")
    print(f"  Coverage Score: {discovery_result.get('capability_analysis', {}).get('coverage_score', 'N/A')}")
    
    # Analyze capability matrix
    matrix_result = await discovery_tool.analyze_capability_matrix()
    print("\n🧠 Capability Matrix Analysis:")
    print(f"  Coverage Score: {matrix_result['matrix_summary']['coverage_score']}")
    print(f"  Capability Gaps: {matrix_result['capability_gaps']['total_gaps']}")
    print(f"  Agent Clusters: {matrix_result['agent_clusters']['total_clusters']}")
    
    # Get statistics
    stats = await discovery_tool.get_discovery_statistics()
    print("\n📈 System Statistics:")
    print(f"  Config Directory: {stats['system_status']['config_directory']}")
    print(f"  Cache Valid: {stats['cache_information']['cache_valid']}")


if __name__ == "__main__":
    asyncio.run(main())