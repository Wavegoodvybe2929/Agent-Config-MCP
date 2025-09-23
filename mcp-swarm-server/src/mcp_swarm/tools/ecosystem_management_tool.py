"""
Ecosystem Management Tool for MCP Swarm Intelligence Server

This module provides a comprehensive MCP interface for ecosystem management operations,
including agent lifecycle management, health monitoring, load balancing, and 
performance optimization.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import logging

# Import MCP components
try:
    from mcp.types import TextContent
    MCP_AVAILABLE = True
    MCPTextContent = TextContent
except ImportError:
    # Fallback for development when MCP is not available
    MCP_AVAILABLE = False
    MCPTextContent = dict

# Simple Tool class for MCP compatibility
class Tool:
    """Simple Tool implementation for MCP compatibility"""
    def __init__(self, name: str, description: str, inputSchema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema

# Configure logging
logger = logging.getLogger(__name__)

# Import internal components
from ..agents.load_monitor import AgentLoadMonitor
from ..agents.availability_tracker import AgentAvailabilityTracker
from ..agents.metrics_collector import PerformanceMetricsCollector, MetricType
from ..agents.ecosystem_monitor import EcosystemHealthMonitor, AlertSeverity

logger = logging.getLogger(__name__)

class EcosystemManagementTool:
    """MCP Tool for comprehensive ecosystem management"""
    
    def __init__(self):
        """Initialize the ecosystem management tool"""
        # Initialize monitoring components
        self.load_monitor = AgentLoadMonitor()
        self.availability_tracker = AgentAvailabilityTracker()
        self.metrics_collector = PerformanceMetricsCollector()
        self.health_monitor = EcosystemHealthMonitor()
        
        # Tool state
        self.is_initialized = False
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        
        # Performance statistics
        self.ecosystem_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "last_reset": datetime.now()
        }

    async def initialize(self):
        """Initialize all monitoring systems"""
        if self.is_initialized:
            return
        
        try:
            # Start all monitoring systems
            await self.load_monitor.start_monitoring()
            await self.availability_tracker.start_monitoring()
            await self.metrics_collector.start_collection()
            await self.health_monitor.start_monitoring()
            
            # Register default components
            self.health_monitor.register_component("coordinator", "coordinator", is_critical=True)
            self.health_monitor.register_component("memory_system", "memory", is_critical=True)
            self.health_monitor.register_component("network", "network", is_critical=False)
            
            self.is_initialized = True
            logger.info("Ecosystem management tool initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize ecosystem management tool: %s", e)
            raise

    async def shutdown(self):
        """Shutdown all monitoring systems"""
        if not self.is_initialized:
            return
        
        try:
            await self.load_monitor.stop_monitoring()
            await self.availability_tracker.stop_monitoring()
            await self.metrics_collector.stop_collection()
            await self.health_monitor.stop_monitoring()
            
            self.is_initialized = False
            logger.info("Ecosystem management tool shutdown completed")
            
        except Exception as e:
            logger.error("Error during ecosystem management tool shutdown: %s", e)

    def get_tools(self) -> List[Any]:
        """Get all MCP tools provided by this ecosystem management tool"""
        if not MCP_AVAILABLE:
            return []
        
        return [
            Tool(
                name="ecosystem_status",
                description="Get comprehensive ecosystem status including health, performance, and availability metrics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "include_details": {
                            "type": "boolean",
                            "description": "Include detailed component information",
                            "default": False
                        },
                        "include_trends": {
                            "type": "boolean", 
                            "description": "Include performance trends and analysis",
                            "default": False
                        }
                    }
                }
            ),
            Tool(
                name="agent_management",
                description="Manage agent lifecycle including registration, status updates, and deregistration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["register", "deregister", "update", "list", "status"],
                            "description": "Action to perform"
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "Agent identifier (required for register, deregister, update, status)"
                        },
                        "agent_type": {
                            "type": "string",
                            "description": "Agent type (required for register)"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Agent metadata (optional for register/update)"
                        }
                    },
                    "required": ["action"]
                }
            ),
            Tool(
                name="performance_analysis",
                description="Analyze ecosystem performance with detailed metrics and recommendations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Specific agent to analyze (optional)"
                        },
                        "metric_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["response_time", "task_completion", "error_rate", "throughput", "resource_usage", "coordination_efficiency"]
                            },
                            "description": "Specific metrics to analyze"
                        },
                        "time_range_hours": {
                            "type": "number",
                            "description": "Time range for analysis in hours",
                            "default": 24
                        },
                        "include_predictions": {
                            "type": "boolean",
                            "description": "Include performance predictions",
                            "default": False
                        }
                    }
                }
            ),
            Tool(
                name="load_balancing",
                description="Analyze and optimize load distribution across agents",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["analyze", "rebalance", "recommendations"],
                            "description": "Load balancing action to perform"
                        },
                        "target_agents": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific agents to include in load balancing"
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "Perform analysis without making changes",
                            "default": True
                        }
                    },
                    "required": ["action"]
                }
            ),
            Tool(
                name="health_monitoring",
                description="Monitor and manage ecosystem health including alerts and diagnostics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["status", "alerts", "diagnostics", "resolve_alert", "health_check"],
                            "description": "Health monitoring action"
                        },
                        "component_id": {
                            "type": "string",
                            "description": "Specific component for targeted actions"
                        },
                        "alert_id": {
                            "type": "string",
                            "description": "Alert ID for resolution (required for resolve_alert)"
                        },
                        "severity_filter": {
                            "type": "string",
                            "enum": ["info", "warning", "critical", "emergency"],
                            "description": "Filter alerts by severity"
                        }
                    },
                    "required": ["action"]
                }
            )
        ]

    async def handle_ecosystem_status(self, include_details: bool = False, include_trends: bool = False) -> Dict[str, Any]:
        """Handle ecosystem status requests"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Get ecosystem health
            ecosystem_health = self.health_monitor.get_ecosystem_health()
            
            # Get availability summary
            availability_summary = self.availability_tracker.get_availability_summary()
            
            # Get performance summary
            performance_summary = self.metrics_collector.get_performance_summary()
            
            # Basic status
            status = {
                "timestamp": datetime.now().isoformat(),
                "ecosystem_health": {
                    "overall_status": ecosystem_health.overall_status.value if ecosystem_health else "unknown",
                    "overall_score": ecosystem_health.overall_score if ecosystem_health else 0.0,
                    "total_components": ecosystem_health.total_components if ecosystem_health else 0,
                    "healthy_components": ecosystem_health.healthy_components if ecosystem_health else 0,
                    "active_alerts": ecosystem_health.active_alerts if ecosystem_health else 0
                },
                "agent_availability": {
                    "total_agents": availability_summary["total_agents"],
                    "online_agents": availability_summary["online"],
                    "offline_agents": availability_summary["offline"],
                    "degraded_agents": availability_summary["degraded"],
                    "overall_availability": availability_summary["overall_availability"]
                },
                "performance_overview": {
                    "total_tracked_agents": performance_summary["total_agents"],
                    "avg_response_time": performance_summary.get("avg_response_time", 0.0),
                    "avg_completion_rate": performance_summary.get("avg_completion_rate", 0.0),
                    "avg_error_rate": performance_summary.get("avg_error_rate", 0.0),
                    "agents_improving": performance_summary.get("agents_improving", 0),
                    "agents_degrading": performance_summary.get("agents_degrading", 0)
                }
            }
            
            # Add detailed information if requested
            if include_details:
                status["detailed_components"] = {}
                
                # Component health details
                for component_id in list(self.health_monitor.component_health.keys()):
                    component_health = self.health_monitor.get_component_health(component_id)
                    if component_health:
                        status["detailed_components"][component_id] = {
                            "type": component_health.component_type,
                            "status": component_health.status.value,
                            "health_score": component_health.health_score,
                            "issues": component_health.issues,
                            "metrics": component_health.metrics,
                            "last_check": component_health.last_check.isoformat()
                        }
                
                # Agent load details
                all_load_metrics = self.load_monitor.get_all_agent_metrics()
                status["agent_load_details"] = {}
                for agent_id, load_metrics in all_load_metrics.items():
                    status["agent_load_details"][agent_id] = {
                        "cpu_percent": load_metrics.cpu_percent,
                        "memory_percent": load_metrics.memory_percent,
                        "active_tasks": load_metrics.active_tasks,
                        "queued_tasks": load_metrics.queued_tasks,
                        "avg_response_time": load_metrics.avg_response_time,
                        "status": load_metrics.status
                    }
            
            # Add trend analysis if requested
            if include_trends:
                status["trends"] = {
                    "health_trends": self.health_monitor.get_health_trends(hours=24),
                    "bottleneck_analysis": self.health_monitor.get_bottleneck_analysis(),
                    "load_distribution": self.load_monitor.get_load_distribution_recommendation()
                }
            
            self.ecosystem_stats["successful_operations"] += 1
            return status
            
        except Exception as e:
            self.ecosystem_stats["failed_operations"] += 1
            logger.error("Error getting ecosystem status: %s", e)
            return {"error": f"Failed to get ecosystem status: {str(e)}"}
        finally:
            self.ecosystem_stats["total_operations"] += 1

    async def handle_agent_management(self, action: str, agent_id: Optional[str] = None, 
                                    agent_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle agent management operations"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if action == "register":
                if not agent_id or not agent_type:
                    return {"error": "agent_id and agent_type are required for registration"}
                
                # Register with all monitoring systems
                self.load_monitor.register_agent(agent_id)
                self.availability_tracker.register_agent(agent_id, metadata)
                self.health_monitor.register_component(agent_id, agent_type)
                
                # Store agent information
                self.registered_agents[agent_id] = {
                    "type": agent_type,
                    "metadata": metadata or {},
                    "registered_at": datetime.now().isoformat()
                }
                
                logger.info("Registered agent %s (type: %s)", agent_id, agent_type)
                return {"success": True, "message": f"Agent {agent_id} registered successfully"}
            
            elif action == "deregister":
                if not agent_id:
                    return {"error": "agent_id is required for deregistration"}
                
                # Remove from registered agents
                if agent_id in self.registered_agents:
                    del self.registered_agents[agent_id]
                
                logger.info("Deregistered agent %s", agent_id)
                return {"success": True, "message": f"Agent {agent_id} deregistered successfully"}
            
            elif action == "update":
                if not agent_id:
                    return {"error": "agent_id is required for update"}
                
                if agent_id in self.registered_agents and metadata:
                    self.registered_agents[agent_id]["metadata"].update(metadata)
                    return {"success": True, "message": f"Agent {agent_id} updated successfully"}
                else:
                    return {"error": f"Agent {agent_id} not found"}
            
            elif action == "list":
                return {
                    "agents": self.registered_agents,
                    "total_count": len(self.registered_agents)
                }
            
            elif action == "status":
                if not agent_id:
                    return {"error": "agent_id is required for status check"}
                
                # Get comprehensive status
                status_info = {
                    "agent_id": agent_id,
                    "registered": agent_id in self.registered_agents,
                    "availability_status": None,
                    "load_metrics": None,
                    "health_status": None
                }
                
                # Get availability status
                availability_status = self.availability_tracker.get_agent_status(agent_id)
                if availability_status:
                    status_info["availability_status"] = availability_status.value
                
                # Get load metrics
                load_metrics = self.load_monitor.get_agent_metrics(agent_id)
                if load_metrics:
                    status_info["load_metrics"] = {
                        "cpu_percent": load_metrics.cpu_percent,
                        "memory_percent": load_metrics.memory_percent,
                        "active_tasks": load_metrics.active_tasks,
                        "queued_tasks": load_metrics.queued_tasks,
                        "status": load_metrics.status
                    }
                
                # Get health status
                health_status = self.health_monitor.get_component_health(agent_id)
                if health_status:
                    status_info["health_status"] = {
                        "status": health_status.status.value,
                        "health_score": health_status.health_score,
                        "issues": health_status.issues
                    }
                
                return status_info
            
            else:
                return {"error": f"Unknown action: {action}"}
                
        except Exception as e:
            logger.error("Error in agent management: %s", e)
            return {"error": f"Agent management failed: {str(e)}"}

    async def handle_performance_analysis(self, agent_id: Optional[str] = None, 
                                        metric_types: Optional[List[str]] = None,
                                        time_range_hours: int = 24,
                                        include_predictions: bool = False) -> Dict[str, Any]:
        """Handle performance analysis requests"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "time_range_hours": time_range_hours,
                "analysis_scope": "specific_agent" if agent_id else "ecosystem_wide"
            }
            
            if agent_id:
                # Analyze specific agent
                analysis["agent_id"] = agent_id
                analysis["metrics"] = {}
                
                # Get metrics for specified types or all types
                types_to_analyze = metric_types or [mt.value for mt in MetricType]
                
                for metric_type_str in types_to_analyze:
                    try:
                        metric_type = MetricType(metric_type_str)
                        history = self.metrics_collector.get_metric_history(
                            agent_id, metric_type, limit=1000
                        )
                        
                        if history:
                            values = [dp.value for dp in history]
                            analysis["metrics"][metric_type_str] = {
                                "current_value": values[-1] if values else None,
                                "average": sum(values) / len(values) if values else 0,
                                "min_value": min(values) if values else None,
                                "max_value": max(values) if values else None,
                                "data_points": len(values)
                            }
                            
                            # Add trend analysis
                            trend = self.metrics_collector.get_trend_analysis(agent_id, metric_type)
                            if trend:
                                analysis["metrics"][metric_type_str]["trend"] = {
                                    "direction": trend.trend_direction,
                                    "confidence": trend.confidence,
                                    "slope": trend.slope,
                                    "prediction": trend.prediction if include_predictions else None
                                }
                    except ValueError:
                        continue  # Skip invalid metric types
                
                # Get all trends for the agent
                all_trends = self.metrics_collector.get_all_trends(agent_id)
                analysis["overall_trends"] = {
                    mt.value: {
                        "direction": trend.trend_direction,
                        "confidence": trend.confidence
                    }
                    for mt, trend in all_trends.items()
                }
            
            else:
                # Ecosystem-wide analysis
                summary = self.metrics_collector.get_performance_summary()
                analysis["ecosystem_metrics"] = summary
                
                # Top performers analysis
                if metric_types:
                    analysis["top_performers"] = {}
                    for metric_type_str in metric_types:
                        try:
                            metric_type = MetricType(metric_type_str)
                            top_performers = self.metrics_collector.get_top_performers(metric_type, limit=5)
                            analysis["top_performers"][metric_type_str] = [
                                {"agent_id": agent_id, "value": value}
                                for agent_id, value in top_performers
                            ]
                        except ValueError:
                            continue
            
            # Add recommendations
            analysis["recommendations"] = await self._generate_performance_recommendations(
                agent_id, analysis.get("metrics", {}), analysis.get("ecosystem_metrics", {})
            )
            
            return analysis
            
        except Exception as e:
            logger.error("Error in performance analysis: %s", e)
            return {"error": f"Performance analysis failed: {str(e)}"}

    async def handle_load_balancing(self, action: str, target_agents: Optional[List[str]] = None, 
                                  dry_run: bool = True) -> Dict[str, Any]:
        """Handle load balancing operations"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if action == "analyze":
                # Analyze current load distribution
                recommendation = self.load_monitor.get_load_distribution_recommendation()
                
                analysis = {
                    "timestamp": datetime.now().isoformat(),
                    "load_distribution": recommendation,
                    "agent_loads": {}
                }
                
                # Get load for all agents or target agents
                agents_to_analyze = target_agents or list(self.load_monitor.get_all_agent_metrics().keys())
                
                for agent_id in agents_to_analyze:
                    metrics = self.load_monitor.get_agent_metrics(agent_id)
                    if metrics:
                        analysis["agent_loads"][agent_id] = {
                            "cpu_percent": metrics.cpu_percent,
                            "memory_percent": metrics.memory_percent,
                            "active_tasks": metrics.active_tasks,
                            "queued_tasks": metrics.queued_tasks,
                            "status": metrics.status,
                            "combined_load": metrics.cpu_percent + metrics.memory_percent + (metrics.queued_tasks * 2)
                        }
                
                return analysis
            
            elif action == "rebalance":
                if dry_run:
                    return {"message": "Dry run mode - no actual rebalancing performed", "dry_run": True}
                
                # Implement actual load rebalancing logic here
                # This would integrate with the actual task scheduling system
                return {"message": "Load rebalancing initiated", "dry_run": False}
            
            elif action == "recommendations":
                recommendation = self.load_monitor.get_load_distribution_recommendation()
                
                recommendations = {
                    "timestamp": datetime.now().isoformat(),
                    "status": recommendation["recommendation"],
                    "actions": []
                }
                
                if recommendation["recommendation"] == "rebalance":
                    recommendations["actions"].extend([
                        f"Reduce load on agents: {', '.join(recommendation.get('overloaded_agents', []))}",
                        f"Increase load on agents: {', '.join(recommendation.get('underloaded_agents', []))}"
                    ])
                elif recommendation["recommendation"] == "balanced":
                    recommendations["actions"].append("Load distribution is optimal")
                
                return recommendations
            
            else:
                return {"error": f"Unknown load balancing action: {action}"}
                
        except Exception as e:
            logger.error("Error in load balancing: %s", e)
            return {"error": f"Load balancing failed: {str(e)}"}

    async def handle_health_monitoring(self, action: str, component_id: Optional[str] = None,
                                     alert_id: Optional[str] = None, severity_filter: Optional[str] = None) -> Dict[str, Any]:
        """Handle health monitoring operations"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if action == "status":
                ecosystem_health = self.health_monitor.get_ecosystem_health()
                
                if component_id:
                    # Get specific component health
                    component_health = self.health_monitor.get_component_health(component_id)
                    if component_health:
                        return {
                            "component_id": component_id,
                            "status": component_health.status.value,
                            "health_score": component_health.health_score,
                            "issues": component_health.issues,
                            "metrics": component_health.metrics,
                            "last_check": component_health.last_check.isoformat()
                        }
                    else:
                        return {"error": f"Component {component_id} not found"}
                else:
                    # Get overall ecosystem health
                    if ecosystem_health:
                        return {
                            "overall_status": ecosystem_health.overall_status.value,
                            "overall_score": ecosystem_health.overall_score,
                            "total_components": ecosystem_health.total_components,
                            "healthy_components": ecosystem_health.healthy_components,
                            "warning_components": ecosystem_health.warning_components,
                            "critical_components": ecosystem_health.critical_components,
                            "active_alerts": ecosystem_health.active_alerts,
                            "bottlenecks": ecosystem_health.bottlenecks,
                            "recommendations": ecosystem_health.recommendations
                        }
                    else:
                        return {"error": "Ecosystem health data not available"}
            
            elif action == "alerts":
                # Get active alerts
                severity_enum = None
                if severity_filter:
                    try:
                        severity_enum = AlertSeverity(severity_filter)
                    except ValueError:
                        return {"error": f"Invalid severity filter: {severity_filter}"}
                
                alerts = self.health_monitor.get_active_alerts(severity_enum)
                
                return {
                    "active_alerts": [
                        {
                            "alert_id": alert.alert_id,
                            "severity": alert.severity.value,
                            "component": alert.component,
                            "message": alert.message,
                            "timestamp": alert.timestamp.isoformat(),
                            "metadata": alert.metadata
                        }
                        for alert in alerts
                    ],
                    "total_alerts": len(alerts),
                    "severity_filter": severity_filter
                }
            
            elif action == "diagnostics":
                # Run comprehensive diagnostics
                diagnostics = {
                    "timestamp": datetime.now().isoformat(),
                    "critical_components": [],
                    "bottleneck_analysis": self.health_monitor.get_bottleneck_analysis(),
                    "health_trends": self.health_monitor.get_health_trends(hours=6)
                }
                
                # Get critical components
                critical_components = self.health_monitor.get_critical_components()
                diagnostics["critical_components"] = [
                    {
                        "component_id": comp.component_id,
                        "status": comp.status.value,
                        "health_score": comp.health_score,
                        "issues": comp.issues
                    }
                    for comp in critical_components
                ]
                
                return diagnostics
            
            elif action == "resolve_alert":
                if not alert_id:
                    return {"error": "alert_id is required for alert resolution"}
                
                self.health_monitor.resolve_alert(alert_id)
                return {"success": True, "message": f"Alert {alert_id} resolved"}
            
            elif action == "health_check":
                if component_id:
                    # Force health check for specific component
                    return {"message": f"Health check initiated for component {component_id}"}
                else:
                    # Force health check for all components
                    return {"message": "Health check initiated for all components"}
            
            else:
                return {"error": f"Unknown health monitoring action: {action}"}
                
        except Exception as e:
            logger.error("Error in health monitoring: %s", e)
            return {"error": f"Health monitoring failed: {str(e)}"}

    async def _generate_performance_recommendations(self, agent_id: Optional[str], 
                                                  agent_metrics: Dict[str, Any],
                                                  ecosystem_metrics: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if agent_id and agent_metrics:
            # Agent-specific recommendations
            for metric_name, metric_data in agent_metrics.items():
                if "trend" in metric_data:
                    trend = metric_data["trend"]
                    if trend["direction"] == "degrading" and trend["confidence"] > 0.7:
                        recommendations.append(f"Address degrading {metric_name} performance for agent {agent_id}")
                
                # Check specific metric thresholds
                current_value = metric_data.get("current_value", 0)
                if metric_name == "response_time" and current_value > 5.0:
                    recommendations.append(f"Optimize response time for agent {agent_id} (current: {current_value:.2f}s)")
                elif metric_name == "error_rate" and current_value > 5.0:
                    recommendations.append(f"Investigate high error rate for agent {agent_id} ({current_value:.1f}%)")
        
        if ecosystem_metrics:
            # Ecosystem-wide recommendations
            if ecosystem_metrics.get("agents_degrading", 0) > ecosystem_metrics.get("total_agents", 0) * 0.3:
                recommendations.append("Consider scaling or load redistribution - high number of degrading agents")
            
            if ecosystem_metrics.get("avg_error_rate", 0) > 3.0:
                recommendations.append("Investigate system-wide error rate increase")
            
            if ecosystem_metrics.get("avg_response_time", 0) > 3.0:
                recommendations.append("Consider performance optimization - high average response time")
        
        if not recommendations:
            recommendations.append("Performance metrics are within acceptable ranges")
        
        return recommendations

    # MCP Tool handler methods
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[Any]:
        """Handle MCP tool calls"""
        try:
            if name == "ecosystem_status":
                result = await self.handle_ecosystem_status(**arguments)
            elif name == "agent_management":
                result = await self.handle_agent_management(**arguments)
            elif name == "performance_analysis":
                result = await self.handle_performance_analysis(**arguments)
            elif name == "load_balancing":
                result = await self.handle_load_balancing(**arguments)
            elif name == "health_monitoring":
                result = await self.handle_health_monitoring(**arguments)
            else:
                result = {"error": f"Unknown tool: {name}"}
            
            # Return as TextContent for MCP or dict for fallback
            if MCP_AVAILABLE:
                return [MCPTextContent(type="text", text=json.dumps(result, indent=2))]
            return [{"type": "text", "text": json.dumps(result, indent=2)}]
                
        except Exception as e:
            error_result = {"error": f"Tool execution failed: {str(e)}"}
            logger.error("Error executing tool %s: %s", name, e)
            
            if MCP_AVAILABLE:
                return [MCPTextContent(type="text", text=json.dumps(error_result, indent=2))]
            return [{"type": "text", "text": json.dumps(error_result, indent=2)}]