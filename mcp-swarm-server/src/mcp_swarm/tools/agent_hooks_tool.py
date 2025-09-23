"""
Agent Hooks MCP Tool for MCP Swarm Intelligence Server.

This module provides an MCP tool interface for managing and executing agent hooks,
including status monitoring, hook execution, configuration management, and performance monitoring.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime
from pathlib import Path

from ..agents.hook_engine import HookExecutionEngine, HookType
from ..agents.hook_config import HookConfigurationManager
from ..agents.hook_monitor import HookPerformanceMonitor
from ..agents.hook_error_handler import HookErrorHandler

logger = logging.getLogger(__name__)


class HookSystemManager:
    """Singleton manager for hook system components."""
    
    _instance: Optional['HookSystemManager'] = None
    
    def __init__(self):
        self.hook_engine: Optional[HookExecutionEngine] = None
        self.config_manager: Optional[HookConfigurationManager] = None
        self.performance_monitor: Optional[HookPerformanceMonitor] = None
        self.error_handler: Optional[HookErrorHandler] = None
    
    @classmethod
    def get_instance(cls) -> 'HookSystemManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_hook_engine(self) -> HookExecutionEngine:
        """Get or create the hook execution engine."""
        if self.hook_engine is None:
            self.hook_engine = HookExecutionEngine()
        return self.hook_engine

    def get_config_manager(self) -> HookConfigurationManager:
        """Get or create the hook configuration manager."""
        if self.config_manager is None:
            config_path = Path("agent-config/agent-hooks.md")
            self.config_manager = HookConfigurationManager(config_path)
        return self.config_manager

    def get_performance_monitor(self) -> HookPerformanceMonitor:
        """Get or create the hook performance monitor."""
        if self.performance_monitor is None:
            self.performance_monitor = HookPerformanceMonitor()
        return self.performance_monitor

    def get_error_handler(self) -> HookErrorHandler:
        """Get or create the hook error handler."""
        if self.error_handler is None:
            self.error_handler = HookErrorHandler()
        return self.error_handler


async def agent_hooks_tool(
    action: str = "status",
    hook_type: Optional[str] = None,
    agent_id: Optional[str] = None,
    execute_hooks: bool = False,
    context: Optional[Dict[str, Any]] = None,
    timeframe: str = "24h",
    configuration_reload: bool = False
) -> Dict[str, Any]:
    """
    MCP tool for managing and executing agent hooks.
    
    This comprehensive tool provides access to all hook management capabilities
    including status monitoring, hook execution, configuration management,
    and performance monitoring.
    
    Args:
        action: Hook management action (status/execute/configure/monitor/register/unregister)
        hook_type: Specific hook type to target (optional)
        agent_id: Specific agent for hook execution (optional)
        execute_hooks: Whether to actually execute hooks (default: False for safety)
        context: Execution context for hooks (optional)
        timeframe: Timeframe for performance analysis (default: 24h)
        configuration_reload: Whether to reload hook configurations (default: False)
        
    Returns:
        Hook management results with execution details and comprehensive information
    """
    try:
        # Get system components
        manager = HookSystemManager.get_instance()
        hook_engine = manager.get_hook_engine()
        config_manager = manager.get_config_manager()
        performance_monitor = manager.get_performance_monitor()
        
        # Reload configurations if requested
        if configuration_reload:
            await config_manager.reload_configurations()
        
        # Handle different actions
        if action == "status":
            return await _handle_status_action(
                hook_engine, config_manager, performance_monitor,
                hook_type, agent_id
            )
        
        elif action == "execute" and execute_hooks:
            return await _handle_execute_action(
                hook_engine, performance_monitor,
                hook_type, context or {}
            )
        
        elif action == "configure":
            return await _handle_configure_action(
                config_manager, hook_engine, configuration_reload
            )
        
        elif action == "monitor":
            return await _handle_monitor_action(
                performance_monitor, timeframe, hook_type
            )
        
        elif action == "register":
            return await _handle_register_action(
                hook_engine, config_manager, hook_type
            )
        
        elif action == "unregister":
            return await _handle_unregister_action(
                hook_engine, hook_type
            )
        
        else:
            return {
                "available_actions": [
                    "status", "execute", "configure", "monitor", "register", "unregister"
                ],
                "current_action": action,
                "execute_hooks_flag": execute_hooks,
                "help": {
                    "status": "Get comprehensive hook system status",
                    "execute": "Execute hooks (requires execute_hooks=True)",
                    "configure": "Manage hook configurations",
                    "monitor": "Performance monitoring and analysis",
                    "register": "Register new hooks",
                    "unregister": "Unregister existing hooks"
                }
            }
    
    except (ValueError, TypeError, AttributeError, KeyError) as e:
        logger.error("Agent hooks tool error: %s", e)
        return {
            "error": str(e),
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        }


async def _handle_status_action(
    hook_engine: HookExecutionEngine,
    config_manager: HookConfigurationManager,
    performance_monitor: HookPerformanceMonitor,
    hook_type: Optional[str],
    agent_id: Optional[str]
) -> Dict[str, Any]:
    """Handle status action for comprehensive system status."""
    
    # Load configurations to get current state
    await config_manager.load_hook_configurations()
    
    # Get hook engine statistics
    engine_stats = hook_engine.get_hook_statistics()
    
    # Get performance monitor statistics
    monitor_stats = performance_monitor.get_monitor_statistics()
    
    # Get configuration summary
    config_summary = config_manager.get_configuration_summary()
    
    status_result = {
        "hook_system_status": "active",
        "timestamp": datetime.utcnow().isoformat(),
        "system_components": {
            "hook_engine": "active",
            "config_manager": "active",
            "performance_monitor": "active",
            "error_handler": "active"
        },
        "configuration_summary": config_summary,
        "engine_statistics": engine_stats,
        "performance_statistics": monitor_stats,
        "hook_types_available": [ht.value for ht in HookType]
    }
    
    # Add specific hook type information if requested
    if hook_type:
        try:
            hook_type_enum = HookType(hook_type)
            hooks_for_type = hook_engine.get_hooks_for_type(hook_type_enum)
            status_result["specific_hook_type"] = {
                "hook_type": hook_type,
                "registered_hooks": hooks_for_type,
                "hook_count": len(hooks_for_type)
            }
        except ValueError:
            status_result["error"] = f"Invalid hook type: {hook_type}"
    
    # Add agent-specific information if requested
    if agent_id:
        status_result["agent_filter"] = {
            "agent_id": agent_id,
            "note": "Agent-specific filtering not yet implemented"
        }
    
    return status_result


async def _handle_execute_action(
    hook_engine: HookExecutionEngine,
    performance_monitor: HookPerformanceMonitor,
    hook_type: Optional[str],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Handle hook execution action."""
    
    if not hook_type:
        return {
            "error": "hook_type parameter is required for execution",
            "available_hook_types": [ht.value for ht in HookType]
        }
    
    try:
        hook_type_enum = HookType(hook_type)
    except ValueError:
        return {
            "error": f"Invalid hook type: {hook_type}",
            "available_hook_types": [ht.value for ht in HookType]
        }
    
    # Add execution metadata to context
    execution_context = context.copy()
    execution_context.update({
        "execution_id": f"hooks_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "executed_by": "agent_hooks_tool",
        "hook_type": hook_type,
        "execution_timestamp": datetime.utcnow().isoformat()
    })
    
    # Execute hooks of specified type
    execution_result = await hook_engine.execute_hooks(
        hook_type_enum, execution_context
    )
    
    # Monitor performance for each hook result
    for hook_result in execution_result.hook_results:
        await performance_monitor.monitor_hook_execution(
            hook_name=hook_result.hook_name,
            execution_time=hook_result.execution_time,
            success=hook_result.success,
            retry_count=hook_result.retry_count,
            error_type=type(hook_result.error).__name__ if hook_result.error else None
        )
    
    return {
        "action": "execute",
        "hook_type": hook_type,
        "execution_result": execution_result.to_dict(),
        "hooks_executed": len(execution_result.hook_results),
        "success_rate": execution_result.success_rate,
        "total_execution_time": execution_result.total_time,
        "execution_context": execution_context,
        "execution_timestamp": datetime.utcnow().isoformat()
    }


async def _handle_configure_action(
    config_manager: HookConfigurationManager,
    hook_engine: HookExecutionEngine,
    reload_requested: bool
) -> Dict[str, Any]:
    """Handle configuration management action."""
    
    # Load or reload configurations
    if reload_requested:
        hook_configs = await config_manager.load_hook_configurations()
        action_taken = "reload"
    else:
        hook_configs = config_manager.hook_configurations or await config_manager.load_hook_configurations()
        action_taken = "load"
    
    # Validate configurations
    validation_result = await config_manager.validate_hook_configurations(hook_configs)
    
    # Get configuration summary
    config_summary = config_manager.get_configuration_summary()
    
    # Register hooks with the engine (if not already registered)
    registration_results = []
    for hook_name, config in hook_configs.items():
        hook_def = config_manager.create_hook_definition(config)
        # Note: Handler will need to be set separately in actual implementation
        registration_success = await hook_engine.register_hook(hook_def)
        registration_results.append({
            "hook_name": hook_name,
            "registration_success": registration_success
        })
    
    return {
        "action": "configure",
        "configuration_action": action_taken,
        "configurations_loaded": len(hook_configs),
        "validation_result": {
            "is_valid": validation_result.is_valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "hook_count": validation_result.hook_count
        },
        "configuration_summary": config_summary,
        "registration_results": registration_results,
        "timestamp": datetime.utcnow().isoformat()
    }


async def _handle_monitor_action(
    performance_monitor: HookPerformanceMonitor,
    timeframe: str,
    hook_type_filter: Optional[str]
) -> Dict[str, Any]:
    """Handle performance monitoring action."""
    
    # Perform performance analysis
    performance_analysis = await performance_monitor.analyze_hook_performance(timeframe)
    
    # Get recent performance alerts
    recent_alerts = performance_monitor.get_performance_alerts(max_age_hours=24)
    
    # Generate optimization recommendations
    optimization_recommendations = await performance_monitor.optimize_hook_execution()
    
    monitor_result = {
        "action": "monitor",
        "timeframe": timeframe,
        "performance_analysis": performance_analysis.to_dict(),
        "recent_alerts": [alert.to_dict() for alert in recent_alerts],
        "optimization_recommendations": [rec.to_dict() for rec in optimization_recommendations],
        "monitor_timestamp": datetime.utcnow().isoformat()
    }
    
    # Add hook type specific monitoring if requested
    if hook_type_filter:
        try:
            HookType(hook_type_filter)  # Validate hook type
            # Filter analysis for specific hook type
            type_specific_performance = {
                hook_name: perf for hook_name, perf in performance_analysis.hook_performance.items()
                if hook_name == hook_type_filter
            }
            monitor_result["hook_type_specific"] = {
                "hook_type": hook_type_filter,
                "performance_data": type_specific_performance
            }
        except ValueError:
            monitor_result["hook_type_filter_error"] = f"Invalid hook type: {hook_type_filter}"
    
    return monitor_result


async def _handle_register_action(
    hook_engine: HookExecutionEngine,
    config_manager: HookConfigurationManager,
    hook_type: Optional[str]
) -> Dict[str, Any]:
    """Handle hook registration action."""
    
    if not hook_type:
        return {
            "error": "hook_type parameter is required for registration",
            "available_hook_types": [ht.value for ht in HookType],
            "action": "register"
        }
    
    try:
        HookType(hook_type)  # Validate hook type
    except ValueError:
        return {
            "error": f"Invalid hook type: {hook_type}",
            "available_hook_types": [ht.value for ht in HookType],
            "action": "register"
        }
    
    # Check if configuration exists
    hook_config = config_manager.get_hook_configuration(hook_type)
    
    if not hook_config:
        return {
            "error": f"No configuration found for hook type: {hook_type}",
            "action": "register",
            "suggestion": "Load configurations first using configure action"
        }
    
    # Create hook definition from configuration
    hook_def = config_manager.create_hook_definition(hook_config)
    
    # Register with engine
    registration_success = await hook_engine.register_hook(hook_def)
    
    return {
        "action": "register",
        "hook_type": hook_type,
        "registration_success": registration_success,
        "hook_configuration": {
            "name": hook_config.name,
            "priority": hook_config.priority,
            "timeout": hook_config.timeout,
            "retry_count": hook_config.retry_count,
            "dependencies": hook_config.dependencies,
            "enabled": hook_config.enabled
        },
        "timestamp": datetime.utcnow().isoformat()
    }


async def _handle_unregister_action(
    hook_engine: HookExecutionEngine,
    hook_type: Optional[str]
) -> Dict[str, Any]:
    """Handle hook unregistration action."""
    
    if not hook_type:
        return {
            "error": "hook_type parameter is required for unregistration",
            "available_hook_types": [ht.value for ht in HookType],
            "action": "unregister"
        }
    
    # Unregister hook
    unregister_success = await hook_engine.unregister_hook(hook_type)
    
    return {
        "action": "unregister",
        "hook_type": hook_type,
        "unregistration_success": unregister_success,
        "timestamp": datetime.utcnow().isoformat()
    }


# Additional utility tools for specific hook management needs

async def hook_status_tool(
    hook_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick status check tool for hook system.
    
    Args:
        hook_type: Optional specific hook type to check
        
    Returns:
        Quick status information
    """
    return await agent_hooks_tool(action="status", hook_type=hook_type)


async def hook_performance_tool(
    timeframe: str = "1h"
) -> Dict[str, Any]:
    """
    Performance monitoring tool for hooks.
    
    Args:
        timeframe: Time period for analysis
        
    Returns:
        Performance analysis results
    """
    return await agent_hooks_tool(action="monitor", timeframe=timeframe)


async def hook_configuration_tool(
    reload: bool = False
) -> Dict[str, Any]:
    """
    Configuration management tool for hooks.
    
    Args:
        reload: Whether to reload configurations
        
    Returns:
        Configuration management results
    """
    return await agent_hooks_tool(action="configure", configuration_reload=reload)


# Export the main tool for MCP registration
__all__ = [
    "agent_hooks_tool",
    "hook_status_tool", 
    "hook_performance_tool",
    "hook_configuration_tool"
]