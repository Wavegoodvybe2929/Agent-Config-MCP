"""
Agent Hook Execution Engine for MCP Swarm Intelligence Server.

This module provides a comprehensive event-driven hook execution system that supports
the agent lifecycle management, swarm coordination, and memory persistence hooks
defined in agent-config/agent-hooks.md.
"""

from typing import Dict, List, Callable, Any, Optional, Set
import asyncio
from enum import Enum
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import traceback
import time
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Enumeration of all supported hook types for MCP Swarm Intelligence Server."""
    PRE_TASK_SETUP = "pre_task_setup"
    PRE_TASK_VALIDATION = "pre_task_validation"
    TASK_EXECUTION = "task_execution"
    POST_TASK_VALIDATION = "post_task_validation"
    POST_TASK_CLEANUP = "post_task_cleanup"
    INTER_AGENT_COORDINATION = "inter_agent_coordination"
    AGENT_HANDOFF_PREPARE = "agent_handoff_prepare"
    AGENT_HANDOFF_EXECUTE = "agent_handoff_execute"
    COLLABORATION_INIT = "collaboration_init"
    COLLABORATION_SYNC = "collaboration_sync"
    MEMORY_PERSISTENCE = "memory_persistence"
    CONTINUOUS_INTEGRATION = "continuous_integration"
    ERROR_HANDLING = "error_handling"
    CLEANUP = "cleanup"


@dataclass
class HookResult:
    """Result of a single hook execution."""
    hook_name: str
    success: bool
    execution_time: float
    output: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert hook result to dictionary for serialization."""
        return {
            "hook_name": self.hook_name,
            "success": self.success,
            "execution_time": self.execution_time,
            "output": self.output,
            "error": str(self.error) if self.error else None,
            "retry_count": self.retry_count,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class HookExecutionResult:
    """Result of executing a set of hooks."""
    hook_type: HookType
    hook_results: List[HookResult]
    total_time: float
    success_rate: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution result to dictionary for serialization."""
        return {
            "hook_type": self.hook_type.value,
            "hook_results": [result.to_dict() for result in self.hook_results],
            "total_time": self.total_time,
            "success_rate": self.success_rate,
            "hooks_executed": len(self.hook_results),
            "successful_hooks": sum(1 for r in self.hook_results if r.success),
            "failed_hooks": sum(1 for r in self.hook_results if not r.success),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class HookDefinition:
    """Definition of a hook with execution parameters and metadata."""
    name: str
    hook_type: HookType
    priority: int
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    handler: Optional[Callable] = None
    description: str = ""
    enabled: bool = True

    def __post_init__(self):
        """Validate hook definition parameters."""
        if self.priority < 0:
            raise ValueError(f"Hook priority must be >= 0, got {self.priority}")
        if self.timeout <= 0:
            raise ValueError(f"Hook timeout must be > 0, got {self.timeout}")
        if self.retry_count < 0:
            raise ValueError(f"Hook retry_count must be >= 0, got {self.retry_count}")


class HookExecutionEngine:
    """
    Main engine for executing agent hooks in the MCP Swarm Intelligence Server.
    
    This engine manages hook registration, dependency resolution, execution ordering,
    error handling with retries, and performance monitoring for all hook types.
    """

    def __init__(self):
        """Initialize the hook execution engine."""
        self.registered_hooks: Dict[HookType, List[HookDefinition]] = {}
        self.execution_history: List[HookExecutionResult] = []
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        self._hook_registry: Dict[str, HookDefinition] = {}
        self._execution_lock = asyncio.Lock()
        self._max_history_size = 1000
        
        # Initialize hook registry for each type
        for hook_type in HookType:
            self.registered_hooks[hook_type] = []

    async def register_hook(self, hook_def: HookDefinition) -> bool:
        """
        Register a hook for execution.
        
        Args:
            hook_def: Hook definition with execution parameters
            
        Returns:
            True if hook was registered successfully, False otherwise
        """
        try:
            # Validate hook definition
            if not hook_def.name:
                raise ValueError("Hook name cannot be empty")
            
            if hook_def.name in self._hook_registry:
                logger.warning("Hook %s already registered, updating", hook_def.name)
            
            # Validate dependencies exist or will exist
            for dep in hook_def.dependencies:
                if dep not in self._hook_registry and dep != hook_def.name:
                    logger.warning("Hook %s depends on %s which is not registered", hook_def.name, dep)
            
            # Register the hook
            self._hook_registry[hook_def.name] = hook_def
            self.registered_hooks[hook_def.hook_type].append(hook_def)
            
            logger.info("Registered hook %s for type %s", hook_def.name, hook_def.hook_type.value)
            return True
            
        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Failed to register hook %s: %s", hook_def.name, e)
            return False

    async def unregister_hook(self, hook_name: str) -> bool:
        """
        Unregister a hook by name.
        
        Args:
            hook_name: Name of the hook to unregister
            
        Returns:
            True if hook was unregistered successfully, False otherwise
        """
        try:
            if hook_name not in self._hook_registry:
                logger.warning("Hook %s not found for unregistration", hook_name)
                return False
            
            hook_def = self._hook_registry[hook_name]
            
            # Remove from type-specific registry
            self.registered_hooks[hook_def.hook_type] = [
                h for h in self.registered_hooks[hook_def.hook_type] 
                if h.name != hook_name
            ]
            
            # Remove from main registry
            del self._hook_registry[hook_name]
            
            logger.info("Unregistered hook %s", hook_name)
            return True
            
        except (KeyError, AttributeError) as e:
            logger.error("Failed to unregister hook %s: %s", hook_name, e)
            return False

    async def execute_hooks(
        self, 
        hook_type: HookType, 
        context: Dict[str, Any],
        filter_enabled: bool = True
    ) -> HookExecutionResult:
        """
        Execute all hooks of specified type with proper dependency ordering.
        
        Args:
            hook_type: Type of hooks to execute
            context: Execution context passed to all hooks
            filter_enabled: Whether to only execute enabled hooks
            
        Returns:
            Results of hook execution including individual hook results and summary
        """
        start_time = time.time()
        hook_results = []
        
        async with self._execution_lock:
            try:
                # Get hooks for this type
                hooks = self.registered_hooks.get(hook_type, [])
                
                if filter_enabled:
                    hooks = [h for h in hooks if h.enabled]
                
                if not hooks:
                    logger.info("No hooks registered for type %s", hook_type.value)
                    return HookExecutionResult(
                        hook_type=hook_type,
                        hook_results=[],
                        total_time=0.0,
                        success_rate=1.0
                    )
                
                # Resolve dependencies and order hooks
                ordered_hooks = self._resolve_hook_dependencies(hooks)
                
                logger.info("Executing %d hooks for type %s", len(ordered_hooks), hook_type.value)
                
                # Execute hooks in dependency order
                for hook in ordered_hooks:
                    hook_result = await self._execute_single_hook(hook, context)
                    hook_results.append(hook_result)
                    
                    # Track performance metrics
                    self._track_hook_performance(hook, hook_result.execution_time)
                
                # Calculate success rate
                successful_hooks = sum(1 for result in hook_results if result.success)
                success_rate = successful_hooks / len(hook_results) if hook_results else 1.0
                
                total_time = time.time() - start_time
                
                # Create execution result
                execution_result = HookExecutionResult(
                    hook_type=hook_type,
                    hook_results=hook_results,
                    total_time=total_time,
                    success_rate=success_rate
                )
                
                # Store in history (with size limit)
                self.execution_history.append(execution_result)
                if len(self.execution_history) > self._max_history_size:
                    self.execution_history = self.execution_history[-self._max_history_size:]
                
                logger.info(
                    "Hook execution completed for %s: %d/%d successful, total time: %.3fs",
                    hook_type.value, successful_hooks, len(hook_results), total_time
                )
                
                return execution_result
                
            except (asyncio.TimeoutError, ValueError, RuntimeError) as e:
                logger.error("Hook execution failed for %s: %s", hook_type.value, e)
                logger.error(traceback.format_exc())
                
                # Return failure result
                return HookExecutionResult(
                    hook_type=hook_type,
                    hook_results=hook_results,
                    total_time=time.time() - start_time,
                    success_rate=0.0
                )

    def _resolve_hook_dependencies(self, hooks: List[HookDefinition]) -> List[HookDefinition]:
        """
        Resolve hook execution order based on dependencies using topological sort.
        
        Args:
            hooks: List of hooks to order by dependencies
            
        Returns:
            Hooks ordered by dependencies and priority
        """
        # Create dependency graph
        hook_map = {hook.name: hook for hook in hooks}
        in_degree = {hook.name: 0 for hook in hooks}
        adjacency = {hook.name: [] for hook in hooks}
        
        # Build adjacency list and calculate in-degrees
        for hook in hooks:
            for dep in hook.dependencies:
                if dep in hook_map:
                    adjacency[dep].append(hook.name)
                    in_degree[hook.name] += 1
                else:
                    logger.warning(f"Hook {hook.name} depends on {dep} which is not in current set")
        
        # Topological sort with priority ordering
        queue = []
        result = []
        
        # Start with hooks that have no dependencies, ordered by priority
        no_deps = [hook for hook in hooks if in_degree[hook.name] == 0]
        no_deps.sort(key=lambda h: h.priority)
        queue.extend(no_deps)
        
        while queue:
            current_hook = queue.pop(0)
            result.append(current_hook)
            
            # Process hooks that depend on current hook
            next_hooks = []
            for next_hook_name in adjacency[current_hook.name]:
                in_degree[next_hook_name] -= 1
                if in_degree[next_hook_name] == 0:
                    next_hooks.append(hook_map[next_hook_name])
            
            # Sort by priority and add to queue
            next_hooks.sort(key=lambda h: h.priority)
            queue.extend(next_hooks)
        
        # Check for circular dependencies
        if len(result) != len(hooks):
            remaining = [hook for hook in hooks if hook not in result]
            logger.error("Circular dependency detected in hooks: %s", [h.name for h in remaining])
            # Add remaining hooks in priority order as fallback
            remaining.sort(key=lambda h: h.priority)
            result.extend(remaining)
        
        return result

    async def _execute_single_hook(
        self, 
        hook: HookDefinition, 
        context: Dict[str, Any]
    ) -> HookResult:
        """
        Execute a single hook with error handling and retries.
        
        Args:
            hook: Hook definition to execute
            context: Execution context
            
        Returns:
            Result of hook execution
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(hook.retry_count + 1):
            try:
                logger.debug("Executing hook %s (attempt %d)", hook.name, attempt + 1)
                
                if hook.handler is None:
                    logger.warning("Hook %s has no handler, skipping", hook.name)
                    return HookResult(
                        hook_name=hook.name,
                        success=True,
                        execution_time=time.time() - start_time,
                        output="No handler defined",
                        retry_count=attempt
                    )
                
                # Execute hook with timeout
                try:
                    if asyncio.iscoroutinefunction(hook.handler):
                        output = await asyncio.wait_for(
                            hook.handler(context), 
                            timeout=hook.timeout
                        )
                    else:
                        # Run synchronous handler in thread pool
                        output = await asyncio.get_event_loop().run_in_executor(
                            None, hook.handler, context
                        )
                    
                    execution_time = time.time() - start_time
                    
                    logger.debug("Hook %s completed successfully in %.3fs", hook.name, execution_time)
                    
                    return HookResult(
                        hook_name=hook.name,
                        success=True,
                        execution_time=execution_time,
                        output=output,
                        retry_count=attempt
                    )
                    
                except asyncio.TimeoutError:
                    raise Exception(f"Hook {hook.name} timed out after {hook.timeout}s")
                
            except Exception as e:
                last_error = e
                execution_time = time.time() - start_time
                
                logger.warning("Hook %s failed on attempt %d: %s", hook.name, attempt + 1, e)
                
                # If not the last attempt, wait before retry
                if attempt < hook.retry_count:
                    await asyncio.sleep(hook.retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                
                # All attempts failed
                logger.error("Hook %s failed after %d attempts: %s", hook.name, attempt + 1, e)
                
                return HookResult(
                    hook_name=hook.name,
                    success=False,
                    execution_time=execution_time,
                    error=last_error,
                    retry_count=attempt
                )
        
        # This should never be reached due to the retry loop, but add for type safety
        return HookResult(
            hook_name=hook.name,
            success=False,
            execution_time=time.time() - start_time,
            error=last_error,
            retry_count=hook.retry_count
        )

    def _track_hook_performance(
        self, 
        hook: HookDefinition, 
        execution_time: float
    ) -> None:
        """
        Track hook execution performance metrics.
        
        Args:
            hook: Hook definition that was executed
            execution_time: Time taken to execute the hook
        """
        hook_name = hook.name
        
        if hook_name not in self.performance_metrics:
            self.performance_metrics[hook_name] = {
                "execution_count": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
                "success_count": 0,
                "failure_count": 0,
                "last_execution": None
            }
        
        metrics = self.performance_metrics[hook_name]
        metrics["execution_count"] += 1
        metrics["total_time"] += execution_time
        metrics["average_time"] = metrics["total_time"] / metrics["execution_count"]
        metrics["min_time"] = min(metrics["min_time"], execution_time)
        metrics["max_time"] = max(metrics["max_time"], execution_time)
        metrics["last_execution"] = datetime.utcnow().isoformat()

    def get_hook_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive hook execution statistics.
        
        Returns:
            Dictionary containing hook execution statistics
        """
        total_executions = len(self.execution_history)
        if total_executions == 0:
            return {
                "total_executions": 0,
                "average_success_rate": 0.0,
                "registered_hooks": len(self._hook_registry),
                "hook_types": len(self.registered_hooks),
                "performance_metrics": {}
            }
        
        # Calculate overall statistics
        total_success_rate = sum(result.success_rate for result in self.execution_history)
        average_success_rate = total_success_rate / total_executions
        
        # Recent performance (last 10 executions)
        recent_history = self.execution_history[-10:]
        recent_success_rate = sum(result.success_rate for result in recent_history) / len(recent_history)
        
        return {
            "total_executions": total_executions,
            "average_success_rate": average_success_rate,
            "recent_success_rate": recent_success_rate,
            "registered_hooks": len(self._hook_registry),
            "hook_types": len([ht for ht, hooks in self.registered_hooks.items() if hooks]),
            "performance_metrics": self.performance_metrics.copy(),
            "recent_executions": [result.to_dict() for result in recent_history]
        }

    def get_hooks_for_type(self, hook_type: HookType) -> List[Dict[str, Any]]:
        """
        Get all registered hooks for a specific type.
        
        Args:
            hook_type: Type of hooks to retrieve
            
        Returns:
            List of hook information dictionaries
        """
        hooks = self.registered_hooks.get(hook_type, [])
        return [
            {
                "name": hook.name,
                "priority": hook.priority,
                "dependencies": hook.dependencies,
                "timeout": hook.timeout,
                "retry_count": hook.retry_count,
                "description": hook.description,
                "enabled": hook.enabled,
                "has_handler": hook.handler is not None
            }
            for hook in hooks
        ]

    async def clear_history(self, older_than_hours: Optional[int] = None) -> int:
        """
        Clear execution history, optionally only older entries.
        
        Args:
            older_than_hours: Only clear entries older than this many hours
            
        Returns:
            Number of entries cleared
        """
        if older_than_hours is None:
            count = len(self.execution_history)
            self.execution_history.clear()
            return count
        
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        original_count = len(self.execution_history)
        
        self.execution_history = [
            result for result in self.execution_history
            if result.timestamp > cutoff_time
        ]
        
        return original_count - len(self.execution_history)