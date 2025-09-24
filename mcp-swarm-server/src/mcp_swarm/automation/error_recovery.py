"""
Automated Error Recovery System for MCP Swarm Intelligence Server

This module implements automated error recovery with alternative path execution,
providing robust error handling, recovery strategies, and failover mechanisms
for maintaining system stability and continuity.
"""

from typing import Dict, List, Any, Optional, Callable, Union
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import traceback
import json

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    ALTERNATIVE_PATH = "alternative_path"
    ESCALATE = "escalate"
    ABORT = "abort"
    CIRCUIT_BREAKER = "circuit_breaker"


class ErrorCategory(Enum):
    """Error categorization for targeted recovery"""
    NETWORK = "network"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    DEPENDENCY = "dependency"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information about an error"""
    error_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    traceback: str
    context_data: Dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=datetime.now)
    affected_components: List[str] = field(default_factory=list)
    retry_count: int = 0


@dataclass
class WorkflowError:
    """Workflow-specific error information"""
    workflow_id: str
    task_id: Optional[str]
    agent_id: Optional[str]
    error_context: ErrorContext
    workflow_state: Dict[str, Any] = field(default_factory=dict)
    recovery_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowPath:
    """Workflow execution path definition"""
    path_id: str
    name: str
    steps: List[Dict[str, Any]]
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    estimated_duration: float = 300.0
    risk_level: str = "medium"


@dataclass
class AlternativeExecution:
    """Alternative execution path result"""
    original_path_id: str
    alternative_path_id: str
    execution_result: Dict[str, Any]
    success: bool
    execution_time: float
    recovery_notes: str = ""


@dataclass
class RecoveryResult:
    """Result of error recovery attempt"""
    error_id: str
    recovery_strategy: RecoveryStrategy
    success: bool
    recovery_time: float
    attempts_made: int
    final_result: Any = None
    recovery_notes: str = ""
    alternative_used: Optional[str] = None
    escalated_to: Optional[str] = None


class RecoveryStrategies:
    """Collection of recovery strategies and their implementations"""

    def __init__(self):
        """Initialize recovery strategies"""
        self.strategy_map = {
            RecoveryStrategy.RETRY: self._retry_strategy,
            RecoveryStrategy.FALLBACK: self._fallback_strategy,
            RecoveryStrategy.ALTERNATIVE_PATH: self._alternative_path_strategy,
            RecoveryStrategy.ESCALATE: self._escalate_strategy,
            RecoveryStrategy.ABORT: self._abort_strategy,
            RecoveryStrategy.CIRCUIT_BREAKER: self._circuit_breaker_strategy
        }
        
        # Strategy configuration
        self.retry_configs = {
            ErrorCategory.NETWORK: {"max_attempts": 3, "delay": 1.0, "backoff": 2.0},
            ErrorCategory.TIMEOUT: {"max_attempts": 2, "delay": 0.5, "backoff": 1.5},
            ErrorCategory.RESOURCE: {"max_attempts": 5, "delay": 2.0, "backoff": 1.2},
            ErrorCategory.VALIDATION: {"max_attempts": 1, "delay": 0.1, "backoff": 1.0},
            ErrorCategory.DEPENDENCY: {"max_attempts": 3, "delay": 1.5, "backoff": 2.0},
            ErrorCategory.SYSTEM: {"max_attempts": 2, "delay": 3.0, "backoff": 3.0}
        }

    async def execute_strategy(
        self,
        strategy: RecoveryStrategy,
        error: WorkflowError,
        recovery_context: Dict[str, Any]
    ) -> RecoveryResult:
        """Execute a specific recovery strategy"""
        if strategy not in self.strategy_map:
            raise ValueError(f"Unknown recovery strategy: {strategy}")
        
        strategy_func = self.strategy_map[strategy]
        return await strategy_func(error, recovery_context)

    async def _retry_strategy(
        self,
        error: WorkflowError,
        recovery_context: Dict[str, Any]
    ) -> RecoveryResult:
        """Implement retry recovery strategy"""
        start_time = datetime.now()
        error_context = error.error_context
        category = error_context.category
        
        # Get retry configuration
        retry_config = self.retry_configs.get(category, self.retry_configs[ErrorCategory.NETWORK])
        max_attempts = retry_config["max_attempts"]
        base_delay = retry_config["delay"]
        backoff_factor = retry_config["backoff"]
        
        # Check if we've already exceeded max attempts
        if error_context.retry_count >= max_attempts:
            return RecoveryResult(
                error_id=error_context.error_id,
                recovery_strategy=RecoveryStrategy.RETRY,
                success=False,
                recovery_time=(datetime.now() - start_time).total_seconds(),
                attempts_made=error_context.retry_count,
                recovery_notes=f"Max retry attempts ({max_attempts}) exceeded"
            )
        
        # Attempt retry with exponential backoff
        for attempt in range(error_context.retry_count, max_attempts):
            # Calculate delay
            delay = base_delay * (backoff_factor ** attempt)
            await asyncio.sleep(delay)
            
            try:
                # Attempt to re-execute the failed operation
                original_function = recovery_context.get("original_function")
                original_args = recovery_context.get("original_args", ())
                original_kwargs = recovery_context.get("original_kwargs", {})
                
                if original_function:
                    if asyncio.iscoroutinefunction(original_function):
                        result = await original_function(*original_args, **original_kwargs)
                    else:
                        result = original_function(*original_args, **original_kwargs)
                    
                    return RecoveryResult(
                        error_id=error_context.error_id,
                        recovery_strategy=RecoveryStrategy.RETRY,
                        success=True,
                        recovery_time=(datetime.now() - start_time).total_seconds(),
                        attempts_made=attempt + 1,
                        final_result=result,
                        recovery_notes=f"Successful retry after {attempt + 1} attempts"
                    )
                
            except Exception as retry_error:
                logger.warning("Retry attempt %d failed: %s", attempt + 1, retry_error)
                continue
        
        return RecoveryResult(
            error_id=error_context.error_id,
            recovery_strategy=RecoveryStrategy.RETRY,
            success=False,
            recovery_time=(datetime.now() - start_time).total_seconds(),
            attempts_made=max_attempts,
            recovery_notes=f"All {max_attempts} retry attempts failed"
        )

    async def _fallback_strategy(
        self,
        error: WorkflowError,
        recovery_context: Dict[str, Any]
    ) -> RecoveryResult:
        """Implement fallback recovery strategy"""
        start_time = datetime.now()
        
        # Get fallback function
        fallback_function = recovery_context.get("fallback_function")
        fallback_args = recovery_context.get("fallback_args", ())
        fallback_kwargs = recovery_context.get("fallback_kwargs", {})
        
        if not fallback_function:
            return RecoveryResult(
                error_id=error.error_context.error_id,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                success=False,
                recovery_time=(datetime.now() - start_time).total_seconds(),
                attempts_made=1,
                recovery_notes="No fallback function provided"
            )
        
        try:
            if asyncio.iscoroutinefunction(fallback_function):
                result = await fallback_function(*fallback_args, **fallback_kwargs)
            else:
                result = fallback_function(*fallback_args, **fallback_kwargs)
            
            return RecoveryResult(
                error_id=error.error_context.error_id,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                success=True,
                recovery_time=(datetime.now() - start_time).total_seconds(),
                attempts_made=1,
                final_result=result,
                recovery_notes="Successfully executed fallback function"
            )
            
        except Exception as fallback_error:
            return RecoveryResult(
                error_id=error.error_context.error_id,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                success=False,
                recovery_time=(datetime.now() - start_time).total_seconds(),
                attempts_made=1,
                recovery_notes=f"Fallback function failed: {fallback_error}"
            )

    async def _alternative_path_strategy(
        self,
        error: WorkflowError,
        recovery_context: Dict[str, Any]
    ) -> RecoveryResult:
        """Implement alternative path recovery strategy"""
        start_time = datetime.now()
        
        # Get alternative paths
        alternative_paths = recovery_context.get("alternative_paths", [])
        if not alternative_paths:
            return RecoveryResult(
                error_id=error.error_context.error_id,
                recovery_strategy=RecoveryStrategy.ALTERNATIVE_PATH,
                success=False,
                recovery_time=(datetime.now() - start_time).total_seconds(),
                attempts_made=1,
                recovery_notes="No alternative paths available"
            )
        
        # Try each alternative path
        for path in alternative_paths:
            try:
                path_executor = recovery_context.get("path_executor")
                if path_executor:
                    result = await path_executor(path, error.workflow_state)
                    
                    return RecoveryResult(
                        error_id=error.error_context.error_id,
                        recovery_strategy=RecoveryStrategy.ALTERNATIVE_PATH,
                        success=True,
                        recovery_time=(datetime.now() - start_time).total_seconds(),
                        attempts_made=len(alternative_paths),
                        final_result=result,
                        alternative_used=path.get("path_id", "unknown"),
                        recovery_notes=f"Successfully executed alternative path: {path.get('name', 'unnamed')}"
                    )
                    
            except Exception as path_error:
                logger.warning("Alternative path failed: %s", path_error)
                continue
        
        return RecoveryResult(
            error_id=error.error_context.error_id,
            recovery_strategy=RecoveryStrategy.ALTERNATIVE_PATH,
            success=False,
            recovery_time=(datetime.now() - start_time).total_seconds(),
            attempts_made=len(alternative_paths),
            recovery_notes="All alternative paths failed"
        )

    async def _escalate_strategy(
        self,
        error: WorkflowError,
        recovery_context: Dict[str, Any]
    ) -> RecoveryResult:
        """Implement escalation recovery strategy"""
        start_time = datetime.now()
        
        escalation_handler = recovery_context.get("escalation_handler")
        escalation_target = recovery_context.get("escalation_target", "orchestrator")
        
        if escalation_handler:
            try:
                escalation_result = await escalation_handler(error, escalation_target)
                
                return RecoveryResult(
                    error_id=error.error_context.error_id,
                    recovery_strategy=RecoveryStrategy.ESCALATE,
                    success=True,
                    recovery_time=(datetime.now() - start_time).total_seconds(),
                    attempts_made=1,
                    final_result=escalation_result,
                    escalated_to=escalation_target,
                    recovery_notes=f"Successfully escalated to {escalation_target}"
                )
                
            except Exception as escalation_error:
                return RecoveryResult(
                    error_id=error.error_context.error_id,
                    recovery_strategy=RecoveryStrategy.ESCALATE,
                    success=False,
                    recovery_time=(datetime.now() - start_time).total_seconds(),
                    attempts_made=1,
                    recovery_notes=f"Escalation failed: {escalation_error}"
                )
        
        return RecoveryResult(
            error_id=error.error_context.error_id,
            recovery_strategy=RecoveryStrategy.ESCALATE,
            success=False,
            recovery_time=(datetime.now() - start_time).total_seconds(),
            attempts_made=1,
            recovery_notes="No escalation handler available"
        )

    async def _abort_strategy(
        self,
        error: WorkflowError,
        recovery_context: Dict[str, Any]
    ) -> RecoveryResult:
        """Implement abort recovery strategy"""
        start_time = datetime.now()
        
        # Clean up resources if cleanup function provided
        cleanup_function = recovery_context.get("cleanup_function")
        if cleanup_function:
            try:
                if asyncio.iscoroutinefunction(cleanup_function):
                    await cleanup_function(error.workflow_state)
                else:
                    cleanup_function(error.workflow_state)
            except Exception as cleanup_error:
                logger.warning("Cleanup function failed during abort: %s", cleanup_error)
        
        return RecoveryResult(
            error_id=error.error_context.error_id,
            recovery_strategy=RecoveryStrategy.ABORT,
            success=True,  # Abort is always "successful" in that it completes
            recovery_time=(datetime.now() - start_time).total_seconds(),
            attempts_made=1,
            recovery_notes="Workflow aborted and resources cleaned up"
        )

    async def _circuit_breaker_strategy(
        self,
        error: WorkflowError,
        recovery_context: Dict[str, Any]
    ) -> RecoveryResult:
        """Implement circuit breaker recovery strategy"""
        start_time = datetime.now()
        
        # Circuit breaker logic - temporarily disable failing component
        component_id = error.error_context.affected_components[0] if error.error_context.affected_components else "unknown"
        circuit_breaker_timeout = recovery_context.get("circuit_breaker_timeout", 300)  # 5 minutes default
        
        # Store circuit breaker state (in a real implementation, this would be persistent)
        circuit_breaker_state = recovery_context.get("circuit_breaker_state", {})
        circuit_breaker_state[component_id] = {
            "disabled_until": datetime.now() + timedelta(seconds=circuit_breaker_timeout),
            "error_count": circuit_breaker_state.get(component_id, {}).get("error_count", 0) + 1
        }
        
        return RecoveryResult(
            error_id=error.error_context.error_id,
            recovery_strategy=RecoveryStrategy.CIRCUIT_BREAKER,
            success=True,
            recovery_time=(datetime.now() - start_time).total_seconds(),
            attempts_made=1,
            recovery_notes=f"Circuit breaker activated for {component_id} for {circuit_breaker_timeout} seconds"
        )


class AlternativePathManager:
    """Manager for alternative workflow execution paths"""

    def __init__(self):
        """Initialize alternative path manager"""
        self.path_registry: Dict[str, List[WorkflowPath]] = {}
        self.path_success_rates: Dict[str, float] = {}

    def register_alternative_paths(
        self,
        primary_path_id: str,
        alternative_paths: List[WorkflowPath]
    ):
        """Register alternative paths for a primary workflow path"""
        self.path_registry[primary_path_id] = alternative_paths
        
        # Initialize success rates if not exist
        for path in alternative_paths:
            if path.path_id not in self.path_success_rates:
                self.path_success_rates[path.path_id] = 1.0

    def get_alternative_paths(
        self,
        primary_path_id: str,
        error_context: ErrorContext
    ) -> List[WorkflowPath]:
        """Get alternative paths sorted by success rate and suitability"""
        alternative_paths = self.path_registry.get(primary_path_id, [])
        
        # Filter paths based on error context
        suitable_paths = []
        for path in alternative_paths:
            if self._is_path_suitable(path, error_context):
                suitable_paths.append(path)
        
        # Sort by success rate (descending)
        suitable_paths.sort(
            key=lambda p: self.path_success_rates.get(p.path_id, 0.0),
            reverse=True
        )
        
        return suitable_paths

    def update_path_success_rate(self, path_id: str, success: bool):
        """Update success rate for a path based on execution result"""
        current_rate = self.path_success_rates.get(path_id, 1.0)
        
        # Simple moving average update (in real implementation, use more sophisticated tracking)
        if success:
            self.path_success_rates[path_id] = min(1.0, current_rate * 0.9 + 0.1)
        else:
            self.path_success_rates[path_id] = max(0.0, current_rate * 0.9)

    def _is_path_suitable(self, path: WorkflowPath, error_context: ErrorContext) -> bool:
        """Check if an alternative path is suitable for the error context"""
        # Simple suitability check based on error category
        if error_context.category == ErrorCategory.NETWORK:
            return "offline" in path.name.lower() or "local" in path.name.lower()
        elif error_context.category == ErrorCategory.RESOURCE:
            return "lightweight" in path.name.lower() or "minimal" in path.name.lower()
        elif error_context.category == ErrorCategory.TIMEOUT:
            return "quick" in path.name.lower() or "fast" in path.name.lower()
        
        return True  # Default to suitable if no specific criteria


class AutomatedErrorRecovery:
    """
    Automated error recovery with alternative path execution.
    
    This system provides comprehensive error handling, recovery strategies,
    and alternative path execution to maintain system stability and continuity.
    """

    def __init__(self):
        """Initialize automated error recovery system"""
        self.recovery_strategies = RecoveryStrategies()
        self.alternative_paths = AlternativePathManager()
        
        # Error tracking
        self.error_history: List[WorkflowError] = []
        self.recovery_history: List[RecoveryResult] = []
        
        # Recovery metrics
        self.metrics = {
            "total_errors": 0,
            "successful_recoveries": 0,
            "recovery_success_rate": 0.0,
            "average_recovery_time": 0.0,
            "most_common_error_category": ErrorCategory.UNKNOWN.value
        }

    async def handle_workflow_error(
        self, 
        error: WorkflowError
    ) -> RecoveryResult:
        """
        Handle workflow errors with automated recovery.
        
        Args:
            error: Workflow error to handle
            
        Returns:
            Recovery result with execution details
        """
        logger.info("Handling workflow error %s for workflow %s", 
                   error.error_context.error_id, error.workflow_id)
        
        # Add to error history
        self.error_history.append(error)
        self._update_error_metrics(error)
        
        try:
            # Determine recovery strategy
            recovery_strategy = self._determine_recovery_strategy(error)
            
            # Prepare recovery context
            recovery_context = await self._prepare_recovery_context(error)
            
            # Execute recovery strategy
            recovery_result = await self.recovery_strategies.execute_strategy(
                recovery_strategy, error, recovery_context
            )
            
            # Add to recovery history
            self.recovery_history.append(recovery_result)
            self._update_recovery_metrics(recovery_result)
            
            logger.info("Recovery %s for error %s: %s", 
                       "successful" if recovery_result.success else "failed",
                       error.error_context.error_id,
                       recovery_result.recovery_notes)
            
            return recovery_result
            
        except Exception as recovery_error:
            logger.error("Recovery system failed for error %s: %s", 
                        error.error_context.error_id, recovery_error)
            
            # Create failure recovery result
            return RecoveryResult(
                error_id=error.error_context.error_id,
                recovery_strategy=RecoveryStrategy.ABORT,
                success=False,
                recovery_time=0.0,
                attempts_made=0,
                recovery_notes=f"Recovery system error: {recovery_error}"
            )

    async def _execute_alternative_path(
        self, 
        original_path: WorkflowPath,
        error_context: ErrorContext
    ) -> AlternativeExecution:
        """
        Execute alternative workflow path.
        
        Args:
            original_path: Original workflow path that failed
            error_context: Context about the error
            
        Returns:
            Alternative execution result
        """
        start_time = datetime.now()
        
        # Get alternative paths
        alternative_paths = self.alternative_paths.get_alternative_paths(
            original_path.path_id, error_context
        )
        
        if not alternative_paths:
            return AlternativeExecution(
                original_path_id=original_path.path_id,
                alternative_path_id="none",
                execution_result={"error": "No alternative paths available"},
                success=False,
                execution_time=0.0,
                recovery_notes="No alternative paths found"
            )
        
        # Try the highest-rated alternative path
        best_alternative = alternative_paths[0]
        
        try:
            # Execute alternative path (simplified simulation)
            await asyncio.sleep(0.1)  # Simulate execution time
            
            execution_result = {
                "path_id": best_alternative.path_id,
                "path_name": best_alternative.name,
                "steps_completed": len(best_alternative.steps),
                "status": "completed"
            }
            
            # Update path success rate
            self.alternative_paths.update_path_success_rate(best_alternative.path_id, True)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AlternativeExecution(
                original_path_id=original_path.path_id,
                alternative_path_id=best_alternative.path_id,
                execution_result=execution_result,
                success=True,
                execution_time=execution_time,
                recovery_notes=f"Successfully executed alternative path: {best_alternative.name}"
            )
            
        except Exception as alt_error:
            # Update path success rate
            self.alternative_paths.update_path_success_rate(best_alternative.path_id, False)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AlternativeExecution(
                original_path_id=original_path.path_id,
                alternative_path_id=best_alternative.path_id,
                execution_result={"error": str(alt_error)},
                success=False,
                execution_time=execution_time,
                recovery_notes=f"Alternative path failed: {alt_error}"
            )

    def _determine_recovery_strategy(self, error: WorkflowError) -> RecoveryStrategy:
        """Determine the best recovery strategy for the error"""
        error_context = error.error_context
        
        # Strategy selection based on error characteristics
        if error_context.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ESCALATE
        
        # Category-based strategy selection
        category_strategies = {
            ErrorCategory.NETWORK: RecoveryStrategy.RETRY,
            ErrorCategory.TIMEOUT: RecoveryStrategy.RETRY,
            ErrorCategory.RESOURCE: RecoveryStrategy.ALTERNATIVE_PATH,
            ErrorCategory.VALIDATION: RecoveryStrategy.FALLBACK,
            ErrorCategory.DEPENDENCY: RecoveryStrategy.ALTERNATIVE_PATH,
            ErrorCategory.SYSTEM: RecoveryStrategy.CIRCUIT_BREAKER
        }
        
        strategy = category_strategies.get(error_context.category, RecoveryStrategy.RETRY)
        
        # Adjust strategy based on retry count
        if error_context.retry_count > 2:
            if strategy == RecoveryStrategy.RETRY:
                strategy = RecoveryStrategy.ALTERNATIVE_PATH
            elif strategy == RecoveryStrategy.ALTERNATIVE_PATH:
                strategy = RecoveryStrategy.ESCALATE
        
        return strategy

    async def _prepare_recovery_context(self, error: WorkflowError) -> Dict[str, Any]:
        """Prepare context for recovery strategy execution"""
        context = {
            "error_category": error.error_context.category,
            "error_severity": error.error_context.severity,
            "workflow_state": error.workflow_state,
            "recovery_metadata": error.recovery_metadata
        }
        
        # Add strategy-specific context
        if error.error_context.category == ErrorCategory.NETWORK:
            context.update({
                "original_function": error.recovery_metadata.get("original_function"),
                "original_args": error.recovery_metadata.get("original_args", ()),
                "original_kwargs": error.recovery_metadata.get("original_kwargs", {}),
            })
        
        # Add alternative paths if available
        if error.workflow_id in self.alternative_paths.path_registry:
            context["alternative_paths"] = self.alternative_paths.get_alternative_paths(
                error.workflow_id, error.error_context
            )
        
        return context

    def _update_error_metrics(self, error: WorkflowError):
        """Update error tracking metrics"""
        self.metrics["total_errors"] += 1
        
        # Update most common error category
        category_counts = {}
        for hist_error in self.error_history:
            cat = hist_error.error_context.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        if category_counts:
            self.metrics["most_common_error_category"] = max(
                category_counts.items(), key=lambda x: x[1]
            )[0]

    def _update_recovery_metrics(self, recovery_result: RecoveryResult):
        """Update recovery tracking metrics"""
        if recovery_result.success:
            self.metrics["successful_recoveries"] += 1
        
        # Update success rate
        total_recoveries = len(self.recovery_history)
        if total_recoveries > 0:
            self.metrics["recovery_success_rate"] = (
                self.metrics["successful_recoveries"] / total_recoveries
            ) * 100.0
        
        # Update average recovery time
        if total_recoveries > 0:
            total_time = sum(r.recovery_time for r in self.recovery_history)
            self.metrics["average_recovery_time"] = total_time / total_recoveries

    def get_error_metrics(self) -> Dict[str, Any]:
        """Get error and recovery metrics"""
        return self.metrics.copy()

    def get_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns for insights"""
        if not self.error_history:
            return {"message": "No error history available"}
        
        # Analyze error categories
        category_counts = {}
        for error in self.error_history:
            cat = error.error_context.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Analyze error severity
        severity_counts = {}
        for error in self.error_history:
            sev = error.error_context.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        # Recovery strategy effectiveness
        strategy_success = {}
        for result in self.recovery_history:
            strategy = result.recovery_strategy.value
            if strategy not in strategy_success:
                strategy_success[strategy] = {"total": 0, "successful": 0}
            
            strategy_success[strategy]["total"] += 1
            if result.success:
                strategy_success[strategy]["successful"] += 1
        
        # Calculate success rates
        for strategy, stats in strategy_success.items():
            stats["success_rate"] = (stats["successful"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        
        return {
            "error_categories": category_counts,
            "error_severities": severity_counts,
            "recovery_strategy_effectiveness": strategy_success,
            "total_errors": len(self.error_history),
            "total_recoveries": len(self.recovery_history),
            "overall_recovery_rate": self.metrics["recovery_success_rate"]
        }