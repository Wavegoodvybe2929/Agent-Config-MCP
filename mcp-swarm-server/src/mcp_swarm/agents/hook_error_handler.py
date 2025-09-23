"""
Agent Hook Error Handler for MCP Swarm Intelligence Server.

This module provides comprehensive error handling and recovery strategies
for agent hook execution failures.
"""

from typing import Dict, List, Any, Optional, Type
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import re
import asyncio

from .hook_engine import HookDefinition

logger = logging.getLogger(__name__)


class ErrorClassification(Enum):
    """Classification of hook execution errors."""
    TIMEOUT_ERROR = "timeout_error"
    NETWORK_ERROR = "network_error"
    PERMISSION_ERROR = "permission_error"
    RESOURCE_ERROR = "resource_error"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_ERROR = "dependency_error"
    VALIDATION_ERROR = "validation_error"
    RUNTIME_ERROR = "runtime_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryActionType(Enum):
    """Types of recovery actions."""
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    RECONFIGURE = "reconfigure"
    RESTART = "restart"
    ABORT = "abort"


@dataclass
class RecoveryAction:
    """Recovery action to take for an error."""
    action_type: RecoveryActionType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_probability: float = 0.5
    max_attempts: int = 1
    delay_seconds: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recovery action to dictionary."""
        return {
            "action_type": self.action_type.value,
            "description": self.description,
            "parameters": self.parameters,
            "success_probability": self.success_probability,
            "max_attempts": self.max_attempts,
            "delay_seconds": self.delay_seconds
        }


@dataclass
class ErrorRecoveryResult:
    """Result of error recovery attempt."""
    original_error: Exception
    error_classification: ErrorClassification
    recovery_action: RecoveryAction
    recovery_successful: bool
    recovery_attempts: int
    final_result: Optional[Any] = None
    recovery_error: Optional[Exception] = None
    recovery_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error recovery result to dictionary."""
        return {
            "original_error": str(self.original_error),
            "error_classification": self.error_classification.value,
            "recovery_action": self.recovery_action.to_dict(),
            "recovery_successful": self.recovery_successful,
            "recovery_attempts": self.recovery_attempts,
            "final_result": self.final_result,
            "recovery_error": str(self.recovery_error) if self.recovery_error else None,
            "recovery_timestamp": self.recovery_timestamp.isoformat()
        }


@dataclass
class ErrorPattern:
    """Pattern for matching and handling specific errors."""
    name: str
    error_types: List[Type[Exception]]
    error_message_patterns: List[str]
    classification: ErrorClassification
    recovery_strategy: RecoveryAction
    confidence: float = 0.8
    
    def matches(self, error: Exception) -> bool:
        """Check if error matches this pattern."""
        # Check error type
        if any(isinstance(error, error_type) for error_type in self.error_types):
            return True
        
        # Check error message patterns
        error_message = str(error).lower()
        for pattern in self.error_message_patterns:
            if re.search(pattern.lower(), error_message):
                return True
        
        return False


class HookErrorHandler:
    """
    Comprehensive error handler for agent hook execution.
    
    This handler classifies errors, applies appropriate recovery strategies,
    and learns from error patterns to improve future handling.
    """
    
    def __init__(self):
        """Initialize the hook error handler."""
        self.error_patterns: List[ErrorPattern] = []
        self.recovery_strategies: Dict[ErrorClassification, List[RecoveryAction]] = {}
        self.error_history: List[ErrorRecoveryResult] = []
        self.recovery_statistics: Dict[str, Dict[str, Any]] = {}
        self._max_history_size = 1000
        
        # Initialize with default error patterns and recovery strategies
        self._initialize_default_patterns()
        self._initialize_default_strategies()

    def _initialize_default_patterns(self):
        """Initialize default error patterns for common issues."""
        patterns = [
            ErrorPattern(
                name="timeout_pattern",
                error_types=[asyncio.TimeoutError],
                error_message_patterns=[r"timeout", r"timed out", r"time.*out"],
                classification=ErrorClassification.TIMEOUT_ERROR,
                recovery_strategy=RecoveryAction(
                    action_type=RecoveryActionType.RETRY,
                    description="Retry with increased timeout",
                    parameters={"timeout_multiplier": 1.5, "max_retries": 2},
                    success_probability=0.7,
                    max_attempts=2,
                    delay_seconds=2.0
                )
            ),
            
            ErrorPattern(
                name="network_pattern",
                error_types=[ConnectionError, OSError],
                error_message_patterns=[
                    r"connection.*refused", r"network.*unreachable", 
                    r"connection.*reset", r"connection.*timeout"
                ],
                classification=ErrorClassification.NETWORK_ERROR,
                recovery_strategy=RecoveryAction(
                    action_type=RecoveryActionType.RETRY,
                    description="Retry with exponential backoff",
                    parameters={"backoff_multiplier": 2.0, "max_retries": 3},
                    success_probability=0.6,
                    max_attempts=3,
                    delay_seconds=1.0
                )
            ),
            
            ErrorPattern(
                name="permission_pattern",
                error_types=[PermissionError, OSError],
                error_message_patterns=[
                    r"permission.*denied", r"access.*denied", 
                    r"forbidden", r"unauthorized"
                ],
                classification=ErrorClassification.PERMISSION_ERROR,
                recovery_strategy=RecoveryAction(
                    action_type=RecoveryActionType.ESCALATE,
                    description="Escalate permission issue to administrator",
                    parameters={"escalation_level": "admin"},
                    success_probability=0.3,
                    max_attempts=1,
                    delay_seconds=0.0
                )
            ),
            
            ErrorPattern(
                name="resource_pattern",
                error_types=[MemoryError, OSError],
                error_message_patterns=[
                    r"out of memory", r"disk.*full", r"no space.*left",
                    r"resource.*unavailable", r"too many.*files"
                ],
                classification=ErrorClassification.RESOURCE_ERROR,
                recovery_strategy=RecoveryAction(
                    action_type=RecoveryActionType.FALLBACK,
                    description="Use fallback resource allocation",
                    parameters={"reduce_memory": True, "cleanup_temp": True},
                    success_probability=0.5,
                    max_attempts=1,
                    delay_seconds=5.0
                )
            ),
            
            ErrorPattern(
                name="validation_pattern",
                error_types=[ValueError, TypeError],
                error_message_patterns=[
                    r"invalid.*value", r"invalid.*type", r"validation.*failed",
                    r"missing.*required", r"unexpected.*format"
                ],
                classification=ErrorClassification.VALIDATION_ERROR,
                recovery_strategy=RecoveryAction(
                    action_type=RecoveryActionType.RECONFIGURE,
                    description="Apply default configuration and retry",
                    parameters={"use_defaults": True, "validate_input": True},
                    success_probability=0.8,
                    max_attempts=1,
                    delay_seconds=0.5
                )
            )
        ]
        
        self.error_patterns.extend(patterns)

    def _initialize_default_strategies(self):
        """Initialize default recovery strategies for each error classification."""
        self.recovery_strategies = {
            ErrorClassification.TIMEOUT_ERROR: [
                RecoveryAction(
                    action_type=RecoveryActionType.RETRY,
                    description="Retry with increased timeout",
                    parameters={"timeout_multiplier": 1.5},
                    success_probability=0.7,
                    max_attempts=2,
                    delay_seconds=2.0
                ),
                RecoveryAction(
                    action_type=RecoveryActionType.SKIP,
                    description="Skip non-critical hook execution",
                    parameters={"skip_reason": "timeout"},
                    success_probability=1.0,
                    max_attempts=1,
                    delay_seconds=0.0
                )
            ],
            
            ErrorClassification.NETWORK_ERROR: [
                RecoveryAction(
                    action_type=RecoveryActionType.RETRY,
                    description="Retry with exponential backoff",
                    parameters={"backoff_multiplier": 2.0},
                    success_probability=0.6,
                    max_attempts=3,
                    delay_seconds=1.0
                ),
                RecoveryAction(
                    action_type=RecoveryActionType.FALLBACK,
                    description="Use cached data if available",
                    parameters={"use_cache": True},
                    success_probability=0.4,
                    max_attempts=1,
                    delay_seconds=0.0
                )
            ],
            
            ErrorClassification.PERMISSION_ERROR: [
                RecoveryAction(
                    action_type=RecoveryActionType.ESCALATE,
                    description="Escalate to administrator",
                    parameters={"escalation_level": "admin"},
                    success_probability=0.3,
                    max_attempts=1,
                    delay_seconds=0.0
                ),
                RecoveryAction(
                    action_type=RecoveryActionType.SKIP,
                    description="Skip hook requiring permissions",
                    parameters={"skip_reason": "insufficient_permissions"},
                    success_probability=1.0,
                    max_attempts=1,
                    delay_seconds=0.0
                )
            ],
            
            ErrorClassification.RESOURCE_ERROR: [
                RecoveryAction(
                    action_type=RecoveryActionType.FALLBACK,
                    description="Reduce resource usage and retry",
                    parameters={"reduce_memory": True, "cleanup_temp": True},
                    success_probability=0.5,
                    max_attempts=1,
                    delay_seconds=5.0
                ),
                RecoveryAction(
                    action_type=RecoveryActionType.RESTART,
                    description="Restart hook execution with clean state",
                    parameters={"clean_restart": True},
                    success_probability=0.7,
                    max_attempts=1,
                    delay_seconds=10.0
                )
            ],
            
            ErrorClassification.CONFIGURATION_ERROR: [
                RecoveryAction(
                    action_type=RecoveryActionType.RECONFIGURE,
                    description="Apply default configuration",
                    parameters={"use_defaults": True},
                    success_probability=0.8,
                    max_attempts=1,
                    delay_seconds=0.5
                ),
                RecoveryAction(
                    action_type=RecoveryActionType.ESCALATE,
                    description="Request configuration assistance",
                    parameters={"escalation_level": "config"},
                    success_probability=0.4,
                    max_attempts=1,
                    delay_seconds=0.0
                )
            ],
            
            ErrorClassification.DEPENDENCY_ERROR: [
                RecoveryAction(
                    action_type=RecoveryActionType.RETRY,
                    description="Retry after dependency check",
                    parameters={"check_dependencies": True},
                    success_probability=0.6,
                    max_attempts=2,
                    delay_seconds=3.0
                ),
                RecoveryAction(
                    action_type=RecoveryActionType.FALLBACK,
                    description="Use fallback implementation",
                    parameters={"use_fallback": True},
                    success_probability=0.5,
                    max_attempts=1,
                    delay_seconds=0.0
                )
            ],
            
            ErrorClassification.VALIDATION_ERROR: [
                RecoveryAction(
                    action_type=RecoveryActionType.RECONFIGURE,
                    description="Fix validation issues and retry",
                    parameters={"validate_input": True, "sanitize_data": True},
                    success_probability=0.8,
                    max_attempts=1,
                    delay_seconds=0.5
                )
            ],
            
            ErrorClassification.RUNTIME_ERROR: [
                RecoveryAction(
                    action_type=RecoveryActionType.RETRY,
                    description="Retry with error handling",
                    parameters={"enhanced_error_handling": True},
                    success_probability=0.4,
                    max_attempts=2,
                    delay_seconds=1.0
                ),
                RecoveryAction(
                    action_type=RecoveryActionType.FALLBACK,
                    description="Use safe fallback implementation",
                    parameters={"safe_mode": True},
                    success_probability=0.6,
                    max_attempts=1,
                    delay_seconds=0.0
                )
            ],
            
            ErrorClassification.UNKNOWN_ERROR: [
                RecoveryAction(
                    action_type=RecoveryActionType.RETRY,
                    description="Single retry for unknown error",
                    parameters={"single_retry": True},
                    success_probability=0.3,
                    max_attempts=1,
                    delay_seconds=2.0
                ),
                RecoveryAction(
                    action_type=RecoveryActionType.ESCALATE,
                    description="Escalate unknown error for analysis",
                    parameters={"escalation_level": "technical"},
                    success_probability=0.2,
                    max_attempts=1,
                    delay_seconds=0.0
                )
            ]
        }

    async def handle_hook_error(
        self, 
        hook: HookDefinition, 
        error: Exception,
        context: Dict[str, Any]
    ) -> ErrorRecoveryResult:
        """
        Handle hook execution error with recovery strategies.
        
        Args:
            hook: Hook definition that failed
            error: Exception that occurred
            context: Execution context
            
        Returns:
            Error recovery result
        """
        try:
            # Classify the error
            error_classification = self._classify_error(error)
            
            # Select recovery strategy
            recovery_action = self._select_recovery_strategy(
                error_classification, hook, error, context
            )
            
            logger.info(
                "Handling error for hook %s: %s -> %s", 
                hook.name, error_classification.value, recovery_action.action_type.value
            )
            
            # Apply recovery strategy
            recovery_result = await self._apply_recovery_strategy(
                error_classification, hook, context, error, recovery_action
            )
            
            # Log error for analysis and learning
            self._log_error_for_analysis(hook, error, recovery_result)
            
            # Update recovery statistics
            self._update_recovery_statistics(error_classification, recovery_action, recovery_result)
            
            return recovery_result
            
        except (RuntimeError, ValueError, TypeError) as recovery_error:
            logger.error("Error during error recovery for hook %s: %s", hook.name, recovery_error)
            
            # Return failed recovery result
            return ErrorRecoveryResult(
                original_error=error,
                error_classification=ErrorClassification.UNKNOWN_ERROR,
                recovery_action=RecoveryAction(
                    action_type=RecoveryActionType.ABORT,
                    description="Recovery failed"
                ),
                recovery_successful=False,
                recovery_attempts=0,
                recovery_error=recovery_error
            )

    def _classify_error(self, error: Exception) -> ErrorClassification:
        """
        Classify error type for appropriate handling.
        
        Args:
            error: Exception to classify
            
        Returns:
            Error classification
        """
        # Check against known patterns
        for pattern in self.error_patterns:
            if pattern.matches(error):
                logger.debug("Classified error as %s using pattern %s", pattern.classification.value, pattern.name)
                return pattern.classification
        
        # Fallback classification based on exception type
        if isinstance(error, asyncio.TimeoutError):
            return ErrorClassification.TIMEOUT_ERROR
        elif isinstance(error, (ConnectionError, OSError)):
            return ErrorClassification.NETWORK_ERROR
        elif isinstance(error, PermissionError):
            return ErrorClassification.PERMISSION_ERROR
        elif isinstance(error, (MemoryError, FileNotFoundError)):
            return ErrorClassification.RESOURCE_ERROR
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorClassification.VALIDATION_ERROR
        elif isinstance(error, ImportError):
            return ErrorClassification.DEPENDENCY_ERROR
        elif isinstance(error, RuntimeError):
            return ErrorClassification.RUNTIME_ERROR
        else:
            return ErrorClassification.UNKNOWN_ERROR

    def _select_recovery_strategy(
        self, 
        error_classification: ErrorClassification,
        hook: HookDefinition,
        _error: Exception,  # Marked as unused with underscore
        _context: Dict[str, Any]  # Marked as unused with underscore
    ) -> RecoveryAction:
        """
        Select the best recovery strategy for the error.
        
        Args:
            error_classification: Classified error type
            hook: Hook that failed
            error: Original error
            context: Execution context
            
        Returns:
            Selected recovery action
        """
        # Get available strategies for this error type
        strategies = self.recovery_strategies.get(error_classification, [])
        
        if not strategies:
            # Default fallback strategy
            return RecoveryAction(
                action_type=RecoveryActionType.ABORT,
                description="No recovery strategy available",
                success_probability=0.0,
                max_attempts=0
            )
        
        # Select strategy based on context and success probability
        # For now, select the strategy with highest success probability
        # In the future, this could be enhanced with ML-based selection
        best_strategy = max(strategies, key=lambda s: s.success_probability)
        
        # Customize strategy based on hook and context
        customized_strategy = RecoveryAction(
            action_type=best_strategy.action_type,
            description=f"{best_strategy.description} for hook {hook.name}",
            parameters=best_strategy.parameters.copy(),
            success_probability=best_strategy.success_probability,
            max_attempts=best_strategy.max_attempts,
            delay_seconds=best_strategy.delay_seconds
        )
        
        # Adjust parameters based on hook characteristics
        if hook.timeout and "timeout_multiplier" in customized_strategy.parameters:
            customized_strategy.parameters["new_timeout"] = (
                hook.timeout * customized_strategy.parameters["timeout_multiplier"]
            )
        
        return customized_strategy

    async def _apply_recovery_strategy(
        self, 
        error_class: ErrorClassification,
        hook: HookDefinition,
        context: Dict[str, Any],
        original_error: Exception,
        recovery_action: RecoveryAction
    ) -> ErrorRecoveryResult:
        """
        Apply appropriate recovery strategy for error.
        
        Args:
            error_class: Error classification
            hook: Hook definition
            context: Execution context
            original_error: Original error that occurred
            recovery_action: Recovery action to apply
            
        Returns:
            Recovery result
        """
        recovery_attempts = 0
        
        try:
            for attempt in range(recovery_action.max_attempts):
                recovery_attempts += 1
                
                if recovery_action.delay_seconds > 0:
                    await asyncio.sleep(recovery_action.delay_seconds * attempt)
                
                # Apply recovery strategy based on action type
                if recovery_action.action_type == RecoveryActionType.RETRY:
                    result = await self._apply_retry_strategy(hook, context, recovery_action)
                elif recovery_action.action_type == RecoveryActionType.FALLBACK:
                    result = await self._apply_fallback_strategy(hook, context, recovery_action)
                elif recovery_action.action_type == RecoveryActionType.RECONFIGURE:
                    result = await self._apply_reconfigure_strategy(hook, context, recovery_action)
                elif recovery_action.action_type == RecoveryActionType.SKIP:
                    result = await self._apply_skip_strategy(hook, context, recovery_action)
                elif recovery_action.action_type == RecoveryActionType.ESCALATE:
                    result = await self._apply_escalate_strategy(hook, context, recovery_action)
                elif recovery_action.action_type == RecoveryActionType.RESTART:
                    result = await self._apply_restart_strategy(hook, context, recovery_action)
                else:  # ABORT
                    result = None
                
                if result is not None:
                    return ErrorRecoveryResult(
                        original_error=original_error,
                        error_classification=error_class,
                        recovery_action=recovery_action,
                        recovery_successful=True,
                        recovery_attempts=recovery_attempts,
                        final_result=result
                    )
            
            # All recovery attempts failed
            return ErrorRecoveryResult(
                original_error=original_error,
                error_classification=error_class,
                recovery_action=recovery_action,
                recovery_successful=False,
                recovery_attempts=recovery_attempts
            )
            
        except (OSError, RuntimeError, ValueError) as recovery_error:
            return ErrorRecoveryResult(
                original_error=original_error,
                error_classification=error_class,
                recovery_action=recovery_action,
                recovery_successful=False,
                recovery_attempts=recovery_attempts,
                recovery_error=recovery_error
            )

    async def _apply_retry_strategy(
        self, 
        hook: HookDefinition, 
        context: Dict[str, Any],
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Apply retry recovery strategy."""
        if hook.handler is None:
            return None
        
        # Modify context based on recovery parameters
        modified_context = context.copy()
        if "enhanced_error_handling" in recovery_action.parameters:
            modified_context["error_handling_enabled"] = True
        
        # Execute hook with modified context
        if asyncio.iscoroutinefunction(hook.handler):
            return await hook.handler(modified_context)
        else:
            return hook.handler(modified_context)

    async def _apply_fallback_strategy(
        self, 
        hook: HookDefinition, 
        context: Dict[str, Any],
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Apply fallback recovery strategy."""
        # Return a safe fallback result
        return {
            "hook_name": hook.name,
            "status": "fallback_executed",
            "fallback_reason": recovery_action.description,
            "original_context": context
        }

    async def _apply_reconfigure_strategy(
        self, 
        hook: HookDefinition, 
        context: Dict[str, Any],
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Apply reconfigure recovery strategy."""
        # Apply default configuration and retry
        modified_context = context.copy()
        if "use_defaults" in recovery_action.parameters:
            modified_context.update({
                "use_default_config": True,
                "validate_input": recovery_action.parameters.get("validate_input", False)
            })
        
        if hook.handler and asyncio.iscoroutinefunction(hook.handler):
            return await hook.handler(modified_context)
        elif hook.handler:
            return hook.handler(modified_context)
        
        return None

    async def _apply_skip_strategy(
        self, 
        hook: HookDefinition, 
        context: Dict[str, Any],
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Apply skip recovery strategy."""
        return {
            "hook_name": hook.name,
            "status": "skipped",
            "skip_reason": recovery_action.parameters.get("skip_reason", "error_recovery"),
            "original_context": context
        }

    async def _apply_escalate_strategy(
        self, 
        hook: HookDefinition, 
        _context: Dict[str, Any],  # Marked as unused
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Apply escalate recovery strategy."""
        escalation_level = recovery_action.parameters.get("escalation_level", "admin")
        
        logger.error(
            "Escalating hook %s error to %s level: %s", 
            hook.name, escalation_level, recovery_action.description
        )
        
        return {
            "hook_name": hook.name,
            "status": "escalated",
            "escalation_level": escalation_level,
            "escalation_reason": recovery_action.description,
            "requires_intervention": True
        }

    async def _apply_restart_strategy(
        self, 
        hook: HookDefinition, 
        context: Dict[str, Any],
        recovery_action: RecoveryAction
    ) -> Optional[Any]:
        """Apply restart recovery strategy."""
        # Clean context and retry
        clean_context = {
            "hook_name": hook.name,
            "restart_clean": True,
            "original_context_keys": list(context.keys())
        }
        
        if hook.handler and asyncio.iscoroutinefunction(hook.handler):
            return await hook.handler(clean_context)
        elif hook.handler:
            return hook.handler(clean_context)
        
        return None

    def _log_error_for_analysis(
        self, 
        hook: HookDefinition, 
        error: Exception,
        recovery_result: ErrorRecoveryResult
    ) -> None:
        """
        Log error details for pattern analysis and improvement.
        
        Args:
            hook: Hook that failed
            error: Original error
            recovery_result: Recovery attempt result
        """
        # Store in history for analysis
        self.error_history.append(recovery_result)
        
        # Limit history size
        if len(self.error_history) > self._max_history_size:
            self.error_history = self.error_history[-self._max_history_size:]
        
        # Log error details
        logger.info(
            "Hook error logged - Hook: %s, Error: %s, Classification: %s, Recovery: %s, Success: %s",
            hook.name,
            type(error).__name__,
            recovery_result.error_classification.value,
            recovery_result.recovery_action.action_type.value,
            recovery_result.recovery_successful
        )

    def _update_recovery_statistics(
        self, 
        error_classification: ErrorClassification,
        recovery_action: RecoveryAction,
        recovery_result: ErrorRecoveryResult
    ) -> None:
        """Update recovery statistics for learning and improvement."""
        key = f"{error_classification.value}_{recovery_action.action_type.value}"
        
        if key not in self.recovery_statistics:
            self.recovery_statistics[key] = {
                "total_attempts": 0,
                "successful_recoveries": 0,
                "success_rate": 0.0,
                "average_attempts": 0.0,
                "last_updated": None
            }
        
        stats = self.recovery_statistics[key]
        stats["total_attempts"] += 1
        if recovery_result.recovery_successful:
            stats["successful_recoveries"] += 1
        
        stats["success_rate"] = stats["successful_recoveries"] / stats["total_attempts"]
        stats["average_attempts"] = (
            (stats["average_attempts"] * (stats["total_attempts"] - 1) + 
             recovery_result.recovery_attempts) / stats["total_attempts"]
        )
        stats["last_updated"] = datetime.utcnow().isoformat()

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive error handling statistics.
        
        Returns:
            Dictionary containing error statistics
        """
        if not self.error_history:
            return {
                "total_errors_handled": 0,
                "recovery_success_rate": 0.0,
                "error_classifications": {},
                "recovery_actions": {},
                "patterns_learned": len(self.error_patterns)
            }
        
        total_errors = len(self.error_history)
        successful_recoveries = sum(1 for result in self.error_history if result.recovery_successful)
        
        # Count by classification
        classification_counts = {}
        for result in self.error_history:
            classification = result.error_classification.value
            classification_counts[classification] = classification_counts.get(classification, 0) + 1
        
        # Count by recovery action
        action_counts = {}
        for result in self.error_history:
            action = result.recovery_action.action_type.value
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "total_errors_handled": total_errors,
            "successful_recoveries": successful_recoveries,
            "recovery_success_rate": successful_recoveries / total_errors,
            "error_classifications": classification_counts,
            "recovery_actions": action_counts,
            "patterns_learned": len(self.error_patterns),
            "recovery_statistics": self.recovery_statistics.copy()
        }

    def add_custom_error_pattern(self, pattern: ErrorPattern) -> bool:
        """
        Add a custom error pattern for improved error handling.
        
        Args:
            pattern: Custom error pattern to add
            
        Returns:
            True if pattern was added successfully
        """
        try:
            # Validate pattern
            if not pattern.name or not pattern.error_types and not pattern.error_message_patterns:
                raise ValueError("Pattern must have name and at least one matching criteria")
            
            # Check for duplicate names
            existing_names = [p.name for p in self.error_patterns]
            if pattern.name in existing_names:
                logger.warning("Error pattern %s already exists, updating", pattern.name)
                self.error_patterns = [p for p in self.error_patterns if p.name != pattern.name]
            
            self.error_patterns.append(pattern)
            logger.info("Added custom error pattern: %s", pattern.name)
            return True
            
        except (ValueError, TypeError) as e:
            logger.error("Failed to add custom error pattern: %s", e)
            return False

    def get_recent_errors(
        self, 
        hours: int = 24,
        classification_filter: Optional[ErrorClassification] = None
    ) -> List[ErrorRecoveryResult]:
        """
        Get recent errors with optional filtering.
        
        Args:
            hours: Number of hours to look back
            classification_filter: Optional error classification filter
            
        Returns:
            List of recent error recovery results
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_errors = [
            result for result in self.error_history
            if result.recovery_timestamp >= cutoff_time
        ]
        
        if classification_filter:
            recent_errors = [
                result for result in recent_errors
                if result.error_classification == classification_filter
            ]
        
        return recent_errors