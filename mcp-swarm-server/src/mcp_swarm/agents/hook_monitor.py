"""
Agent Hook Performance Monitor for MCP Swarm Intelligence Server.

This module provides comprehensive performance monitoring and optimization
recommendations for agent hook execution.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import statistics
from enum import Enum

from .hook_engine import HookType, HookResult, HookExecutionResult

logger = logging.getLogger(__name__)


class PerformanceIssueType(Enum):
    """Types of performance issues that can be detected."""
    SLOW_EXECUTION = "slow_execution"
    HIGH_FAILURE_RATE = "high_failure_rate"
    FREQUENT_RETRIES = "frequent_retries"
    TIMEOUT_ISSUES = "timeout_issues"
    DEPENDENCY_BOTTLENECK = "dependency_bottleneck"
    RESOURCE_CONTENTION = "resource_contention"


@dataclass
class PerformanceIssue:
    """Represents a detected performance issue."""
    issue_type: PerformanceIssueType
    hook_name: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    impact: str
    recommendation: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert performance issue to dictionary."""
        return {
            "issue_type": self.issue_type.value,
            "hook_name": self.hook_name,
            "severity": self.severity,
            "description": self.description,
            "impact": self.impact,
            "recommendation": self.recommendation,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "detected_at": self.detected_at.isoformat()
        }


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation for hook execution."""
    hook_name: str
    recommendation_type: str
    description: str
    expected_improvement: str
    implementation_effort: str  # "low", "medium", "high"
    priority: str  # "low", "medium", "high", "critical"
    technical_details: str
    estimated_impact: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert optimization recommendation to dictionary."""
        return {
            "hook_name": self.hook_name,
            "recommendation_type": self.recommendation_type,
            "description": self.description,
            "expected_improvement": self.expected_improvement,
            "implementation_effort": self.implementation_effort,
            "priority": self.priority,
            "technical_details": self.technical_details,
            "estimated_impact": self.estimated_impact or {}
        }


@dataclass
class PerformanceAnalysis:
    """Comprehensive performance analysis result."""
    timeframe: str
    total_executions: int
    analysis_timestamp: datetime
    hook_performance: Dict[str, Dict[str, Any]]
    issues: List[PerformanceIssue]
    recommendations: List[OptimizationRecommendation]
    summary_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert performance analysis to dictionary."""
        return {
            "timeframe": self.timeframe,
            "total_executions": self.total_executions,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "hook_performance": self.hook_performance,
            "issues": [issue.to_dict() for issue in self.issues],
            "recommendations": [rec.to_dict() for rec in self.recommendations],
            "summary_metrics": self.summary_metrics
        }


class HookPerformanceMonitor:
    """
    Performance monitor for agent hook execution in MCP Swarm Intelligence Server.
    
    This monitor tracks execution metrics, identifies performance issues,
    and provides optimization recommendations for hook execution.
    """

    def __init__(self):
        """Initialize the hook performance monitor."""
        self.execution_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.performance_alerts: List[PerformanceIssue] = []
        self.analysis_history: List[PerformanceAnalysis] = []
        self._thresholds = {
            "slow_execution": 5.0,  # seconds
            "high_failure_rate": 0.1,  # 10%
            "frequent_retries": 0.3,  # 30% of executions with retries
            "timeout_ratio": 0.05,  # 5% timeout rate
        }
        self._max_metrics_per_hook = 1000
        self._max_analysis_history = 100

    async def monitor_hook_execution(
        self, 
        hook_name: str, 
        execution_time: float,
        success: bool,
        retry_count: int = 0,
        error_type: Optional[str] = None
    ) -> None:
        """
        Monitor individual hook execution performance.
        
        Args:
            hook_name: Name of the hook that was executed
            execution_time: Time taken to execute the hook
            success: Whether the execution was successful
            retry_count: Number of retries performed
            error_type: Type of error if execution failed
        """
        if hook_name not in self.execution_metrics:
            self.execution_metrics[hook_name] = []
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "execution_time": execution_time,
            "success": success,
            "retry_count": retry_count,
            "error_type": error_type,
            "was_timeout": error_type == "timeout" if error_type else False
        }
        
        self.execution_metrics[hook_name].append(metrics)
        
        # Limit metrics per hook to prevent memory bloat
        if len(self.execution_metrics[hook_name]) > self._max_metrics_per_hook:
            self.execution_metrics[hook_name] = self.execution_metrics[hook_name][-self._max_metrics_per_hook:]
        
        # Real-time issue detection for critical problems
        await self._check_real_time_issues(hook_name, metrics)

    async def _check_real_time_issues(
        self, 
        hook_name: str, 
        current_metrics: Dict[str, Any]
    ) -> None:
        """
        Check for real-time performance issues that need immediate attention.
        
        Args:
            hook_name: Name of the hook
            current_metrics: Current execution metrics
        """
        # Check for extremely slow execution
        if current_metrics["execution_time"] > self._thresholds["slow_execution"] * 2:
            issue = PerformanceIssue(
                issue_type=PerformanceIssueType.SLOW_EXECUTION,
                hook_name=hook_name,
                severity="high",
                description=f"Hook execution extremely slow: {current_metrics['execution_time']:.2f}s",
                impact="May cause significant delays in agent coordination",
                recommendation="Investigate hook implementation for optimization opportunities",
                metric_value=current_metrics["execution_time"],
                threshold=self._thresholds["slow_execution"] * 2
            )
            self.performance_alerts.append(issue)
            logger.warning("Performance alert: %s", issue.description)
        
        # Check for repeated failures
        recent_metrics = self.execution_metrics[hook_name][-5:]  # Last 5 executions
        if len(recent_metrics) >= 3:
            failure_rate = sum(1 for m in recent_metrics if not m["success"]) / len(recent_metrics)
            if failure_rate >= 0.6:  # 60% failure rate in recent executions
                issue = PerformanceIssue(
                    issue_type=PerformanceIssueType.HIGH_FAILURE_RATE,
                    hook_name=hook_name,
                    severity="critical",
                    description=f"High failure rate detected: {failure_rate:.1%} in recent executions",
                    impact="Hook unreliability may compromise agent coordination",
                    recommendation="Investigate error patterns and improve error handling",
                    metric_value=failure_rate,
                    threshold=0.6
                )
                self.performance_alerts.append(issue)
                logger.error("Critical performance alert: %s", issue.description)

    async def analyze_hook_performance(self, timeframe: str = "24h") -> PerformanceAnalysis:
        """
        Analyze hook performance over specified timeframe.
        
        Args:
            timeframe: Time period for analysis (e.g., "1h", "24h", "7d")
            
        Returns:
            Comprehensive performance analysis
        """
        # Parse timeframe
        timeframe_hours = self._parse_timeframe(timeframe)
        cutoff_time = datetime.utcnow() - timedelta(hours=timeframe_hours)
        
        # Analyze each hook's performance
        hook_performance = {}
        all_issues = []
        
        for hook_name, metrics_list in self.execution_metrics.items():
            # Filter metrics by timeframe
            filtered_metrics = [
                m for m in metrics_list
                if datetime.fromisoformat(m["timestamp"]) >= cutoff_time
            ]
            
            if not filtered_metrics:
                continue
            
            # Calculate performance metrics
            performance_stats = self._calculate_hook_statistics(filtered_metrics)
            hook_performance[hook_name] = performance_stats
            
            # Detect issues for this hook
            hook_issues = self._detect_performance_issues(hook_name, performance_stats)
            all_issues.extend(hook_issues)
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(hook_performance)
        
        # Generate optimization recommendations
        recommendations = await self.optimize_hook_execution()
        
        analysis = PerformanceAnalysis(
            timeframe=timeframe,
            total_executions=sum(len(metrics) for metrics in self.execution_metrics.values()),
            analysis_timestamp=datetime.utcnow(),
            hook_performance=hook_performance,
            issues=all_issues,
            recommendations=recommendations,
            summary_metrics=summary_metrics
        )
        
        # Store in analysis history
        self.analysis_history.append(analysis)
        if len(self.analysis_history) > self._max_analysis_history:
            self.analysis_history = self.analysis_history[-self._max_analysis_history:]
        
        return analysis

    def _parse_timeframe(self, timeframe: str) -> float:
        """
        Parse timeframe string to hours.
        
        Args:
            timeframe: Timeframe string (e.g., "1h", "24h", "7d")
            
        Returns:
            Number of hours as float
        """
        timeframe = timeframe.lower().strip()
        
        if timeframe.endswith('h'):
            return float(timeframe[:-1])
        elif timeframe.endswith('d'):
            return float(timeframe[:-1]) * 24
        elif timeframe.endswith('m'):  # minutes
            return float(timeframe[:-1]) / 60
        else:
            # Default to hours if no unit
            try:
                return float(timeframe)
            except ValueError:
                logger.warning("Invalid timeframe format: %s, defaulting to 24h", timeframe)
                return 24.0

    def _calculate_hook_statistics(
        self, 
        metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for a hook.
        
        Args:
            metrics: List of execution metrics
            
        Returns:
            Dictionary containing hook statistics
        """
        if not metrics:
            return {}
        
        execution_times = [m["execution_time"] for m in metrics]
        successes = [m["success"] for m in metrics]
        retry_counts = [m["retry_count"] for m in metrics]
        timeouts = [m.get("was_timeout", False) for m in metrics]
        
        stats = {
            "total_executions": len(metrics),
            "success_rate": sum(successes) / len(successes),
            "failure_rate": 1 - (sum(successes) / len(successes)),
            "average_execution_time": statistics.mean(execution_times),
            "median_execution_time": statistics.median(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "total_retry_count": sum(retry_counts),
            "average_retries": statistics.mean(retry_counts),
            "timeout_rate": sum(timeouts) / len(timeouts),
            "executions_with_retries": sum(1 for r in retry_counts if r > 0),
            "retry_percentage": sum(1 for r in retry_counts if r > 0) / len(retry_counts)
        }
        
        # Add percentiles if we have enough data
        if len(execution_times) >= 10:
            sorted_times = sorted(execution_times)
            stats.update({
                "p50_execution_time": statistics.median(sorted_times),
                "p90_execution_time": self._percentile(sorted_times, 0.9),
                "p95_execution_time": self._percentile(sorted_times, 0.95),
                "p99_execution_time": self._percentile(sorted_times, 0.99)
            })
        
        # Calculate execution time standard deviation if we have enough data
        if len(execution_times) >= 2:
            stats["execution_time_stddev"] = statistics.stdev(execution_times)
        
        return stats

    def _percentile(self, sorted_data: List[float], percentile: float) -> float:
        """Calculate percentile from sorted data."""
        if not sorted_data:
            return 0.0
        
        index = int(percentile * len(sorted_data))
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        
        return sorted_data[index]

    def _detect_performance_issues(
        self, 
        hook_name: str, 
        stats: Dict[str, Any]
    ) -> List[PerformanceIssue]:
        """
        Detect performance issues from hook statistics.
        
        Args:
            hook_name: Name of the hook
            stats: Hook performance statistics
            
        Returns:
            List of detected performance issues
        """
        issues = []
        
        # Check for slow execution
        if stats.get("average_execution_time", 0) > self._thresholds["slow_execution"]:
            severity = "high" if stats["average_execution_time"] > self._thresholds["slow_execution"] * 2 else "medium"
            issues.append(PerformanceIssue(
                issue_type=PerformanceIssueType.SLOW_EXECUTION,
                hook_name=hook_name,
                severity=severity,
                description=f"Average execution time {stats['average_execution_time']:.2f}s exceeds threshold",
                impact="Increased latency in agent coordination workflow",
                recommendation="Profile hook execution and optimize bottlenecks",
                metric_value=stats["average_execution_time"],
                threshold=self._thresholds["slow_execution"]
            ))
        
        # Check for high failure rate
        if stats.get("failure_rate", 0) > self._thresholds["high_failure_rate"]:
            severity = "critical" if stats["failure_rate"] > 0.2 else "high"
            issues.append(PerformanceIssue(
                issue_type=PerformanceIssueType.HIGH_FAILURE_RATE,
                hook_name=hook_name,
                severity=severity,
                description=f"Failure rate {stats['failure_rate']:.1%} exceeds acceptable threshold",
                impact="Reduced reliability of agent coordination",
                recommendation="Investigate error patterns and improve error handling",
                metric_value=stats["failure_rate"],
                threshold=self._thresholds["high_failure_rate"]
            ))
        
        # Check for frequent retries
        if stats.get("retry_percentage", 0) > self._thresholds["frequent_retries"]:
            issues.append(PerformanceIssue(
                issue_type=PerformanceIssueType.FREQUENT_RETRIES,
                hook_name=hook_name,
                severity="medium",
                description=f"Retry rate {stats['retry_percentage']:.1%} indicates reliability issues",
                impact="Increased execution time and resource usage",
                recommendation="Identify root causes of failures and improve hook stability",
                metric_value=stats["retry_percentage"],
                threshold=self._thresholds["frequent_retries"]
            ))
        
        # Check for timeout issues
        if stats.get("timeout_rate", 0) > self._thresholds["timeout_ratio"]:
            issues.append(PerformanceIssue(
                issue_type=PerformanceIssueType.TIMEOUT_ISSUES,
                hook_name=hook_name,
                severity="high",
                description=f"Timeout rate {stats['timeout_rate']:.1%} indicates performance problems",
                impact="Hook executions being cancelled, affecting reliability",
                recommendation="Increase timeout values or optimize hook performance",
                metric_value=stats["timeout_rate"],
                threshold=self._thresholds["timeout_ratio"]
            ))
        
        return issues

    def _calculate_summary_metrics(
        self, 
        hook_performance: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate summary metrics across all hooks.
        
        Args:
            hook_performance: Performance data for all hooks
            
        Returns:
            Summary metrics dictionary
        """
        if not hook_performance:
            return {}
        
        all_success_rates = [stats.get("success_rate", 0) for stats in hook_performance.values()]
        all_execution_times = [stats.get("average_execution_time", 0) for stats in hook_performance.values()]
        total_executions = sum(stats.get("total_executions", 0) for stats in hook_performance.values())
        
        return {
            "total_hooks_analyzed": len(hook_performance),
            "overall_success_rate": statistics.mean(all_success_rates) if all_success_rates else 0,
            "average_hook_execution_time": statistics.mean(all_execution_times) if all_execution_times else 0,
            "total_executions_analyzed": total_executions,
            "fastest_hook": min(hook_performance.items(), key=lambda x: x[1].get("average_execution_time", float('inf')))[0] if hook_performance else None,
            "slowest_hook": max(hook_performance.items(), key=lambda x: x[1].get("average_execution_time", 0))[0] if hook_performance else None,
            "most_reliable_hook": max(hook_performance.items(), key=lambda x: x[1].get("success_rate", 0))[0] if hook_performance else None,
            "least_reliable_hook": min(hook_performance.items(), key=lambda x: x[1].get("success_rate", 1))[0] if hook_performance else None
        }

    async def optimize_hook_execution(self) -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations for hook execution.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        for hook_name, metrics_list in self.execution_metrics.items():
            if not metrics_list:
                continue
            
            recent_metrics = metrics_list[-100:]  # Analyze recent executions
            stats = self._calculate_hook_statistics(recent_metrics)
            
            # Recommendation: Optimize slow hooks
            if stats.get("average_execution_time", 0) > self._thresholds["slow_execution"]:
                recommendations.append(OptimizationRecommendation(
                    hook_name=hook_name,
                    recommendation_type="performance_optimization",
                    description="Optimize hook execution performance",
                    expected_improvement=f"Reduce execution time from {stats['average_execution_time']:.2f}s to target <{self._thresholds['slow_execution']}s",
                    implementation_effort="medium",
                    priority="high",
                    technical_details="Profile hook code, optimize database queries, reduce I/O operations, implement caching",
                    estimated_impact={
                        "time_savings": f"{stats['average_execution_time'] - self._thresholds['slow_execution']:.2f}s per execution",
                        "affected_executions": stats["total_executions"]
                    }
                ))
            
            # Recommendation: Improve reliability
            if stats.get("failure_rate", 0) > self._thresholds["high_failure_rate"]:
                recommendations.append(OptimizationRecommendation(
                    hook_name=hook_name,
                    recommendation_type="reliability_improvement",
                    description="Improve hook reliability and error handling",
                    expected_improvement=f"Reduce failure rate from {stats['failure_rate']:.1%} to target <{self._thresholds['high_failure_rate']:.1%}",
                    implementation_effort="high",
                    priority="critical",
                    technical_details="Add comprehensive error handling, implement circuit breaker pattern, improve input validation",
                    estimated_impact={
                        "reliability_improvement": f"{stats['failure_rate'] - self._thresholds['high_failure_rate']:.1%}",
                        "failed_executions_prevented": int(stats["total_executions"] * (stats["failure_rate"] - self._thresholds["high_failure_rate"]))
                    }
                ))
            
            # Recommendation: Optimize timeout settings
            if stats.get("timeout_rate", 0) > 0:
                recommendations.append(OptimizationRecommendation(
                    hook_name=hook_name,
                    recommendation_type="timeout_optimization",
                    description="Optimize hook timeout settings",
                    expected_improvement="Balance between performance and reliability",
                    implementation_effort="low",
                    priority="medium",
                    technical_details=f"Consider increasing timeout from current value based on p95 execution time: {stats.get('p95_execution_time', 'N/A')}",
                    estimated_impact={
                        "timeout_rate_reduction": f"{stats['timeout_rate']:.1%}",
                        "recommended_timeout": f"{stats.get('p95_execution_time', 30) * 1.2:.1f}s"
                    }
                ))
        
        # Sort recommendations by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 4))
        
        return recommendations

    def get_performance_alerts(
        self, 
        severity_filter: Optional[str] = None,
        max_age_hours: Optional[int] = None
    ) -> List[PerformanceIssue]:
        """
        Get current performance alerts with optional filtering.
        
        Args:
            severity_filter: Filter by severity level
            max_age_hours: Only return alerts within this age
            
        Returns:
            List of performance alerts
        """
        alerts = self.performance_alerts.copy()
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]
        
        if max_age_hours is not None:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            alerts = [alert for alert in alerts if alert.detected_at >= cutoff_time]
        
        return alerts

    def clear_old_alerts(self, max_age_hours: int = 24) -> int:
        """
        Clear old performance alerts.
        
        Args:
            max_age_hours: Maximum age of alerts to keep
            
        Returns:
            Number of alerts cleared
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        original_count = len(self.performance_alerts)
        
        self.performance_alerts = [
            alert for alert in self.performance_alerts
            if alert.detected_at >= cutoff_time
        ]
        
        return original_count - len(self.performance_alerts)

    def get_monitor_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring statistics.
        
        Returns:
            Dictionary containing monitoring statistics
        """
        total_metrics = sum(len(metrics) for metrics in self.execution_metrics.values())
        
        return {
            "monitored_hooks": len(self.execution_metrics),
            "total_metrics_collected": total_metrics,
            "active_alerts": len(self.performance_alerts),
            "analysis_history_count": len(self.analysis_history),
            "alert_severities": {
                severity: len([a for a in self.performance_alerts if a.severity == severity])
                for severity in ["low", "medium", "high", "critical"]
            },
            "last_analysis": self.analysis_history[-1].analysis_timestamp.isoformat() if self.analysis_history else None,
            "monitoring_thresholds": self._thresholds.copy()
        }