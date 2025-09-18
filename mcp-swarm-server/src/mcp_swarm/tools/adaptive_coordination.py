"""
Adaptive Coordination Engine for Dynamic Strategy Management

This module provides real-time performance monitoring, strategy adaptation,
and continuous improvement capabilities for multi-agent coordination,
enabling the system to learn and evolve its coordination approaches.
"""

from typing import Dict, List, Optional, Any, Callable
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from .coordination_strategies import BaseCoordinationStrategy, Task, Agent
from .strategy_selector import StrategySelector, StrategyRecommendation

# Set up logging
logger = logging.getLogger(__name__)


class AdaptationTrigger(Enum):
    """Triggers for strategy adaptation."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    TIMELINE_SLIPPAGE = "timeline_slippage"
    RESOURCE_SHORTAGE = "resource_shortage"
    QUALITY_ISSUES = "quality_issues"
    COORDINATION_FAILURE = "coordination_failure"
    EXTERNAL_CHANGE = "external_change"


class MetricType(Enum):
    """Types of performance metrics."""
    THROUGHPUT = "throughput"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    COLLABORATION = "collaboration"
    TIMELINE = "timeline"
    RESOURCE_UTILIZATION = "resource_utilization"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    type: MetricType
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    baseline: Optional[float] = None
    target: Optional[float] = None


@dataclass
class AdaptationEvent:
    """Record of a strategy adaptation event."""
    timestamp: datetime
    trigger: AdaptationTrigger
    previous_strategy: str
    new_strategy: str
    performance_before: Dict[str, float]
    performance_after: Optional[Dict[str, float]] = None
    reasoning: List[str] = field(default_factory=list)
    success: Optional[bool] = None


@dataclass
class ExecutionContext:
    """Current execution context for monitoring."""
    strategy: BaseCoordinationStrategy
    tasks: List[Task]
    agents: List[Agent]
    start_time: datetime
    metrics: List[PerformanceMetric] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    adaptations: List[AdaptationEvent] = field(default_factory=list)


class PerformanceTracker:
    """Tracks and analyzes performance metrics."""
    
    def __init__(self):
        self.metrics_history: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=100))
        self.baselines: Dict[MetricType, float] = {}
        self.targets: Dict[MetricType, float] = {}
        self.alert_thresholds: Dict[MetricType, Dict[str, float]] = {}
        
        self._initialize_baselines_and_targets()
    
    def _initialize_baselines_and_targets(self):
        """Initialize baseline values and targets for metrics."""
        self.baselines = {
            MetricType.THROUGHPUT: 1.0,
            MetricType.QUALITY: 0.8,
            MetricType.EFFICIENCY: 0.7,
            MetricType.COLLABORATION: 0.6,
            MetricType.TIMELINE: 1.0,
            MetricType.RESOURCE_UTILIZATION: 0.7
        }
        
        self.targets = {
            MetricType.THROUGHPUT: 1.2,
            MetricType.QUALITY: 0.9,
            MetricType.EFFICIENCY: 0.85,
            MetricType.COLLABORATION: 0.8,
            MetricType.TIMELINE: 0.9,  # Under budget
            MetricType.RESOURCE_UTILIZATION: 0.8
        }
        
        self.alert_thresholds = {
            MetricType.THROUGHPUT: {"warning": 0.8, "critical": 0.6},
            MetricType.QUALITY: {"warning": 0.7, "critical": 0.5},
            MetricType.EFFICIENCY: {"warning": 0.6, "critical": 0.4},
            MetricType.COLLABORATION: {"warning": 0.5, "critical": 0.3},
            MetricType.TIMELINE: {"warning": 1.1, "critical": 1.3},  # Over budget
            MetricType.RESOURCE_UTILIZATION: {"warning": 0.9, "critical": 0.95}
        }
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        metric.baseline = self.baselines.get(metric.type)
        metric.target = self.targets.get(metric.type)
        
        self.metrics_history[metric.type].append(metric)
        
        logger.debug("Recorded metric %s: %f (baseline: %f, target: %f)", 
                    metric.type.value, metric.value, 
                    metric.baseline or 0, metric.target or 0)
    
    def get_recent_performance(self, metric_type: MetricType, window_minutes: int = 10) -> List[PerformanceMetric]:
        """Get recent performance metrics within a time window."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        return [
            metric for metric in self.metrics_history[metric_type]
            if metric.timestamp >= cutoff_time
        ]
    
    def calculate_trend(self, metric_type: MetricType, window_minutes: int = 30) -> Optional[float]:
        """Calculate performance trend (positive = improving, negative = degrading)."""
        recent_metrics = self.get_recent_performance(metric_type, window_minutes)
        
        if len(recent_metrics) < 3:
            return None
        
        values = [m.value for m in recent_metrics]
        times = [(m.timestamp - recent_metrics[0].timestamp).total_seconds() for m in recent_metrics]
        
        # Simple linear regression for trend
        if len(times) > 1:
            slope = np.corrcoef(times, values)[0, 1] if np.var(times) > 0 else 0
            return float(slope)
        
        return 0.0
    
    def detect_performance_issues(self) -> List[str]:
        """Detect performance issues based on thresholds and trends."""
        issues = []
        
        for metric_type in MetricType:
            recent_metrics = self.get_recent_performance(metric_type, 5)
            
            if not recent_metrics:
                continue
            
            avg_recent = sum(m.value for m in recent_metrics) / len(recent_metrics)
            
            thresholds = self.alert_thresholds.get(metric_type, {})
            
            # Check critical thresholds
            if "critical" in thresholds:
                if (metric_type == MetricType.TIMELINE and avg_recent >= thresholds["critical"]) or \
                   (metric_type != MetricType.TIMELINE and avg_recent <= thresholds["critical"]):
                    issues.append(f"CRITICAL: {metric_type.value} at {avg_recent:.2f} (threshold: {thresholds['critical']:.2f})")
            
            # Check warning thresholds
            elif "warning" in thresholds:
                if (metric_type == MetricType.TIMELINE and avg_recent >= thresholds["warning"]) or \
                   (metric_type != MetricType.TIMELINE and avg_recent <= thresholds["warning"]):
                    issues.append(f"WARNING: {metric_type.value} at {avg_recent:.2f} (threshold: {thresholds['warning']:.2f})")
            
            # Check trends
            trend = self.calculate_trend(metric_type, 15)
            if trend is not None:
                if (metric_type == MetricType.TIMELINE and trend > 0.1) or \
                   (metric_type != MetricType.TIMELINE and trend < -0.1):
                    issues.append(f"TREND: {metric_type.value} degrading (slope: {trend:.3f})")
        
        return issues


class AdaptiveCoordinationEngine:
    """Adaptive coordination engine with real-time strategy adaptation."""
    
    def __init__(self, swarm_coordinator=None):
        self.swarm_coordinator = swarm_coordinator
        self.performance_tracker = PerformanceTracker()
        self.strategy_selector = StrategySelector(swarm_coordinator)
        
        self.current_context: Optional[ExecutionContext] = None
        self.adaptation_history: List[AdaptationEvent] = []
        self.adaptation_rules: List[Callable] = []
        self.monitoring_enabled = True
        self.adaptation_enabled = True
        
        self._initialize_adaptation_rules()
    
    def _initialize_adaptation_rules(self):
        """Initialize adaptation rules."""
        self.adaptation_rules = [
            self._rule_timeline_slippage,
            self._rule_quality_degradation,
            self._rule_resource_shortage,
            self._rule_coordination_overhead,
            self._rule_performance_optimization
        ]
    
    async def start_execution_monitoring(
        self, 
        strategy: BaseCoordinationStrategy,
        tasks: List[Task],
        agents: List[Agent]
    ) -> ExecutionContext:
        """Start monitoring an execution context."""
        
        self.current_context = ExecutionContext(
            strategy=strategy,
            tasks=tasks,
            agents=agents,
            start_time=datetime.now()
        )
        
        logger.info("Started execution monitoring for strategy: %s", strategy.name)
        
        if self.monitoring_enabled:
            # Start background monitoring
            asyncio.create_task(self._monitoring_loop())
        
        return self.current_context
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.current_context and self.monitoring_enabled:
            try:
                # Collect performance metrics
                await self._collect_performance_metrics()
                
                # Check for adaptation triggers
                if self.adaptation_enabled:
                    await self._check_adaptation_triggers()
                
                # Wait before next monitoring cycle
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error("Monitoring loop error: %s", str(e))
                await asyncio.sleep(60)  # Longer wait on error
    
    async def _collect_performance_metrics(self):
        """Collect current performance metrics."""
        if not self.current_context:
            return
        
        current_time = datetime.now()
        elapsed_time = (current_time - self.current_context.start_time).total_seconds()
        
        # Throughput metric (tasks completed per hour) - simulated for now
        completed_tasks = len([t for t in self.current_context.tasks if getattr(t, 'completed', False)])
        throughput = (completed_tasks / max(elapsed_time / 3600, 0.1))
        
        self.performance_tracker.record_metric(PerformanceMetric(
            type=MetricType.THROUGHPUT,
            value=throughput,
            timestamp=current_time,
            context={"completed_tasks": completed_tasks, "elapsed_hours": elapsed_time / 3600}
        ))
        
        # Resource utilization metric
        avg_utilization = 0.0
        if self.current_context.agents:
            avg_utilization = sum(agent.current_load for agent in self.current_context.agents) / len(self.current_context.agents)
            
            self.performance_tracker.record_metric(PerformanceMetric(
                type=MetricType.RESOURCE_UTILIZATION,
                value=avg_utilization,
                timestamp=current_time,
                context={"agent_count": len(self.current_context.agents)}
            ))
        
        # Timeline metric (actual vs estimated progress)
        estimated_progress = elapsed_time / (sum(task.estimated_duration for task in self.current_context.tasks) * 60)
        actual_progress = completed_tasks / max(len(self.current_context.tasks), 1)
        timeline_ratio = estimated_progress / max(actual_progress, 0.01)
        
        self.performance_tracker.record_metric(PerformanceMetric(
            type=MetricType.TIMELINE,
            value=timeline_ratio,
            timestamp=current_time,
            context={"estimated_progress": estimated_progress, "actual_progress": actual_progress}
        ))
        
        # Quality metric (simulated based on agent performance)
        if self.current_context.agents:
            avg_agent_performance = sum(
                agent.performance_history.get('success_rate', 0.8) 
                for agent in self.current_context.agents
            ) / len(self.current_context.agents)
            
            self.performance_tracker.record_metric(PerformanceMetric(
                type=MetricType.QUALITY,
                value=avg_agent_performance,
                timestamp=current_time,
                context={"agent_performance_avg": avg_agent_performance}
            ))
        
        # Efficiency metric (throughput / resource_utilization)
        efficiency = throughput / max(avg_utilization, 0.1)
        
        self.performance_tracker.record_metric(PerformanceMetric(
            type=MetricType.EFFICIENCY,
            value=efficiency,
            timestamp=current_time,
            context={"throughput": throughput, "utilization": avg_utilization}
        ))
    
    async def _check_adaptation_triggers(self):
        """Check if strategy adaptation is needed."""
        if not self.current_context:
            return
        
        # Detect performance issues
        issues = self.performance_tracker.detect_performance_issues()
        
        if issues:
            self.current_context.issues.extend(issues)
            logger.warning("Performance issues detected: %s", issues)
        
        # Apply adaptation rules
        for rule in self.adaptation_rules:
            try:
                adaptation_needed = await rule()
                if adaptation_needed:
                    await self._trigger_adaptation(adaptation_needed)
                    break  # Only one adaptation at a time
            except Exception as e:
                logger.error("Adaptation rule failed: %s", str(e))
    
    async def _rule_timeline_slippage(self) -> Optional[AdaptationTrigger]:
        """Rule: Adapt if timeline is slipping significantly."""
        timeline_metrics = self.performance_tracker.get_recent_performance(MetricType.TIMELINE, 10)
        
        if not timeline_metrics:
            return None
        
        avg_timeline_ratio = sum(m.value for m in timeline_metrics) / len(timeline_metrics)
        
        if avg_timeline_ratio > 1.2:  # 20% behind schedule
            logger.info("Timeline slippage detected: %f", avg_timeline_ratio)
            return AdaptationTrigger.TIMELINE_SLIPPAGE
        
        return None
    
    async def _rule_quality_degradation(self) -> Optional[AdaptationTrigger]:
        """Rule: Adapt if quality is degrading."""
        quality_trend = self.performance_tracker.calculate_trend(MetricType.QUALITY, 15)
        
        if quality_trend is not None and quality_trend < -0.2:
            logger.info("Quality degradation detected: trend %f", quality_trend)
            return AdaptationTrigger.QUALITY_ISSUES
        
        return None
    
    async def _rule_resource_shortage(self) -> Optional[AdaptationTrigger]:
        """Rule: Adapt if resource utilization is too high."""
        utilization_metrics = self.performance_tracker.get_recent_performance(MetricType.RESOURCE_UTILIZATION, 5)
        
        if not utilization_metrics:
            return None
        
        avg_utilization = sum(m.value for m in utilization_metrics) / len(utilization_metrics)
        
        if avg_utilization > 0.9:  # 90% utilization
            logger.info("High resource utilization detected: %f", avg_utilization)
            return AdaptationTrigger.RESOURCE_SHORTAGE
        
        return None
    
    async def _rule_coordination_overhead(self) -> Optional[AdaptationTrigger]:
        """Rule: Adapt if coordination overhead is too high."""
        efficiency_trend = self.performance_tracker.calculate_trend(MetricType.EFFICIENCY, 20)
        
        if efficiency_trend is not None and efficiency_trend < -0.15:
            logger.info("Coordination overhead detected: efficiency trend %f", efficiency_trend)
            return AdaptationTrigger.COORDINATION_FAILURE
        
        return None
    
    async def _rule_performance_optimization(self) -> Optional[AdaptationTrigger]:
        """Rule: Adapt if there's an opportunity for performance optimization."""
        # Check if all metrics are stable and we might benefit from a different strategy
        all_stable = True
        
        for metric_type in [MetricType.THROUGHPUT, MetricType.QUALITY, MetricType.EFFICIENCY]:
            trend = self.performance_tracker.calculate_trend(metric_type, 20)
            if trend is None or abs(trend) > 0.1:
                all_stable = False
                break
        
        if all_stable:
            # Check if we're underperforming targets
            recent_efficiency = self.performance_tracker.get_recent_performance(MetricType.EFFICIENCY, 5)
            if recent_efficiency:
                avg_efficiency = sum(m.value for m in recent_efficiency) / len(recent_efficiency)
                target_efficiency = self.performance_tracker.targets[MetricType.EFFICIENCY]
                
                if avg_efficiency < target_efficiency * 0.8:  # Significantly below target
                    logger.info("Performance optimization opportunity detected")
                    return AdaptationTrigger.PERFORMANCE_DEGRADATION
        
        return None
    
    async def _trigger_adaptation(self, trigger: AdaptationTrigger):
        """Trigger strategy adaptation."""
        if not self.current_context:
            return
        
        logger.info("Triggering adaptation for: %s", trigger.value)
        
        # Capture current performance
        current_performance = self._get_current_performance_snapshot()
        
        # Get new strategy recommendation
        try:
            recommendation = await self.strategy_selector.select_optimal_strategy(
                self.current_context.tasks,
                self.current_context.agents,
                {
                    "adaptation_trigger": trigger.value,
                    "current_performance": current_performance,
                    "issues": self.current_context.issues[-5:],  # Recent issues
                    "urgency": "high"
                }
            )
            
            if recommendation.primary_strategy.name != self.current_context.strategy.name:
                await self._execute_adaptation(trigger, recommendation, current_performance)
            else:
                logger.info("Adaptation analysis suggests keeping current strategy")
                
        except Exception as e:
            logger.error("Adaptation failed: %s", str(e))
    
    async def _execute_adaptation(
        self, 
        trigger: AdaptationTrigger,
        recommendation: StrategyRecommendation,
        current_performance: Dict[str, float]
    ):
        """Execute the strategy adaptation."""
        if not self.current_context:
            return
        
        previous_strategy = self.current_context.strategy.name
        new_strategy = recommendation.primary_strategy
        
        # Record adaptation event
        adaptation_event = AdaptationEvent(
            timestamp=datetime.now(),
            trigger=trigger,
            previous_strategy=previous_strategy,
            new_strategy=new_strategy.name,
            performance_before=current_performance,
            reasoning=recommendation.reasoning
        )
        
        # Update current context
        self.current_context.strategy = new_strategy
        self.current_context.adaptations.append(adaptation_event)
        self.adaptation_history.append(adaptation_event)
        
        logger.info("Adapted strategy from %s to %s (trigger: %s)", 
                   previous_strategy, new_strategy.name, trigger.value)
        
        # Schedule performance evaluation
        asyncio.create_task(self._evaluate_adaptation_success(adaptation_event))
    
    async def _evaluate_adaptation_success(self, adaptation_event: AdaptationEvent):
        """Evaluate the success of an adaptation after some time."""
        # Wait for adaptation to take effect
        await asyncio.sleep(300)  # 5 minutes
        
        try:
            # Capture post-adaptation performance
            post_performance = self._get_current_performance_snapshot()
            adaptation_event.performance_after = post_performance
            
            # Evaluate success
            success = self._is_adaptation_successful(
                adaptation_event.performance_before,
                post_performance,
                adaptation_event.trigger
            )
            
            adaptation_event.success = success
            
            logger.info("Adaptation evaluation: %s (trigger: %s)", 
                       "SUCCESS" if success else "FAILED", 
                       adaptation_event.trigger.value)
            
            # Update strategy selector with feedback
            if hasattr(self.strategy_selector, 'update_performance_feedback'):
                self.strategy_selector.update_performance_feedback(
                    adaptation_event.new_strategy,
                    {
                        "performance_score": self._calculate_overall_performance_score(post_performance),
                        "adaptation_success": success,
                        "trigger": adaptation_event.trigger.value
                    }
                )
                
        except Exception as e:
            logger.error("Adaptation evaluation failed: %s", str(e))
    
    def _get_current_performance_snapshot(self) -> Dict[str, float]:
        """Get current performance snapshot."""
        snapshot = {}
        
        for metric_type in MetricType:
            recent_metrics = self.performance_tracker.get_recent_performance(metric_type, 5)
            if recent_metrics:
                snapshot[metric_type.value] = sum(m.value for m in recent_metrics) / len(recent_metrics)
        
        return snapshot
    
    def _is_adaptation_successful(
        self, 
        before: Dict[str, float], 
        after: Dict[str, float], 
        trigger: AdaptationTrigger
    ) -> bool:
        """Determine if adaptation was successful."""
        
        # Define success criteria based on trigger
        if trigger == AdaptationTrigger.TIMELINE_SLIPPAGE:
            timeline_before = before.get('timeline', 1.0)
            timeline_after = after.get('timeline', 1.0)
            return timeline_after < timeline_before  # Improvement
        
        elif trigger == AdaptationTrigger.QUALITY_ISSUES:
            quality_before = before.get('quality', 0.8)
            quality_after = after.get('quality', 0.8)
            return quality_after > quality_before
        
        elif trigger == AdaptationTrigger.RESOURCE_SHORTAGE:
            utilization_before = before.get('resource_utilization', 0.7)
            utilization_after = after.get('resource_utilization', 0.7)
            return utilization_after < utilization_before * 1.1  # Some improvement
        
        elif trigger == AdaptationTrigger.COORDINATION_FAILURE:
            efficiency_before = before.get('efficiency', 0.7)
            efficiency_after = after.get('efficiency', 0.7)
            return efficiency_after > efficiency_before
        
        elif trigger == AdaptationTrigger.PERFORMANCE_DEGRADATION:
            # Overall performance improvement
            overall_before = self._calculate_overall_performance_score(before)
            overall_after = self._calculate_overall_performance_score(after)
            return overall_after > overall_before * 1.05  # 5% improvement
        
        # Default: general improvement check
        overall_before = self._calculate_overall_performance_score(before)
        overall_after = self._calculate_overall_performance_score(after)
        return overall_after > overall_before
    
    def _calculate_overall_performance_score(self, performance: Dict[str, float]) -> float:
        """Calculate overall performance score."""
        weights = {
            'throughput': 0.25,
            'quality': 0.25,
            'efficiency': 0.20,
            'timeline': 0.20,  # Inverse for timeline (lower is better)
            'resource_utilization': 0.10
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in performance:
                value = performance[metric]
                
                # Normalize metrics to 0-1 scale (higher is better)
                if metric == 'timeline':
                    normalized_value = max(0, 2.0 - value)  # Invert timeline (1.0 becomes 1.0, 2.0 becomes 0)
                else:
                    normalized_value = min(value, 1.0)
                
                score += normalized_value * weight
                total_weight += weight
        
        return score / max(total_weight, 0.01)
    
    async def adapt_strategy_realtime(
        self, 
        current_strategy: BaseCoordinationStrategy,
        execution_context: Dict[str, Any]
    ) -> Optional[BaseCoordinationStrategy]:
        """Adapt coordination strategy during execution (public interface)."""
        
        # Check if adaptation is needed based on current context
        issues = execution_context.get('issues', [])
        performance_metrics = execution_context.get('performance_metrics', {})
        
        # Simple adaptation logic based on issues
        adaptation_needed = False
        trigger = None
        
        if 'timeline_slippage' in str(issues):
            adaptation_needed = True
            trigger = AdaptationTrigger.TIMELINE_SLIPPAGE
        elif 'quality_degradation' in str(issues):
            adaptation_needed = True
            trigger = AdaptationTrigger.QUALITY_ISSUES
        elif 'resource_shortage' in str(issues):
            adaptation_needed = True
            trigger = AdaptationTrigger.RESOURCE_SHORTAGE
        
        if not adaptation_needed:
            return None
        
        # Get alternative strategy
        try:
            tasks = execution_context.get('tasks', [])
            agents = execution_context.get('agents', [])
            
            recommendation = await self.strategy_selector.select_optimal_strategy(
                tasks, agents, {
                    "adaptation_trigger": trigger.value if trigger else "manual",
                    "current_strategy": current_strategy.name,
                    "issues": issues
                }
            )
            
            if recommendation.primary_strategy.name != current_strategy.name:
                logger.info("Real-time adaptation: %s -> %s", 
                           current_strategy.name, recommendation.primary_strategy.name)
                return recommendation.primary_strategy
            
        except Exception as e:
            logger.error("Real-time adaptation failed: %s", str(e))
        
        return None
    
    def stop_monitoring(self):
        """Stop execution monitoring."""
        self.monitoring_enabled = False
        
        if self.current_context:
            execution_time = (datetime.now() - self.current_context.start_time).total_seconds()
            logger.info("Stopped monitoring after %d seconds", int(execution_time))
            
            # Final performance summary
            final_performance = self._get_current_performance_snapshot()
            logger.info("Final performance: %s", final_performance)
        
        self.current_context = None
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation events and performance."""
        if not self.adaptation_history:
            return {"message": "No adaptations performed"}
        
        recent_adaptations = self.adaptation_history[-10:]  # Last 10 adaptations
        
        successful_adaptations = [a for a in recent_adaptations if a.success is True]
        failed_adaptations = [a for a in recent_adaptations if a.success is False]
        
        trigger_counts = defaultdict(int)
        for adaptation in recent_adaptations:
            trigger_counts[adaptation.trigger.value] += 1
        
        return {
            "total_adaptations": len(self.adaptation_history),
            "recent_adaptations": len(recent_adaptations),
            "success_rate": len(successful_adaptations) / max(len(recent_adaptations), 1),
            "most_common_triggers": dict(trigger_counts),
            "current_monitoring": self.monitoring_enabled,
            "current_adaptation": self.adaptation_enabled
        }
    
    def enable_adaptation(self, enabled: bool = True):
        """Enable or disable automatic adaptation."""
        self.adaptation_enabled = enabled
        logger.info("Adaptation %s", "enabled" if enabled else "disabled")
    
    def enable_monitoring(self, enabled: bool = True):
        """Enable or disable performance monitoring."""
        self.monitoring_enabled = enabled
        logger.info("Monitoring %s", "enabled" if enabled else "disabled")