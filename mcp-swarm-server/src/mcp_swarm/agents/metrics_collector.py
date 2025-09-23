"""
Performance Metrics Collector for MCP Swarm Intelligence Server

This module provides comprehensive metrics collection, performance analysis,
and trend monitoring for agent ecosystem performance optimization.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import statistics
import logging

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics that can be collected"""
    RESPONSE_TIME = "response_time"
    TASK_COMPLETION = "task_completion"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"
    COORDINATION_EFFICIENCY = "coordination_efficiency"
    LEARNING_PROGRESS = "learning_progress"

@dataclass
class MetricDataPoint:
    """A single metric data point"""
    timestamp: datetime
    value: float
    metric_type: MetricType
    agent_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics for an agent"""
    agent_id: str
    avg_response_time: float
    completion_rate: float
    error_rate: float
    throughput: float
    resource_efficiency: float
    coordination_score: float
    trend_direction: str  # 'improving', 'stable', 'degrading'
    last_updated: datetime

@dataclass
class TrendAnalysis:
    """Trend analysis for a specific metric"""
    metric_type: MetricType
    agent_id: str
    trend_direction: str
    slope: float
    confidence: float
    recent_average: float
    historical_average: float
    prediction: Optional[float]

class PerformanceMetricsCollector:
    """Comprehensive performance metrics collection and analysis system"""
    
    def __init__(self,
                 collection_interval: float = 30.0,
                 retention_days: int = 7,
                 trend_analysis_window: int = 100,
                 alert_threshold_degradation: float = 20.0):
        """
        Initialize the metrics collector
        
        Args:
            collection_interval: Interval between metric collection cycles
            retention_days: Number of days to retain metric data
            trend_analysis_window: Number of data points for trend analysis
            alert_threshold_degradation: Percentage degradation to trigger alerts
        """
        self.collection_interval = collection_interval
        self.retention_days = retention_days
        self.trend_analysis_window = trend_analysis_window
        self.alert_threshold_degradation = alert_threshold_degradation
        
        # Data storage
        self.raw_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated_metrics: Dict[str, PerformanceMetrics] = {}
        self.trend_analyses: Dict[Tuple[str, MetricType], TrendAnalysis] = {}
        
        # Time series data by metric type and agent
        self.time_series: Dict[Tuple[str, MetricType], deque] = defaultdict(
            lambda: deque(maxlen=self.trend_analysis_window * 2)
        )
        
        # Performance baselines
        self.baselines: Dict[Tuple[str, MetricType], float] = {}
        
        # Collection control
        self.is_collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        
        # Alert callbacks
        self.alert_callbacks: List[Any] = []
        
        # Task tracking for throughput calculation
        self.task_starts: Dict[str, List[datetime]] = defaultdict(list)
        self.task_completions: Dict[str, List[datetime]] = defaultdict(list)
        self.task_errors: Dict[str, List[datetime]] = defaultdict(list)

    async def start_collection(self):
        """Start the metrics collection process"""
        if self.is_collecting:
            logger.warning("Metrics collection already started")
            return
            
        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Performance metrics collection started")

    async def stop_collection(self):
        """Stop the metrics collection process"""
        if not self.is_collecting:
            return
            
        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance metrics collection stopped")

    async def _collection_loop(self):
        """Main metrics collection loop"""
        while self.is_collecting:
            try:
                await self._collect_all_metrics()
                await self._perform_trend_analysis()
                await self._cleanup_old_data()
                await asyncio.sleep(self.collection_interval)
            except (asyncio.CancelledError, OSError) as e:
                logger.error("Error in metrics collection loop: %s", e)
                await asyncio.sleep(self.collection_interval)

    async def _collect_all_metrics(self):
        """Collect metrics for all registered agents"""
        current_time = datetime.now()
        
        # Get all unique agent IDs from various sources
        agent_ids = set()
        agent_ids.update(self.task_starts.keys())
        agent_ids.update(self.task_completions.keys())
        agent_ids.update(self.task_errors.keys())
        
        for agent_id in agent_ids:
            try:
                # Collect various metrics for this agent
                await self._collect_agent_metrics(agent_id, current_time)
            except (ValueError, KeyError) as e:
                logger.error("Error collecting metrics for agent %s: %s", agent_id, e)

    async def _collect_agent_metrics(self, agent_id: str, timestamp: datetime):
        """Collect all metrics for a specific agent"""
        # Response time metrics
        response_time = await self._calculate_avg_response_time(agent_id)
        if response_time is not None:
            self.record_metric(agent_id, MetricType.RESPONSE_TIME, response_time, timestamp)
        
        # Task completion rate
        completion_rate = await self._calculate_completion_rate(agent_id)
        if completion_rate is not None:
            self.record_metric(agent_id, MetricType.TASK_COMPLETION, completion_rate, timestamp)
        
        # Error rate
        error_rate = await self._calculate_error_rate(agent_id)
        if error_rate is not None:
            self.record_metric(agent_id, MetricType.ERROR_RATE, error_rate, timestamp)
        
        # Throughput
        throughput = await self._calculate_throughput(agent_id)
        if throughput is not None:
            self.record_metric(agent_id, MetricType.THROUGHPUT, throughput, timestamp)
        
        # Resource usage (would integrate with actual resource monitoring)
        resource_usage = await self._calculate_resource_efficiency(agent_id)
        if resource_usage is not None:
            self.record_metric(agent_id, MetricType.RESOURCE_USAGE, resource_usage, timestamp)
        
        # Coordination efficiency
        coord_efficiency = await self._calculate_coordination_efficiency(agent_id)
        if coord_efficiency is not None:
            self.record_metric(agent_id, MetricType.COORDINATION_EFFICIENCY, coord_efficiency, timestamp)

    async def _calculate_avg_response_time(self, agent_id: str) -> Optional[float]:
        """Calculate average response time for an agent"""
        key = (agent_id, MetricType.RESPONSE_TIME)
        if key not in self.time_series or not self.time_series[key]:
            return None
        
        # Get recent response times (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_times = [
            dp.value for dp in self.time_series[key]
            if dp.timestamp > cutoff_time
        ]
        
        return statistics.mean(recent_times) if recent_times else None

    async def _calculate_completion_rate(self, agent_id: str) -> Optional[float]:
        """Calculate task completion rate for an agent"""
        if agent_id not in self.task_starts or agent_id not in self.task_completions:
            return None
        
        # Get recent task data (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        recent_starts = [t for t in self.task_starts[agent_id] if t > cutoff_time]
        recent_completions = [t for t in self.task_completions[agent_id] if t > cutoff_time]
        
        if not recent_starts:
            return None
        
        completion_rate = len(recent_completions) / len(recent_starts) * 100
        return min(completion_rate, 100.0)

    async def _calculate_error_rate(self, agent_id: str) -> Optional[float]:
        """Calculate error rate for an agent"""
        if agent_id not in self.task_starts or agent_id not in self.task_errors:
            return None
        
        # Get recent data (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        recent_starts = [t for t in self.task_starts[agent_id] if t > cutoff_time]
        recent_errors = [t for t in self.task_errors[agent_id] if t > cutoff_time]
        
        if not recent_starts:
            return None
        
        error_rate = len(recent_errors) / len(recent_starts) * 100
        return min(error_rate, 100.0)

    async def _calculate_throughput(self, agent_id: str) -> Optional[float]:
        """Calculate throughput (tasks per minute) for an agent"""
        if agent_id not in self.task_completions:
            return None
        
        # Get completions in the last hour
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_completions = [t for t in self.task_completions[agent_id] if t > cutoff_time]
        
        if not recent_completions:
            return 0.0
        
        # Calculate tasks per minute
        hours_elapsed = 1.0  # We're looking at last hour
        return len(recent_completions) / (hours_elapsed * 60)

    async def _calculate_resource_efficiency(self, agent_id: str) -> Optional[float]:
        """Calculate resource efficiency score"""
        # This would integrate with actual resource monitoring
        # For now, simulate based on completion rate and response time
        completion_rate = await self._calculate_completion_rate(agent_id)
        response_time = await self._calculate_avg_response_time(agent_id)
        
        if completion_rate is None or response_time is None:
            return None
        
        # Higher completion rate and lower response time = higher efficiency
        efficiency = completion_rate * (1.0 / max(response_time, 0.1))
        return min(efficiency, 100.0)

    async def _calculate_coordination_efficiency(self, agent_id: str) -> Optional[float]:
        """Calculate coordination efficiency score"""
        # This would integrate with swarm coordination metrics
        # For now, simulate based on error rate and completion rate
        error_rate = await self._calculate_error_rate(agent_id)
        completion_rate = await self._calculate_completion_rate(agent_id)
        
        if error_rate is None or completion_rate is None:
            return None
        
        # Lower error rate and higher completion rate = better coordination
        coordination_score = completion_rate * (1.0 - (error_rate / 100.0))
        return max(0.0, coordination_score)

    async def _perform_trend_analysis(self):
        """Perform trend analysis on collected metrics"""
        
        for (agent_id, metric_type), data_points in self.time_series.items():
            if len(data_points) < 10:  # Need minimum data for trend analysis
                continue
            
            try:
                trend = await self._analyze_trend(agent_id, metric_type, data_points)
                if trend:
                    self.trend_analyses[(agent_id, metric_type)] = trend
                    
                    # Check for performance degradation
                    if trend.trend_direction == 'degrading' and trend.confidence > 0.7:
                        await self._trigger_degradation_alert(agent_id, metric_type, trend)
                        
            except (ValueError, statistics.StatisticsError) as e:
                logger.error("Error in trend analysis for %s/%s: %s", agent_id, metric_type.value, e)

    async def _analyze_trend(self, agent_id: str, metric_type: MetricType, data_points: deque) -> Optional[TrendAnalysis]:
        """Analyze trend for a specific metric"""
        if len(data_points) < 5:
            return None
        
        # Get recent data points for analysis
        recent_points = list(data_points)[-self.trend_analysis_window:]
        values = [dp.value for dp in recent_points]
        
        # Calculate trend slope using linear regression
        n = len(values)
        x_values = list(range(n))
        
        # Linear regression calculations
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Calculate confidence (R-squared)
        y_mean = statistics.mean(values)
        ss_tot = sum((y - y_mean) ** 2 for y in values)
        y_pred = [slope * x + (sum_y - slope * sum_x) / n for x in x_values]
        ss_res = sum((y - y_pred[i]) ** 2 for i, y in enumerate(values))
        
        confidence = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Determine trend direction
        trend_direction = 'stable'
        if abs(slope) > 0.1:  # Threshold for significant change
            if metric_type in [MetricType.RESPONSE_TIME, MetricType.ERROR_RATE]:
                # For these metrics, positive slope is bad
                trend_direction = 'degrading' if slope > 0 else 'improving'
            else:
                # For other metrics, positive slope is good
                trend_direction = 'improving' if slope > 0 else 'degrading'
        
        # Calculate recent vs historical averages
        recent_average = statistics.mean(values[-10:]) if len(values) >= 10 else statistics.mean(values)
        historical_average = statistics.mean(values[:-10]) if len(values) >= 20 else recent_average
        
        # Simple prediction (next value based on trend)
        prediction = values[-1] + slope if slope else None
        
        return TrendAnalysis(
            metric_type=metric_type,
            agent_id=agent_id,
            trend_direction=trend_direction,
            slope=slope,
            confidence=confidence,
            recent_average=recent_average,
            historical_average=historical_average,
            prediction=prediction
        )

    async def _trigger_degradation_alert(self, agent_id: str, metric_type: MetricType, trend: TrendAnalysis):
        """Trigger alert for performance degradation"""
        logger.warning("Performance degradation detected for agent %s: %s trending %s",
                      agent_id, metric_type.value, trend.trend_direction)
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_id, metric_type, trend)
                else:
                    callback(agent_id, metric_type, trend)
            except (ValueError, TypeError) as e:
                logger.error("Error in alert callback: %s", e)

    async def _cleanup_old_data(self):
        """Clean up old metric data"""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        
        # Clean up raw metrics
        for agent_id, metrics in self.raw_metrics.items():
            # Remove old data points
            while metrics and metrics[0].timestamp < cutoff_time:
                metrics.popleft()
        
        # Clean up time series data
        for _, data_points in self.time_series.items():
            while data_points and data_points[0].timestamp < cutoff_time:
                data_points.popleft()
        
        # Clean up task tracking data
        for agent_id in list(self.task_starts.keys()):
            self.task_starts[agent_id] = [t for t in self.task_starts[agent_id] if t > cutoff_time]
            if not self.task_starts[agent_id]:
                del self.task_starts[agent_id]
        
        for agent_id in list(self.task_completions.keys()):
            self.task_completions[agent_id] = [t for t in self.task_completions[agent_id] if t > cutoff_time]
            if not self.task_completions[agent_id]:
                del self.task_completions[agent_id]
        
        for agent_id in list(self.task_errors.keys()):
            self.task_errors[agent_id] = [t for t in self.task_errors[agent_id] if t > cutoff_time]
            if not self.task_errors[agent_id]:
                del self.task_errors[agent_id]

    # Public interface methods
    
    def record_metric(self, agent_id: str, metric_type: MetricType, value: float, 
                     timestamp: Optional[datetime] = None, metadata: Optional[Dict[str, Any]] = None):
        """Record a metric data point"""
        if timestamp is None:
            timestamp = datetime.now()
        
        data_point = MetricDataPoint(
            timestamp=timestamp,
            value=value,
            metric_type=metric_type,
            agent_id=agent_id,
            metadata=metadata or {}
        )
        
        # Store in raw metrics
        self.raw_metrics[agent_id].append(data_point)
        
        # Store in time series
        key = (agent_id, metric_type)
        self.time_series[key].append(data_point)

    def record_task_start(self, agent_id: str, timestamp: Optional[datetime] = None):
        """Record the start of a task for an agent"""
        if timestamp is None:
            timestamp = datetime.now()
        self.task_starts[agent_id].append(timestamp)

    def record_task_completion(self, agent_id: str, timestamp: Optional[datetime] = None):
        """Record the completion of a task for an agent"""
        if timestamp is None:
            timestamp = datetime.now()
        self.task_completions[agent_id].append(timestamp)

    def record_task_error(self, agent_id: str, timestamp: Optional[datetime] = None):
        """Record an error for an agent"""
        if timestamp is None:
            timestamp = datetime.now()
        self.task_errors[agent_id].append(timestamp)

    def record_response_time(self, agent_id: str, response_time: float, timestamp: Optional[datetime] = None):
        """Record response time for an agent"""
        self.record_metric(agent_id, MetricType.RESPONSE_TIME, response_time, timestamp)

    def get_agent_metrics(self, agent_id: str) -> Optional[PerformanceMetrics]:
        """Get current performance metrics for an agent"""
        return self.aggregated_metrics.get(agent_id)

    def get_metric_history(self, agent_id: str, metric_type: MetricType, 
                          limit: int = 100) -> List[MetricDataPoint]:
        """Get metric history for an agent and metric type"""
        key = (agent_id, metric_type)
        if key not in self.time_series:
            return []
        
        data_points = list(self.time_series[key])
        return data_points[-limit:] if limit else data_points

    def get_trend_analysis(self, agent_id: str, metric_type: MetricType) -> Optional[TrendAnalysis]:
        """Get trend analysis for a specific metric"""
        return self.trend_analyses.get((agent_id, metric_type))

    def get_all_trends(self, agent_id: str) -> Dict[MetricType, TrendAnalysis]:
        """Get all trend analyses for an agent"""
        return {
            metric_type: trend for (aid, metric_type), trend in self.trend_analyses.items()
            if aid == agent_id
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        if not self.aggregated_metrics:
            return {
                "total_agents": 0,
                "avg_response_time": 0.0,
                "avg_completion_rate": 0.0,
                "avg_error_rate": 0.0,
                "agents_improving": 0,
                "agents_degrading": 0
            }
        
        metrics = list(self.aggregated_metrics.values())
        
        improving_agents = sum(1 for m in metrics if m.trend_direction == 'improving')
        degrading_agents = sum(1 for m in metrics if m.trend_direction == 'degrading')
        
        return {
            "total_agents": len(metrics),
            "avg_response_time": statistics.mean(m.avg_response_time for m in metrics),
            "avg_completion_rate": statistics.mean(m.completion_rate for m in metrics),
            "avg_error_rate": statistics.mean(m.error_rate for m in metrics),
            "avg_throughput": statistics.mean(m.throughput for m in metrics),
            "agents_improving": improving_agents,
            "agents_degrading": degrading_agents,
            "agents_stable": len(metrics) - improving_agents - degrading_agents
        }

    def get_top_performers(self, metric_type: MetricType, limit: int = 5) -> List[Tuple[str, float]]:
        """Get top performing agents for a specific metric"""
        agent_values = []
        
        for (agent_id, mt), data_points in self.time_series.items():
            if mt == metric_type and data_points:
                recent_avg = statistics.mean([dp.value for dp in list(data_points)[-10:]])
                agent_values.append((agent_id, recent_avg))
        
        # Sort based on metric type (higher is better for most, lower for response time and error rate)
        reverse = metric_type not in [MetricType.RESPONSE_TIME, MetricType.ERROR_RATE]
        agent_values.sort(key=lambda x: x[1], reverse=reverse)
        
        return agent_values[:limit]

    def export_metrics(self, agent_id: Optional[str] = None, 
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Export metrics data as JSON-serializable dictionary"""
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        export_data = {
            "export_timestamp": end_time.isoformat(),
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "agents": {}
        }
        
        # Filter by agent if specified
        agent_ids = [agent_id] if agent_id else list(set(aid for aid, _ in self.time_series.keys()))
        
        for aid in agent_ids:
            agent_data = {"metrics": {}}
            
            for metric_type in MetricType:
                key = (aid, metric_type)
                if key in self.time_series:
                    data_points = [
                        {
                            "timestamp": dp.timestamp.isoformat(),
                            "value": dp.value,
                            "metadata": dp.metadata
                        }
                        for dp in self.time_series[key]
                        if start_time <= dp.timestamp <= end_time
                    ]
                    
                    if data_points:
                        agent_data["metrics"][metric_type.value] = data_points
            
            # Include trend analysis
            trends = {}
            for (aid_trend, metric_type), trend in self.trend_analyses.items():
                if aid_trend == aid:
                    trends[metric_type.value] = {
                        "trend_direction": trend.trend_direction,
                        "slope": trend.slope,
                        "confidence": trend.confidence,
                        "recent_average": trend.recent_average,
                        "historical_average": trend.historical_average,
                        "prediction": trend.prediction
                    }
            
            if trends:
                agent_data["trends"] = trends
            
            if agent_data["metrics"]:
                export_data["agents"][aid] = agent_data
        
        return export_data

    def add_alert_callback(self, callback: Callable):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable):
        """Remove alert callback"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)