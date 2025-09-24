"""
System Health Monitor for MCP Swarm Intelligence Server

This module provides comprehensive system health monitoring with predictive analytics
and automated alert management for the MCP swarm intelligence system.
"""

import psutil
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class HealthStatus(Enum):
    """System health status levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILURE = "failure"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class HealthMetric:
    """Individual health metric data"""
    name: str
    value: float
    threshold: float
    status: HealthStatus
    timestamp: datetime
    unit: str = ""


@dataclass
class PredictedIssue:
    """Predicted system issue"""
    issue_type: str
    severity: AlertSeverity
    probability: float
    eta_hours: float
    description: str
    recommended_actions: List[str]


@dataclass
class HealthRecommendation:
    """Health improvement recommendation"""
    category: str
    priority: int
    action: str
    expected_improvement: str
    effort_level: str


@dataclass
class SystemHealthStatus:
    """Complete system health status"""
    overall_score: float
    status: HealthStatus
    metrics: Dict[str, HealthMetric]
    efficiency_metrics: Dict[str, float]
    alerts: List[Dict[str, Any]]
    predictions: List[PredictedIssue]
    timestamp: datetime


class HealthMetricsCollector:
    """Collects various system health metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.baseline_metrics = {}
        self._last_net_time = None
        self._last_net_bytes = None
        
    async def collect_cpu_metrics(self) -> HealthMetric:
        """Collect CPU usage metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        status = self._evaluate_cpu_status(cpu_percent)
        
        return HealthMetric(
            name="cpu_usage",
            value=cpu_percent,
            threshold=80.0,
            status=status,
            timestamp=datetime.utcnow(),
            unit="%"
        )
    
    async def collect_memory_metrics(self) -> HealthMetric:
        """Collect memory usage metrics"""
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        status = self._evaluate_memory_status(memory_percent)
        
        return HealthMetric(
            name="memory_usage",
            value=memory_percent,
            threshold=85.0,
            status=status,
            timestamp=datetime.utcnow(),
            unit="%"
        )
    
    async def collect_disk_metrics(self) -> HealthMetric:
        """Collect disk usage metrics"""
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        status = self._evaluate_disk_status(disk_percent)
        
        return HealthMetric(
            name="disk_usage",
            value=disk_percent,
            threshold=90.0,
            status=status,
            timestamp=datetime.utcnow(),
            unit="%"
        )
    
    async def collect_network_metrics(self) -> HealthMetric:
        """Collect network I/O metrics"""
        net_io = psutil.net_io_counters()
        # Calculate network utilization as bytes per second
        current_time = time.time()
        
        if self._last_net_time is not None and self._last_net_bytes is not None:
            time_delta = current_time - self._last_net_time
            bytes_delta = (net_io.bytes_sent + net_io.bytes_recv) - self._last_net_bytes
            bytes_per_sec = bytes_delta / time_delta if time_delta > 0 else 0
        else:
            bytes_per_sec = 0
            
        self._last_net_time = current_time
        self._last_net_bytes = net_io.bytes_sent + net_io.bytes_recv
        
        # Convert to MB/s
        mb_per_sec = bytes_per_sec / (1024 * 1024)
        status = self._evaluate_network_status(mb_per_sec)
        
        return HealthMetric(
            name="network_io",
            value=mb_per_sec,
            threshold=100.0,  # 100 MB/s threshold
            status=status,
            timestamp=datetime.utcnow(),
            unit="MB/s"
        )
    
    def _evaluate_cpu_status(self, cpu_percent: float) -> HealthStatus:
        """Evaluate CPU status based on usage"""
        if cpu_percent < 50:
            return HealthStatus.EXCELLENT
        elif cpu_percent < 70:
            return HealthStatus.GOOD
        elif cpu_percent < 85:
            return HealthStatus.WARNING
        elif cpu_percent < 95:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.FAILURE
    
    def _evaluate_memory_status(self, memory_percent: float) -> HealthStatus:
        """Evaluate memory status based on usage"""
        if memory_percent < 60:
            return HealthStatus.EXCELLENT
        elif memory_percent < 75:
            return HealthStatus.GOOD
        elif memory_percent < 85:
            return HealthStatus.WARNING
        elif memory_percent < 95:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.FAILURE
    
    def _evaluate_disk_status(self, disk_percent: float) -> HealthStatus:
        """Evaluate disk status based on usage"""
        if disk_percent < 70:
            return HealthStatus.EXCELLENT
        elif disk_percent < 80:
            return HealthStatus.GOOD
        elif disk_percent < 90:
            return HealthStatus.WARNING
        elif disk_percent < 95:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.FAILURE
    
    def _evaluate_network_status(self, mb_per_sec: float) -> HealthStatus:
        """Evaluate network status based on I/O"""
        if mb_per_sec < 50:
            return HealthStatus.EXCELLENT
        elif mb_per_sec < 75:
            return HealthStatus.GOOD
        elif mb_per_sec < 90:
            return HealthStatus.WARNING
        elif mb_per_sec < 100:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.FAILURE


class PredictiveAnalytics:
    """Predictive analytics for system issues"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.historical_data = []
        
    async def predict_issues(self, current_metrics: Dict[str, HealthMetric]) -> List[PredictedIssue]:
        """Predict potential system issues"""
        predictions = []
        
        # CPU trend analysis
        if "cpu_usage" in current_metrics:
            cpu_prediction = await self._predict_cpu_issues(current_metrics["cpu_usage"])
            if cpu_prediction:
                predictions.append(cpu_prediction)
        
        # Memory trend analysis
        if "memory_usage" in current_metrics:
            memory_prediction = await self._predict_memory_issues(current_metrics["memory_usage"])
            if memory_prediction:
                predictions.append(memory_prediction)
        
        # Disk trend analysis
        if "disk_usage" in current_metrics:
            disk_prediction = await self._predict_disk_issues(current_metrics["disk_usage"])
            if disk_prediction:
                predictions.append(disk_prediction)
        
        return predictions
    
    async def _predict_cpu_issues(self, cpu_metric: HealthMetric) -> Optional[PredictedIssue]:
        """Predict CPU-related issues"""
        if cpu_metric.value > 75:
            return PredictedIssue(
                issue_type="cpu_overload",
                severity=AlertSeverity.WARNING,
                probability=0.75,
                eta_hours=2.0,
                description="CPU usage trending towards overload",
                recommended_actions=[
                    "Scale up compute resources",
                    "Optimize CPU-intensive processes",
                    "Implement load balancing"
                ]
            )
        return None
    
    async def _predict_memory_issues(self, memory_metric: HealthMetric) -> Optional[PredictedIssue]:
        """Predict memory-related issues"""
        if memory_metric.value > 80:
            return PredictedIssue(
                issue_type="memory_exhaustion",
                severity=AlertSeverity.CRITICAL,
                probability=0.85,
                eta_hours=1.5,
                description="Memory usage approaching critical levels",
                recommended_actions=[
                    "Increase available memory",
                    "Optimize memory usage patterns",
                    "Implement garbage collection tuning"
                ]
            )
        return None
    
    async def _predict_disk_issues(self, disk_metric: HealthMetric) -> Optional[PredictedIssue]:
        """Predict disk-related issues"""
        if disk_metric.value > 85:
            return PredictedIssue(
                issue_type="disk_space_exhaustion",
                severity=AlertSeverity.CRITICAL,
                probability=0.90,
                eta_hours=6.0,
                description="Disk space approaching critical levels",
                recommended_actions=[
                    "Clean up temporary files",
                    "Archive old data",
                    "Expand disk capacity"
                ]
            )
        return None


class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_alerts = []
        
    async def process_health_status(self, health_status: SystemHealthStatus) -> List[Dict[str, Any]]:
        """Process health status and generate alerts"""
        alerts = []
        
        for metric in health_status.metrics.values():
            if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL, HealthStatus.FAILURE]:
                alert = await self._create_alert(metric)
                alerts.append(alert)
        
        # Process predictions
        for prediction in health_status.predictions:
            if prediction.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                alert = await self._create_prediction_alert(prediction)
                alerts.append(alert)
        
        return alerts
    
    async def _create_alert(self, metric: HealthMetric) -> Dict[str, Any]:
        """Create alert from health metric"""
        severity_map = {
            HealthStatus.WARNING: AlertSeverity.WARNING,
            HealthStatus.CRITICAL: AlertSeverity.CRITICAL,
            HealthStatus.FAILURE: AlertSeverity.EMERGENCY
        }
        
        return {
            "type": "metric_alert",
            "severity": severity_map.get(metric.status, AlertSeverity.INFO).value,
            "metric": metric.name,
            "value": metric.value,
            "threshold": metric.threshold,
            "message": f"{metric.name} is at {metric.value}{metric.unit} (threshold: {metric.threshold}{metric.unit})",
            "timestamp": metric.timestamp.isoformat(),
            "requires_action": metric.status in [HealthStatus.CRITICAL, HealthStatus.FAILURE]
        }
    
    async def _create_prediction_alert(self, prediction: PredictedIssue) -> Dict[str, Any]:
        """Create alert from prediction"""
        return {
            "type": "prediction_alert",
            "severity": prediction.severity.value,
            "issue_type": prediction.issue_type,
            "probability": prediction.probability,
            "eta_hours": prediction.eta_hours,
            "message": prediction.description,
            "recommended_actions": prediction.recommended_actions,
            "timestamp": datetime.utcnow().isoformat(),
            "requires_action": True
        }


class SystemHealthMonitor:
    """Comprehensive system health monitoring with predictive analytics"""
    
    def __init__(self):
        self.health_metrics = HealthMetricsCollector()
        self.predictive_analytics = PredictiveAnalytics()
        self.alert_manager = AlertManager()
        self.logger = logging.getLogger(__name__)
        
    async def monitor_system_health(self) -> SystemHealthStatus:
        """Monitor comprehensive system health metrics"""
        try:
            # Collect all metrics
            metrics = {}
            
            # Collect CPU metrics
            cpu_metric = await self.health_metrics.collect_cpu_metrics()
            metrics["cpu_usage"] = cpu_metric
            
            # Collect memory metrics
            memory_metric = await self.health_metrics.collect_memory_metrics()
            metrics["memory_usage"] = memory_metric
            
            # Collect disk metrics
            disk_metric = await self.health_metrics.collect_disk_metrics()
            metrics["disk_usage"] = disk_metric
            
            # Collect network metrics
            network_metric = await self.health_metrics.collect_network_metrics()
            metrics["network_io"] = network_metric
            
            # Calculate overall health score
            overall_score = await self._calculate_health_score(metrics)
            overall_status = await self._determine_overall_status(metrics)
            
            # Calculate efficiency metrics
            efficiency_metrics = await self._calculate_efficiency_metrics(metrics)
            
            # Predict potential issues
            predictions = await self.predictive_analytics.predict_issues(metrics)
            
            # Create health status object
            health_status = SystemHealthStatus(
                overall_score=overall_score,
                status=overall_status,
                metrics=metrics,
                efficiency_metrics=efficiency_metrics,
                alerts=[],
                predictions=predictions,
                timestamp=datetime.utcnow()
            )
            
            # Generate alerts
            alerts = await self.alert_manager.process_health_status(health_status)
            health_status.alerts = alerts
            
            return health_status
            
        except Exception as e:
            self.logger.error("Error monitoring system health: %s", str(e))
            raise
    
    async def predict_system_issues(self) -> List[PredictedIssue]:
        """Predict potential system issues before they occur"""
        try:
            # Get current health status
            health_status = await self.monitor_system_health()
            return health_status.predictions
            
        except Exception as e:
            self.logger.error("Error predicting system issues: %s", str(e))
            return []
    
    async def generate_health_recommendations(self, health_status: SystemHealthStatus) -> List[HealthRecommendation]:
        """Generate actionable health improvement recommendations"""
        recommendations = []
        
        # CPU recommendations
        if "cpu_usage" in health_status.metrics:
            cpu_metric = health_status.metrics["cpu_usage"]
            if cpu_metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                recommendations.append(HealthRecommendation(
                    category="performance",
                    priority=1,
                    action="Optimize CPU-intensive processes and consider scaling",
                    expected_improvement="20-30% CPU usage reduction",
                    effort_level="medium"
                ))
        
        # Memory recommendations
        if "memory_usage" in health_status.metrics:
            memory_metric = health_status.metrics["memory_usage"]
            if memory_metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                recommendations.append(HealthRecommendation(
                    category="memory",
                    priority=1,
                    action="Implement memory optimization and garbage collection tuning",
                    expected_improvement="15-25% memory usage reduction",
                    effort_level="medium"
                ))
        
        # Disk recommendations
        if "disk_usage" in health_status.metrics:
            disk_metric = health_status.metrics["disk_usage"]
            if disk_metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                recommendations.append(HealthRecommendation(
                    category="storage",
                    priority=2,
                    action="Clean up unnecessary files and expand storage capacity",
                    expected_improvement="Prevent disk space exhaustion",
                    effort_level="low"
                ))
        
        return recommendations
    
    async def _calculate_health_score(self, metrics: Dict[str, HealthMetric]) -> float:
        """Calculate overall health score from metrics"""
        if not metrics:
            return 0.0
        
        status_scores = {
            HealthStatus.EXCELLENT: 100.0,
            HealthStatus.GOOD: 80.0,
            HealthStatus.WARNING: 60.0,
            HealthStatus.CRITICAL: 30.0,
            HealthStatus.FAILURE: 0.0
        }
        
        total_score = sum(status_scores.get(metric.status, 0.0) for metric in metrics.values())
        return total_score / len(metrics)
    
    async def _determine_overall_status(self, metrics: Dict[str, HealthMetric]) -> HealthStatus:
        """Determine overall system status"""
        if not metrics:
            return HealthStatus.FAILURE
        
        worst_status = HealthStatus.EXCELLENT
        status_priority = {
            HealthStatus.FAILURE: 5,
            HealthStatus.CRITICAL: 4,
            HealthStatus.WARNING: 3,
            HealthStatus.GOOD: 2,
            HealthStatus.EXCELLENT: 1
        }
        
        for metric in metrics.values():
            if status_priority.get(metric.status, 0) > status_priority.get(worst_status, 0):
                worst_status = metric.status
        
        return worst_status
    
    async def _calculate_efficiency_metrics(self, metrics: Dict[str, HealthMetric]) -> Dict[str, float]:
        """Calculate system efficiency metrics"""
        efficiency = {}
        
        if "cpu_usage" in metrics:
            cpu_value = metrics["cpu_usage"].value
            efficiency["cpu_efficiency"] = max(0.0, (100.0 - cpu_value) / 100.0)
        
        if "memory_usage" in metrics:
            memory_value = metrics["memory_usage"].value
            efficiency["memory_efficiency"] = max(0.0, (100.0 - memory_value) / 100.0)
        
        if "disk_usage" in metrics:
            disk_value = metrics["disk_usage"].value
            efficiency["storage_efficiency"] = max(0.0, (100.0 - disk_value) / 100.0)
        
        # Calculate overall efficiency
        if efficiency:
            efficiency["overall_efficiency"] = sum(efficiency.values()) / len(efficiency)
        else:
            efficiency["overall_efficiency"] = 0.0
        
        return efficiency