"""
Ecosystem Health Monitor for MCP Swarm Intelligence Server

This module provides comprehensive ecosystem health monitoring, alerting,
and bottleneck detection to ensure optimal swarm performance and resilience.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Overall health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    RECOVERING = "recovering"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class HealthAlert:
    """Health alert information"""
    alert_id: str
    severity: AlertSeverity
    component: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class ComponentHealth:
    """Health status for a system component"""
    component_id: str
    component_type: str  # 'agent', 'coordinator', 'memory', 'network'
    status: HealthStatus
    health_score: float  # 0-100
    last_check: datetime
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class EcosystemHealth:
    """Overall ecosystem health summary"""
    overall_status: HealthStatus
    overall_score: float
    total_components: int
    healthy_components: int
    warning_components: int
    critical_components: int
    active_alerts: int
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

class EcosystemHealthMonitor:
    """Comprehensive ecosystem health monitoring system"""
    
    def __init__(self,
                 monitoring_interval: float = 60.0,
                 health_check_timeout: float = 30.0,
                 alert_retention_hours: int = 24,
                 bottleneck_threshold: float = 80.0):
        """
        Initialize the ecosystem health monitor
        
        Args:
            monitoring_interval: Interval between health checks in seconds
            health_check_timeout: Timeout for individual health checks
            alert_retention_hours: Hours to retain resolved alerts
            bottleneck_threshold: Threshold for identifying bottlenecks
        """
        self.monitoring_interval = monitoring_interval
        self.health_check_timeout = health_check_timeout
        self.alert_retention_hours = alert_retention_hours
        self.bottleneck_threshold = bottleneck_threshold
        
        # Health tracking
        self.component_health: Dict[str, ComponentHealth] = {}
        self.ecosystem_health: Optional[EcosystemHealth] = None
        self.health_history: List[EcosystemHealth] = []
        
        # Alert management
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.alert_history: List[HealthAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        # Component dependencies and relationships
        self.component_dependencies: Dict[str, Set[str]] = {}
        self.critical_components: Set[str] = set()
        
        # Monitoring control
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Health check functions
        self.health_checkers: Dict[str, Callable] = {}
        
        # Performance thresholds
        self.thresholds = {
            "response_time_warning": 5.0,
            "response_time_critical": 10.0,
            "error_rate_warning": 5.0,
            "error_rate_critical": 10.0,
            "cpu_warning": 70.0,
            "cpu_critical": 90.0,
            "memory_warning": 80.0,
            "memory_critical": 95.0,
            "availability_warning": 95.0,
            "availability_critical": 90.0
        }

    async def start_monitoring(self):
        """Start the ecosystem health monitoring"""
        if self.is_monitoring:
            logger.warning("Ecosystem health monitoring already started")
            return
            
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Ecosystem health monitoring started")

    async def stop_monitoring(self):
        """Stop the health monitoring"""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Ecosystem health monitoring stopped")

    async def _monitoring_loop(self):
        """Main health monitoring loop"""
        while self.is_monitoring:
            try:
                await self._perform_health_checks()
                await self._analyze_ecosystem_health()
                await self._detect_bottlenecks()
                await self._cleanup_old_alerts()
                await asyncio.sleep(self.monitoring_interval)
            except (asyncio.CancelledError, OSError) as e:
                logger.error("Error in health monitoring loop: %s", e)
                await asyncio.sleep(self.monitoring_interval)

    async def _perform_health_checks(self):
        """Perform health checks on all registered components"""
        check_time = datetime.now()
        
        # Check all registered components
        for component_id in list(self.component_health.keys()):
            try:
                await self._check_component_health(component_id, check_time)
            except (ValueError, KeyError, asyncio.TimeoutError) as e:
                logger.error("Health check failed for component %s: %s", component_id, e)
                await self._handle_health_check_failure(component_id, str(e))

    async def _check_component_health(self, component_id: str, check_time: datetime):
        """Check health of a specific component"""
        component = self.component_health[component_id]
        
        # Run custom health checker if available
        if component_id in self.health_checkers:
            try:
                health_data = await asyncio.wait_for(
                    self.health_checkers[component_id](),
                    timeout=self.health_check_timeout
                )
                await self._update_component_health(component_id, health_data, check_time)
            except asyncio.TimeoutError:
                await self._handle_health_check_timeout(component_id)
        else:
            # Default health check based on component type
            await self._default_health_check(component_id, check_time)

    async def _default_health_check(self, component_id: str, check_time: datetime):
        """Default health check implementation"""
        # Initialize health data
        health_score = 100.0
        issues = []
        metrics = {}
        
        # Simulate health checks based on component type
        if component_id.startswith('agent_'):
            # Check agent-specific metrics
            health_score, issues, metrics = await self._check_agent_health(component_id)
        elif component_id.startswith('coordinator_'):
            # Check coordinator health
            health_score, issues, metrics = await self._check_coordinator_health(component_id)
        elif component_id.startswith('memory_'):
            # Check memory system health
            health_score, issues, metrics = await self._check_memory_health(component_id)
        elif component_id.startswith('network_'):
            # Check network connectivity
            health_score, issues, metrics = await self._check_network_health(component_id)
        
        # Determine status based on health score
        status = self._determine_health_status(health_score, issues)
        
        # Update component health
        component = self.component_health[component_id]
        component.status = status
        component.health_score = health_score
        component.last_check = check_time
        component.issues = issues
        component.metrics = metrics

    async def _check_agent_health(self, _agent_id: str) -> tuple[float, List[str], Dict[str, float]]:
        """Check health of an agent component"""
        health_score = 100.0
        issues = []
        metrics = {}
        
        # This would integrate with actual agent monitoring systems
        # For now, simulate based on thresholds
        
        # Simulate metrics (these would come from actual monitoring)
        response_time = 2.0  # seconds
        error_rate = 1.0     # percentage
        cpu_usage = 45.0     # percentage
        memory_usage = 60.0  # percentage
        availability = 99.5  # percentage
        
        metrics = {
            "response_time": response_time,
            "error_rate": error_rate,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "availability": availability
        }
        
        # Check thresholds and calculate health score
        if response_time > self.thresholds["response_time_critical"]:
            health_score -= 30
            issues.append("Critical response time")
        elif response_time > self.thresholds["response_time_warning"]:
            health_score -= 15
            issues.append("High response time")
        
        if error_rate > self.thresholds["error_rate_critical"]:
            health_score -= 25
            issues.append("Critical error rate")
        elif error_rate > self.thresholds["error_rate_warning"]:
            health_score -= 10
            issues.append("Elevated error rate")
        
        if cpu_usage > self.thresholds["cpu_critical"]:
            health_score -= 20
            issues.append("Critical CPU usage")
        elif cpu_usage > self.thresholds["cpu_warning"]:
            health_score -= 10
            issues.append("High CPU usage")
        
        if memory_usage > self.thresholds["memory_critical"]:
            health_score -= 20
            issues.append("Critical memory usage")
        elif memory_usage > self.thresholds["memory_warning"]:
            health_score -= 10
            issues.append("High memory usage")
        
        if availability < self.thresholds["availability_critical"]:
            health_score -= 35
            issues.append("Critical availability")
        elif availability < self.thresholds["availability_warning"]:
            health_score -= 15
            issues.append("Low availability")
        
        return max(0.0, health_score), issues, metrics

    async def _check_coordinator_health(self, _coordinator_id: str) -> tuple[float, List[str], Dict[str, float]]:
        """Check health of the swarm coordinator"""
        health_score = 100.0
        issues = []
        metrics = {}
        
        # Simulate coordinator metrics
        coordination_efficiency = 95.0
        decision_latency = 1.5
        consensus_success_rate = 98.0
        active_agents = 15
        
        metrics = {
            "coordination_efficiency": coordination_efficiency,
            "decision_latency": decision_latency,
            "consensus_success_rate": consensus_success_rate,
            "active_agents": active_agents
        }
        
        # Check coordinator-specific thresholds
        if coordination_efficiency < 80.0:
            health_score -= 25
            issues.append("Low coordination efficiency")
        elif coordination_efficiency < 90.0:
            health_score -= 10
            issues.append("Reduced coordination efficiency")
        
        if decision_latency > 5.0:
            health_score -= 20
            issues.append("High decision latency")
        elif decision_latency > 3.0:
            health_score -= 10
            issues.append("Elevated decision latency")
        
        if consensus_success_rate < 90.0:
            health_score -= 30
            issues.append("Low consensus success rate")
        elif consensus_success_rate < 95.0:
            health_score -= 15
            issues.append("Reduced consensus success rate")
        
        return max(0.0, health_score), issues, metrics

    async def _check_memory_health(self, _memory_id: str) -> tuple[float, List[str], Dict[str, float]]:
        """Check health of memory systems"""
        health_score = 100.0
        issues = []
        metrics = {}
        
        # Simulate memory system metrics
        query_latency = 0.5
        storage_usage = 65.0
        index_health = 98.0
        backup_status = 100.0
        
        metrics = {
            "query_latency": query_latency,
            "storage_usage": storage_usage,
            "index_health": index_health,
            "backup_status": backup_status
        }
        
        # Check memory-specific thresholds
        if query_latency > 2.0:
            health_score -= 20
            issues.append("High query latency")
        elif query_latency > 1.0:
            health_score -= 10
            issues.append("Elevated query latency")
        
        if storage_usage > 90.0:
            health_score -= 25
            issues.append("Critical storage usage")
        elif storage_usage > 80.0:
            health_score -= 10
            issues.append("High storage usage")
        
        if index_health < 95.0:
            health_score -= 15
            issues.append("Index health degraded")
        
        if backup_status < 100.0:
            health_score -= 20
            issues.append("Backup system issues")
        
        return max(0.0, health_score), issues, metrics

    async def _check_network_health(self, _network_id: str) -> tuple[float, List[str], Dict[str, float]]:
        """Check health of network connectivity"""
        health_score = 100.0
        issues = []
        metrics = {}
        
        # Simulate network metrics
        latency = 50.0  # ms
        packet_loss = 0.1  # percentage
        bandwidth_usage = 40.0  # percentage
        connection_count = 25
        
        metrics = {
            "latency": latency,
            "packet_loss": packet_loss,
            "bandwidth_usage": bandwidth_usage,
            "connection_count": connection_count
        }
        
        # Check network-specific thresholds
        if latency > 200.0:
            health_score -= 25
            issues.append("High network latency")
        elif latency > 100.0:
            health_score -= 10
            issues.append("Elevated network latency")
        
        if packet_loss > 1.0:
            health_score -= 30
            issues.append("High packet loss")
        elif packet_loss > 0.5:
            health_score -= 15
            issues.append("Elevated packet loss")
        
        if bandwidth_usage > 90.0:
            health_score -= 20
            issues.append("Critical bandwidth usage")
        elif bandwidth_usage > 80.0:
            health_score -= 10
            issues.append("High bandwidth usage")
        
        return max(0.0, health_score), issues, metrics

    def _determine_health_status(self, health_score: float, issues: List[str]) -> HealthStatus:
        """Determine health status based on score and issues"""
        if health_score >= 90.0 and not issues:
            return HealthStatus.HEALTHY
        elif health_score >= 70.0:
            return HealthStatus.WARNING
        elif health_score >= 50.0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.CRITICAL

    async def _update_component_health(self, component_id: str, health_data: Dict[str, Any], check_time: datetime):
        """Update component health with new data"""
        if component_id not in self.component_health:
            return
        
        component = self.component_health[component_id]
        old_status = component.status
        
        # Extract health information
        health_score = health_data.get('health_score', 100.0)
        issues = health_data.get('issues', [])
        metrics = health_data.get('metrics', {})
        
        # Update component
        component.health_score = health_score
        component.issues = issues
        component.metrics = metrics
        component.last_check = check_time
        component.status = self._determine_health_status(health_score, issues)
        
        # Check for status changes
        if old_status != component.status:
            await self._handle_status_change(component_id, old_status, component.status)

    async def _handle_status_change(self, component_id: str, old_status: HealthStatus, new_status: HealthStatus):
        """Handle component status changes"""
        logger.info("Component %s status changed from %s to %s", component_id, old_status.value, new_status.value)
        
        # Generate alerts for critical status changes
        if new_status == HealthStatus.CRITICAL:
            await self._create_alert(
                component_id,
                AlertSeverity.CRITICAL,
                f"Component {component_id} is in critical state",
                {"old_status": old_status.value, "new_status": new_status.value}
            )
        elif new_status == HealthStatus.WARNING and old_status == HealthStatus.HEALTHY:
            await self._create_alert(
                component_id,
                AlertSeverity.WARNING,
                f"Component {component_id} performance degraded",
                {"old_status": old_status.value, "new_status": new_status.value}
            )

    async def _handle_health_check_failure(self, component_id: str, error_message: str):
        """Handle health check failures"""
        if component_id in self.component_health:
            component = self.component_health[component_id]
            component.status = HealthStatus.CRITICAL
            component.issues.append(f"Health check failed: {error_message}")
            component.last_check = datetime.now()
            
            await self._create_alert(
                component_id,
                AlertSeverity.CRITICAL,
                f"Health check failed for {component_id}",
                {"error": error_message}
            )

    async def _handle_health_check_timeout(self, component_id: str):
        """Handle health check timeouts"""
        await self._handle_health_check_failure(component_id, "Health check timed out")

    async def _analyze_ecosystem_health(self):
        """Analyze overall ecosystem health"""
        if not self.component_health:
            return
        
        components = list(self.component_health.values())
        
        # Calculate overall statistics
        total_components = len(components)
        healthy_count = sum(1 for c in components if c.status == HealthStatus.HEALTHY)
        warning_count = sum(1 for c in components if c.status == HealthStatus.WARNING)
        critical_count = sum(1 for c in components if c.status == HealthStatus.CRITICAL)
        degraded_count = sum(1 for c in components if c.status == HealthStatus.DEGRADED)
        
        # Calculate overall health score
        total_score = sum(c.health_score for c in components)
        overall_score = total_score / total_components if total_components > 0 else 0.0
        
        # Determine overall status
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif critical_count + degraded_count > total_components * 0.3:  # More than 30% degraded
            overall_status = HealthStatus.DEGRADED
        elif warning_count > total_components * 0.5:  # More than 50% warnings
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Generate recommendations
        recommendations = self._generate_recommendations(components)
        
        # Update ecosystem health
        self.ecosystem_health = EcosystemHealth(
            overall_status=overall_status,
            overall_score=overall_score,
            total_components=total_components,
            healthy_components=healthy_count,
            warning_components=warning_count,
            critical_components=critical_count,
            active_alerts=len(self.active_alerts),
            bottlenecks=await self._identify_bottlenecks(),
            recommendations=recommendations
        )
        
        # Store in history
        self.health_history.append(self.ecosystem_health)
        if len(self.health_history) > 1000:  # Keep last 1000 records
            self.health_history = self.health_history[-500:]

    def _generate_recommendations(self, components: List[ComponentHealth]) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []
        
        # Check for common issues
        critical_components = [c for c in components if c.status == HealthStatus.CRITICAL]
        if critical_components:
            recommendations.append(f"Address critical issues in {len(critical_components)} components")
        
        high_cpu_components = [c for c in components if c.metrics.get('cpu_usage', 0) > 80]
        if high_cpu_components:
            recommendations.append("Consider load balancing for high CPU usage components")
        
        high_memory_components = [c for c in components if c.metrics.get('memory_usage', 0) > 80]
        if high_memory_components:
            recommendations.append("Monitor memory usage and consider optimization")
        
        high_error_rate_components = [c for c in components if c.metrics.get('error_rate', 0) > 5]
        if high_error_rate_components:
            recommendations.append("Investigate high error rates in affected components")
        
        # Check for capacity issues
        if len([c for c in components if c.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]]) > len(components) * 0.3:
            recommendations.append("Consider scaling ecosystem capacity")
        
        return recommendations

    async def _detect_bottlenecks(self):
        """Detect performance bottlenecks in the ecosystem"""
        bottlenecks = []
        
        for component_id, component in self.component_health.items():
            # Check if component is a bottleneck based on metrics
            metrics = component.metrics
            
            # High latency bottleneck
            if metrics.get('response_time', 0) > self.bottleneck_threshold / 10:  # > 8 seconds
                bottlenecks.append(f"{component_id}: High response time")
            
            # Resource bottleneck
            if metrics.get('cpu_usage', 0) > self.bottleneck_threshold:
                bottlenecks.append(f"{component_id}: CPU bottleneck")
            
            if metrics.get('memory_usage', 0) > self.bottleneck_threshold:
                bottlenecks.append(f"{component_id}: Memory bottleneck")
            
            # Coordination bottleneck
            if component.component_type == 'coordinator' and metrics.get('decision_latency', 0) > 3.0:
                bottlenecks.append(f"{component_id}: Decision latency bottleneck")
        
        return bottlenecks

    async def _identify_bottlenecks(self) -> List[str]:
        """Identify current system bottlenecks"""
        return await self._detect_bottlenecks()

    async def _create_alert(self, component: str, severity: AlertSeverity, message: str, metadata: Dict[str, Any]):
        """Create a new health alert"""
        alert_id = f"{component}_{severity.value}_{int(datetime.now().timestamp())}"
        
        alert = HealthAlert(
            alert_id=alert_id,
            severity=severity,
            component=component,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        logger.warning("Health alert created: %s - %s", severity.value.upper(), message)
        
        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except (ValueError, TypeError) as e:
                logger.error("Error in alert callback: %s", e)

    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        cutoff_time = datetime.now() - timedelta(hours=self.alert_retention_hours)
        
        # Remove old alerts from history
        self.alert_history = [
            alert for alert in self.alert_history
            if not alert.resolved or (alert.resolution_time and alert.resolution_time > cutoff_time)
        ]

    # Public interface methods
    
    def register_component(self, component_id: str, component_type: str, is_critical: bool = False):
        """Register a component for health monitoring"""
        self.component_health[component_id] = ComponentHealth(
            component_id=component_id,
            component_type=component_type,
            status=HealthStatus.HEALTHY,
            health_score=100.0,
            last_check=datetime.now()
        )
        
        if is_critical:
            self.critical_components.add(component_id)
        
        logger.info("Registered component %s (type: %s) for health monitoring", component_id, component_type)

    def register_health_checker(self, component_id: str, health_checker: Callable):
        """Register a custom health checker for a component"""
        self.health_checkers[component_id] = health_checker

    def add_component_dependency(self, component_id: str, depends_on: str):
        """Add a dependency relationship between components"""
        if component_id not in self.component_dependencies:
            self.component_dependencies[component_id] = set()
        self.component_dependencies[component_id].add(depends_on)

    def get_ecosystem_health(self) -> Optional[EcosystemHealth]:
        """Get current ecosystem health status"""
        return self.ecosystem_health

    def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Get health status for a specific component"""
        return self.component_health.get(component_id)

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[HealthAlert]:
        """Get active alerts, optionally filtered by severity"""
        alerts = list(self.active_alerts.values())
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        return alerts

    def get_critical_components(self) -> List[ComponentHealth]:
        """Get components in critical state"""
        return [
            component for component in self.component_health.values()
            if component.status == HealthStatus.CRITICAL
        ]

    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            del self.active_alerts[alert_id]
            logger.info("Alert %s resolved", alert_id)

    def get_health_trends(self, hours: int = 24) -> Dict[str, List[float]]:
        """Get health trends for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_health = [h for h in self.health_history if h.last_updated > cutoff_time]
        
        if not recent_health:
            return {}
        
        trends = {
            "overall_score": [h.overall_score for h in recent_health],
            "healthy_percentage": [(h.healthy_components / h.total_components * 100) for h in recent_health],
            "critical_percentage": [(h.critical_components / h.total_components * 100) for h in recent_health],
            "timestamps": [h.last_updated.isoformat() for h in recent_health]
        }
        
        return trends

    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Get detailed bottleneck analysis"""
        if not self.ecosystem_health:
            return {"bottlenecks": [], "recommendations": [], "severity": "unknown"}
        
        bottleneck_count = len(self.ecosystem_health.bottlenecks)
        total_components = self.ecosystem_health.total_components
        
        severity = "low"
        if bottleneck_count > total_components * 0.3:
            severity = "high"
        elif bottleneck_count > total_components * 0.1:
            severity = "medium"
        
        return {
            "bottlenecks": self.ecosystem_health.bottlenecks,
            "recommendations": self.ecosystem_health.recommendations,
            "bottleneck_count": bottleneck_count,
            "severity": severity,
            "bottleneck_percentage": (bottleneck_count / total_components * 100) if total_components > 0 else 0
        }

    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update performance thresholds"""
        self.thresholds.update(new_thresholds)
        logger.info("Updated health monitoring thresholds")

    def add_alert_callback(self, callback: Callable):
        """Add callback for health alerts"""
        self.alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable):
        """Remove alert callback"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)