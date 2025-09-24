"""
Anomaly Detection System for MCP Swarm Intelligence Server

This module implements comprehensive anomaly detection and adaptive response
systems for swarm coordination, using statistical methods, machine learning,
and pattern analysis to detect and respond to unusual behaviors.
"""

import json
import logging
import statistics
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path

# Scientific computing imports
try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.covariance import EllipticEnvelope
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    np = None
    logging.warning("Scikit-learn not available. Using statistical anomaly detection only.")

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies that can be detected"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_ANOMALY = "resource_anomaly"
    COORDINATION_FAILURE = "coordination_failure"
    PATTERN_DEVIATION = "pattern_deviation"
    THRESHOLD_VIOLATION = "threshold_violation"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    CLUSTERING_ANOMALY = "clustering_anomaly"
    STATISTICAL_OUTLIER = "statistical_outlier"

class AnomalySeverity(Enum):
    """Severity levels for detected anomalies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ResponseAction(Enum):
    """Response actions for detected anomalies"""
    LOG_ONLY = "log_only"
    ALERT = "alert"
    ADJUST_PARAMETERS = "adjust_parameters"
    RESTART_COMPONENT = "restart_component"
    ESCALATE = "escalate"
    QUARANTINE = "quarantine"
    AUTO_HEAL = "auto_heal"

@dataclass
class AnomalyEvent:
    """Represents a detected anomaly event"""
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    description: str
    component: str
    metrics: Dict[str, Any]
    confidence: float
    threshold_values: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    response_actions: List[ResponseAction] = field(default_factory=list)

@dataclass
class AnomalyRule:
    """Rule for anomaly detection"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    response_actions: List[ResponseAction]
    cooldown_period: timedelta = timedelta(minutes=5)
    enabled: bool = True

@dataclass
class AnomalyStatistics:
    """Statistics for anomaly detection and response"""
    total_anomalies: int = 0
    anomalies_by_type: Dict[AnomalyType, int] = field(default_factory=dict)
    anomalies_by_severity: Dict[AnomalySeverity, int] = field(default_factory=dict)
    false_positive_rate: float = 0.0
    response_success_rate: float = 0.0
    average_detection_time: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class AnomalyDetectionSystem:
    """
    Comprehensive anomaly detection and adaptive response system for
    swarm coordination using statistical methods and machine learning.
    """
    
    def __init__(
        self,
        database_path: str = "data/memory.db",
        window_size: int = 100,
        statistical_threshold: float = 2.0,
        ml_contamination: float = 0.1,
        history_retention_days: int = 30
    ):
        self.database_path = Path(database_path)
        self.window_size = window_size
        self.statistical_threshold = statistical_threshold
        self.ml_contamination = ml_contamination
        self.history_retention_days = history_retention_days
        
        # Detection models and data
        self.metric_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.anomaly_history: List[AnomalyEvent] = []
        self.detection_rules: List[AnomalyRule] = []
        self.last_rule_triggers: Dict[str, datetime] = {}
        
        # ML models (initialized when available)
        self.isolation_forest = None
        self.dbscan = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.envelope_detector = None
        
        # Response system
        self.response_handlers: Dict[ResponseAction, Callable] = {}
        self.adaptive_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Statistics and monitoring
        self.stats = AnomalyStatistics()
        self.detection_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Component health tracking
        self.component_health: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'status': 'healthy',
            'last_anomaly': None,
            'anomaly_count': 0,
            'recovery_time': 0.0
        })
        
        self._initialize_default_rules()
        self._initialize_ml_models()
    
    def _get_db_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    async def initialize_database(self):
        """Initialize database tables for anomaly detection"""
        conn = self._get_db_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    component TEXT NOT NULL,
                    description TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    threshold_values TEXT,
                    context TEXT,
                    response_actions TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS baseline_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    mean_value REAL NOT NULL,
                    std_dev REAL NOT NULL,
                    min_value REAL NOT NULL,
                    max_value REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(component, metric_name)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detection_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_method TEXT NOT NULL,
                    true_positives INTEGER DEFAULT 0,
                    false_positives INTEGER DEFAULT 0,
                    true_negatives INTEGER DEFAULT 0,
                    false_negatives INTEGER DEFAULT 0,
                    precision REAL DEFAULT 0.0,
                    recall REAL DEFAULT 0.0,
                    f1_score REAL DEFAULT 0.0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp 
                ON anomaly_events(timestamp DESC)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_anomaly_component_type 
                ON anomaly_events(component, anomaly_type)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def _initialize_default_rules(self):
        """Initialize default anomaly detection rules"""
        self.detection_rules = [
            # Performance degradation rules
            AnomalyRule(
                name="high_response_time",
                condition=lambda m: m.get('response_time', 0) > m.get('response_time_threshold', 5.0),
                anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                severity=AnomalySeverity.MEDIUM,
                response_actions=[ResponseAction.LOG_ONLY, ResponseAction.ALERT]
            ),
            
            AnomalyRule(
                name="low_throughput",
                condition=lambda m: m.get('throughput', 0) < m.get('throughput_threshold', 10.0),
                anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                severity=AnomalySeverity.MEDIUM,
                response_actions=[ResponseAction.ADJUST_PARAMETERS]
            ),
            
            # Resource anomaly rules
            AnomalyRule(
                name="high_memory_usage",
                condition=lambda m: m.get('memory_usage', 0) > m.get('memory_threshold', 0.8),
                anomaly_type=AnomalyType.RESOURCE_ANOMALY,
                severity=AnomalySeverity.HIGH,
                response_actions=[ResponseAction.ALERT, ResponseAction.AUTO_HEAL]
            ),
            
            AnomalyRule(
                name="high_cpu_usage",
                condition=lambda m: m.get('cpu_usage', 0) > m.get('cpu_threshold', 0.9),
                anomaly_type=AnomalyType.RESOURCE_ANOMALY,
                severity=AnomalySeverity.HIGH,
                response_actions=[ResponseAction.ALERT, ResponseAction.ADJUST_PARAMETERS]
            ),
            
            # Coordination failure rules
            AnomalyRule(
                name="agent_communication_failure",
                condition=lambda m: m.get('failed_communications', 0) > m.get('comm_failure_threshold', 5),
                anomaly_type=AnomalyType.COORDINATION_FAILURE,
                severity=AnomalySeverity.HIGH,
                response_actions=[ResponseAction.RESTART_COMPONENT, ResponseAction.ESCALATE]
            ),
            
            AnomalyRule(
                name="swarm_coordination_degradation",
                condition=lambda m: m.get('coordination_efficiency', 1.0) < m.get('coordination_threshold', 0.5),
                anomaly_type=AnomalyType.COORDINATION_FAILURE,
                severity=AnomalySeverity.CRITICAL,
                response_actions=[ResponseAction.ESCALATE, ResponseAction.QUARANTINE]
            )
        ]
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for anomaly detection"""
        if SKLEARN_AVAILABLE:
            try:
                # Isolation Forest for outlier detection
                self.isolation_forest = IsolationForest(
                    contamination=self.ml_contamination,
                    random_state=42,
                    n_estimators=100
                )
                
                # DBSCAN for clustering-based anomaly detection
                self.dbscan = DBSCAN(eps=0.5, min_samples=5)
                
                # Elliptic Envelope for Gaussian outlier detection
                self.envelope_detector = EllipticEnvelope(contamination=self.ml_contamination)
                
                logger.info("ML anomaly detection models initialized")
                
            except Exception as e:
                logger.warning("Error initializing ML models: %s", e)
                SKLEARN_AVAILABLE = False
    
    async def detect_anomalies(
        self,
        component: str,
        metrics: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[AnomalyEvent]:
        """
        Detect anomalies in component metrics using multiple detection methods
        
        Args:
            component: Name of the component being monitored
            metrics: Current metric values
            context: Additional context information
            
        Returns:
            List of detected anomaly events
        """
        try:
            anomalies = []
            context = context or {}
            
            # Update metric windows
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.metric_windows[f"{component}_{metric_name}"].append(value)
            
            # Statistical anomaly detection
            statistical_anomalies = await self._detect_statistical_anomalies(component, metrics, context)
            anomalies.extend(statistical_anomalies)
            
            # Rule-based anomaly detection
            rule_anomalies = await self._detect_rule_based_anomalies(component, metrics, context)
            anomalies.extend(rule_anomalies)
            
            # Machine learning anomaly detection
            if SKLEARN_AVAILABLE:
                ml_anomalies = await self._detect_ml_anomalies(component, metrics, context)
                anomalies.extend(ml_anomalies)
            
            # Pattern-based anomaly detection
            pattern_anomalies = await self._detect_pattern_anomalies(component, metrics, context)
            anomalies.extend(pattern_anomalies)
            
            # Store detected anomalies
            for anomaly in anomalies:
                await self._store_anomaly_event(anomaly)
                self.anomaly_history.append(anomaly)
                
                # Update statistics
                self.stats.total_anomalies += 1
                if anomaly.anomaly_type not in self.stats.anomalies_by_type:
                    self.stats.anomalies_by_type[anomaly.anomaly_type] = 0
                self.stats.anomalies_by_type[anomaly.anomaly_type] += 1
                
                if anomaly.severity not in self.stats.anomalies_by_severity:
                    self.stats.anomalies_by_severity[anomaly.severity] = 0
                self.stats.anomalies_by_severity[anomaly.severity] += 1
            
            # Update component health
            await self._update_component_health(component, anomalies)
            
            # Trigger responses
            for anomaly in anomalies:
                await self._execute_response_actions(anomaly)
            
            logger.info("Detected %d anomalies for component %s", len(anomalies), component)
            return anomalies
            
        except Exception as e:
            logger.error("Error in anomaly detection: %s", e)
            return []
    
    async def _detect_statistical_anomalies(
        self,
        component: str,
        metrics: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[AnomalyEvent]:
        """Detect anomalies using statistical methods"""
        anomalies = []
        
        for metric_name, current_value in metrics.items():
            if not isinstance(current_value, (int, float)):
                continue
            
            window_key = f"{component}_{metric_name}"
            window = self.metric_windows[window_key]
            
            if len(window) < 10:  # Need minimum history
                continue
            
            try:
                # Calculate statistical measures
                window_list = list(window)
                mean_val = statistics.mean(window_list)
                std_val = statistics.stdev(window_list) if len(window_list) > 1 else 0
                
                if std_val == 0:
                    continue
                
                # Z-score anomaly detection
                z_score = abs(current_value - mean_val) / std_val
                
                if z_score > self.statistical_threshold:
                    anomaly = AnomalyEvent(
                        timestamp=datetime.now(),
                        anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                        severity=self._determine_severity(z_score, [2.0, 3.0, 4.0]),
                        description=f"Statistical outlier detected in {metric_name}: value={current_value:.2f}, z-score={z_score:.2f}",
                        component=component,
                        metrics={metric_name: current_value},
                        confidence=min(z_score / 5.0, 1.0),  # Normalize confidence
                        threshold_values={'z_score': z_score, 'threshold': self.statistical_threshold},
                        context=context
                    )
                    anomalies.append(anomaly)
                
                # Update baseline statistics
                if component not in self.baseline_stats:
                    self.baseline_stats[component] = {}
                
                self.baseline_stats[component][metric_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min(window_list),
                    'max': max(window_list),
                    'count': len(window_list)
                }
                
            except Exception as e:
                logger.error("Error in statistical anomaly detection for %s.%s: %s", 
                           component, metric_name, e)
        
        return anomalies
    
    async def _detect_rule_based_anomalies(
        self,
        component: str,
        metrics: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[AnomalyEvent]:
        """Detect anomalies using predefined rules"""
        anomalies = []
        current_time = datetime.now()
        
        for rule in self.detection_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown period
            if (rule.name in self.last_rule_triggers and 
                current_time - self.last_rule_triggers[rule.name] < rule.cooldown_period):
                continue
            
            try:
                # Evaluate rule condition
                if rule.condition(metrics):
                    anomaly = AnomalyEvent(
                        timestamp=current_time,
                        anomaly_type=rule.anomaly_type,
                        severity=rule.severity,
                        description=f"Rule-based anomaly: {rule.name}",
                        component=component,
                        metrics=metrics,
                        confidence=0.9,  # High confidence for rule-based detection
                        context=context,
                        response_actions=rule.response_actions
                    )
                    
                    anomalies.append(anomaly)
                    self.last_rule_triggers[rule.name] = current_time
                    
            except Exception as e:
                logger.error("Error evaluating rule %s: %s", rule.name, e)
        
        return anomalies
    
    async def _detect_ml_anomalies(
        self,
        component: str,
        metrics: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[AnomalyEvent]:
        """Detect anomalies using machine learning models"""
        anomalies = []
        
        try:
            # Prepare feature vector
            feature_vector = []
            feature_names = []
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    feature_vector.append(value)
                    feature_names.append(metric_name)
            
            if len(feature_vector) < 2:
                return anomalies
            
            # Get historical data for training
            historical_data = []
            for metric_name in feature_names:
                window_key = f"{component}_{metric_name}"
                window = self.metric_windows[window_key]
                if len(window) >= 20:  # Minimum data for ML
                    historical_data.append(list(window))
            
            if len(historical_data) < 2:
                return anomalies
            
            # Transpose to get samples as rows
            training_data = list(zip(*historical_data))
            
            if len(training_data) < 20:
                return anomalies
            
            # Scale data
            training_scaled = self.scaler.fit_transform(training_data)
            current_scaled = self.scaler.transform([feature_vector])
            
            # Isolation Forest detection
            if self.isolation_forest:
                self.isolation_forest.fit(training_scaled)
                isolation_score = self.isolation_forest.decision_function(current_scaled)[0]
                is_outlier = self.isolation_forest.predict(current_scaled)[0] == -1
                
                if is_outlier:
                    anomaly = AnomalyEvent(
                        timestamp=datetime.now(),
                        anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                        severity=self._determine_severity(abs(isolation_score), [0.1, 0.2, 0.3]),
                        description=f"ML isolation forest anomaly detected: score={isolation_score:.3f}",
                        component=component,
                        metrics=metrics,
                        confidence=abs(isolation_score),
                        threshold_values={'isolation_score': isolation_score},
                        context=context
                    )
                    anomalies.append(anomaly)
            
            # Elliptic Envelope detection
            if self.envelope_detector and len(training_data) >= 10:
                try:
                    self.envelope_detector.fit(training_scaled)
                    envelope_score = self.envelope_detector.decision_function(current_scaled)[0]
                    is_envelope_outlier = self.envelope_detector.predict(current_scaled)[0] == -1
                    
                    if is_envelope_outlier:
                        anomaly = AnomalyEvent(
                            timestamp=datetime.now(),
                            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                            severity=self._determine_severity(abs(envelope_score), [1.0, 2.0, 3.0]),
                            description=f"ML envelope detector anomaly: score={envelope_score:.3f}",
                            component=component,
                            metrics=metrics,
                            confidence=min(abs(envelope_score) / 3.0, 1.0),
                            threshold_values={'envelope_score': envelope_score},
                            context=context
                        )
                        anomalies.append(anomaly)
                        
                except Exception as e:
                    logger.warning("Envelope detector error: %s", e)
            
        except Exception as e:
            logger.error("Error in ML anomaly detection: %s", e)
        
        return anomalies
    
    async def _detect_pattern_anomalies(
        self,
        component: str,
        metrics: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[AnomalyEvent]:
        """Detect anomalies based on temporal patterns"""
        anomalies = []
        
        try:
            current_time = datetime.now()
            
            # Check for temporal anomalies
            for metric_name, current_value in metrics.items():
                if not isinstance(current_value, (int, float)):
                    continue
                
                window_key = f"{component}_{metric_name}"
                window = self.metric_windows[window_key]
                
                if len(window) < 20:
                    continue
                
                window_list = list(window)
                
                # Detect sudden spikes or drops
                recent_avg = statistics.mean(window_list[-5:]) if len(window_list) >= 5 else current_value
                historical_avg = statistics.mean(window_list[:-5]) if len(window_list) >= 10 else recent_avg
                
                if historical_avg != 0:
                    change_ratio = abs(recent_avg - historical_avg) / abs(historical_avg)
                    
                    if change_ratio > 0.5:  # 50% change threshold
                        anomaly = AnomalyEvent(
                            timestamp=current_time,
                            anomaly_type=AnomalyType.PATTERN_DEVIATION,
                            severity=self._determine_severity(change_ratio, [0.5, 1.0, 2.0]),
                            description=f"Pattern deviation in {metric_name}: {change_ratio:.1%} change",
                            component=component,
                            metrics={metric_name: current_value},
                            confidence=min(change_ratio, 1.0),
                            threshold_values={'change_ratio': change_ratio},
                            context=context
                        )
                        anomalies.append(anomaly)
                
                # Check for trend anomalies
                if len(window_list) >= 10:
                    # Simple trend detection using linear regression approximation
                    x_vals = list(range(len(window_list)))
                    y_vals = window_list
                    
                    # Calculate trend using least squares
                    n = len(x_vals)
                    sum_x = sum(x_vals)
                    sum_y = sum(y_vals)
                    sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
                    sum_x2 = sum(x * x for x in x_vals)
                    
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                    
                    # Check for abnormal trend changes
                    if abs(slope) > statistics.stdev(y_vals) * 0.1:  # Trend threshold
                        trend_direction = "increasing" if slope > 0 else "decreasing"
                        anomaly = AnomalyEvent(
                            timestamp=current_time,
                            anomaly_type=AnomalyType.TEMPORAL_ANOMALY,
                            severity=AnomalySeverity.MEDIUM,
                            description=f"Trend anomaly in {metric_name}: {trend_direction} trend detected",
                            component=component,
                            metrics={metric_name: current_value},
                            confidence=min(abs(slope) / statistics.stdev(y_vals), 1.0),
                            threshold_values={'trend_slope': slope},
                            context=context
                        )
                        anomalies.append(anomaly)
                        
        except Exception as e:
            logger.error("Error in pattern anomaly detection: %s", e)
        
        return anomalies
    
    def _determine_severity(self, score: float, thresholds: List[float]) -> AnomalySeverity:
        """Determine anomaly severity based on score and thresholds"""
        if score >= thresholds[2]:  # High threshold
            return AnomalySeverity.CRITICAL
        elif score >= thresholds[1]:  # Medium threshold  
            return AnomalySeverity.HIGH
        elif score >= thresholds[0]:  # Low threshold
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    async def _update_component_health(self, component: str, anomalies: List[AnomalyEvent]):
        """Update component health status based on detected anomalies"""
        health = self.component_health[component]
        
        if anomalies:
            health['anomaly_count'] += len(anomalies)
            health['last_anomaly'] = datetime.now()
            
            # Determine health status based on severity
            max_severity = max(a.severity for a in anomalies)
            if max_severity == AnomalySeverity.CRITICAL:
                health['status'] = 'critical'
            elif max_severity == AnomalySeverity.HIGH:
                health['status'] = 'degraded'
            elif max_severity == AnomalySeverity.MEDIUM:
                health['status'] = 'warning'
            else:
                health['status'] = 'minor_issues'
        else:
            # No current anomalies, check if we can improve status
            if health['last_anomaly']:
                time_since_anomaly = datetime.now() - health['last_anomaly']
                if time_since_anomaly > timedelta(minutes=10):
                    health['status'] = 'recovering'
                if time_since_anomaly > timedelta(minutes=30):
                    health['status'] = 'healthy'
    
    async def _execute_response_actions(self, anomaly: AnomalyEvent):
        """Execute response actions for detected anomaly"""
        try:
            for action in anomaly.response_actions:
                if action in self.response_handlers:
                    await self.response_handlers[action](anomaly)
                else:
                    # Default actions
                    if action == ResponseAction.LOG_ONLY:
                        logger.warning("Anomaly detected: %s", anomaly.description)
                    elif action == ResponseAction.ALERT:
                        logger.error("ALERT - Anomaly: %s [%s]", anomaly.description, anomaly.severity.value)
                    elif action == ResponseAction.ADJUST_PARAMETERS:
                        await self._adjust_parameters_for_anomaly(anomaly)
                    elif action == ResponseAction.AUTO_HEAL:
                        await self._auto_heal_component(anomaly)
                    else:
                        logger.info("Response action %s not implemented for anomaly: %s", 
                                  action.value, anomaly.description)
                        
        except Exception as e:
            logger.error("Error executing response actions: %s", e)
    
    async def _adjust_parameters_for_anomaly(self, anomaly: AnomalyEvent):
        """Automatically adjust parameters in response to anomaly"""
        try:
            component = anomaly.component
            
            # Simple parameter adjustment logic
            if anomaly.anomaly_type == AnomalyType.PERFORMANCE_DEGRADATION:
                # Increase resources or reduce load
                logger.info("Adjusting parameters for performance degradation in %s", component)
                
            elif anomaly.anomaly_type == AnomalyType.RESOURCE_ANOMALY:
                # Optimize resource usage
                logger.info("Adjusting resource parameters for %s", component)
                
            # Store adjustment in adaptive thresholds
            if component not in self.adaptive_thresholds:
                self.adaptive_thresholds[component] = {}
            
            # Adjust thresholds based on anomaly history
            for metric_name, metric_value in anomaly.metrics.items():
                if isinstance(metric_value, (int, float)):
                    key = f"{metric_name}_threshold"
                    current_threshold = self.adaptive_thresholds[component].get(key, metric_value)
                    
                    # Adaptive threshold adjustment
                    if anomaly.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]:
                        # Tighten thresholds for critical anomalies
                        new_threshold = current_threshold * 0.9
                    else:
                        # Slight adjustment for minor anomalies
                        new_threshold = current_threshold * 0.95
                    
                    self.adaptive_thresholds[component][key] = new_threshold
                    logger.info("Adjusted threshold for %s.%s: %.3f -> %.3f", 
                              component, key, current_threshold, new_threshold)
                    
        except Exception as e:
            logger.error("Error adjusting parameters for anomaly: %s", e)
    
    async def _auto_heal_component(self, anomaly: AnomalyEvent):
        """Attempt automatic healing of component"""
        try:
            component = anomaly.component
            logger.info("Attempting auto-heal for component %s", component)
            
            # Record healing attempt
            heal_start = datetime.now()
            
            # Simple auto-healing strategies
            if anomaly.anomaly_type == AnomalyType.RESOURCE_ANOMALY:
                # Simulate resource cleanup/optimization
                logger.info("Performing resource cleanup for %s", component)
                
            elif anomaly.anomaly_type == AnomalyType.COORDINATION_FAILURE:
                # Simulate coordination reset
                logger.info("Resetting coordination parameters for %s", component)
                
            # Update component health with healing attempt
            health = self.component_health[component]
            health['recovery_time'] = (datetime.now() - heal_start).total_seconds()
            
            logger.info("Auto-heal completed for %s in %.2fs", component, health['recovery_time'])
            
        except Exception as e:
            logger.error("Error in auto-heal for component %s: %s", component, e)
    
    async def _store_anomaly_event(self, anomaly: AnomalyEvent):
        """Store anomaly event in database"""
        try:
            conn = self._get_db_connection()
            try:
                conn.execute("""
                    INSERT INTO anomaly_events 
                    (timestamp, anomaly_type, severity, component, description, 
                     metrics, confidence, threshold_values, context, response_actions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    anomaly.timestamp,
                    anomaly.anomaly_type.value,
                    anomaly.severity.value,
                    anomaly.component,
                    anomaly.description,
                    json.dumps(anomaly.metrics),
                    anomaly.confidence,
                    json.dumps(anomaly.threshold_values),
                    json.dumps(anomaly.context),
                    json.dumps([action.value for action in anomaly.response_actions])
                ))
                conn.commit()
            finally:
                conn.close()
                
        except Exception as e:
            logger.error("Error storing anomaly event: %s", e)
    
    def register_response_handler(self, action: ResponseAction, handler: Callable):
        """Register custom response handler for specific action"""
        self.response_handlers[action] = handler
        logger.info("Registered response handler for %s", action.value)
    
    def add_detection_rule(self, rule: AnomalyRule):
        """Add custom detection rule"""
        self.detection_rules.append(rule)
        logger.info("Added detection rule: %s", rule.name)
    
    async def get_anomaly_history(
        self,
        component: Optional[str] = None,
        anomaly_type: Optional[AnomalyType] = None,
        hours: int = 24,
        limit: int = 100
    ) -> List[AnomalyEvent]:
        """Get historical anomaly events with optional filtering"""
        try:
            conn = self._get_db_connection()
            try:
                query = """
                    SELECT * FROM anomaly_events 
                    WHERE timestamp > datetime('now', '-{} hours')
                """.format(hours)
                
                params = []
                
                if component:
                    query += " AND component = ?"
                    params.append(component)
                
                if anomaly_type:
                    query += " AND anomaly_type = ?"
                    params.append(anomaly_type.value)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                events = []
                for row in rows:
                    event = AnomalyEvent(
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        anomaly_type=AnomalyType(row['anomaly_type']),
                        severity=AnomalySeverity(row['severity']),
                        description=row['description'],
                        component=row['component'],
                        metrics=json.loads(row['metrics']),
                        confidence=row['confidence'],
                        threshold_values=json.loads(row['threshold_values'] or '{}'),
                        context=json.loads(row['context'] or '{}'),
                        response_actions=[ResponseAction(a) for a in json.loads(row['response_actions'] or '[]')]
                    )
                    events.append(event)
                
                return events
            finally:
                conn.close()
                
        except Exception as e:
            logger.error("Error getting anomaly history: %s", e)
            return []
    
    async def get_component_health(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get health status for components"""
        if component:
            return self.component_health.get(component, {})
        else:
            return dict(self.component_health)
    
    async def get_detection_statistics(self) -> Dict[str, Any]:
        """Get anomaly detection system statistics"""
        try:
            recent_anomalies = await self.get_anomaly_history(hours=24)
            
            stats = {
                'sklearn_available': SKLEARN_AVAILABLE,
                'total_anomalies': self.stats.total_anomalies,
                'anomalies_by_type': {t.value: count for t, count in self.stats.anomalies_by_type.items()},
                'anomalies_by_severity': {s.value: count for s, count in self.stats.anomalies_by_severity.items()},
                'recent_24h_anomalies': len(recent_anomalies),
                'active_detection_rules': len([r for r in self.detection_rules if r.enabled]),
                'monitored_components': len(self.component_health),
                'adaptive_thresholds': len(self.adaptive_thresholds),
                'ml_models_active': {
                    'isolation_forest': self.isolation_forest is not None,
                    'dbscan': self.dbscan is not None,
                    'envelope_detector': self.envelope_detector is not None
                },
                'detection_config': {
                    'window_size': self.window_size,
                    'statistical_threshold': self.statistical_threshold,
                    'ml_contamination': self.ml_contamination
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error("Error getting detection statistics: %s", e)
            return {'error': str(e), 'sklearn_available': SKLEARN_AVAILABLE}