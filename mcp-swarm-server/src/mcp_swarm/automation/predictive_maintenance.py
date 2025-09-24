"""
Predictive Maintenance System for MCP Swarm Intelligence Server

This module provides predictive maintenance with failure prediction, maintenance
scheduling, and preemptive issue resolution for the MCP swarm intelligence system.
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class FailureType(Enum):
    """Types of system failures that can be predicted"""
    HARDWARE_FAILURE = "hardware_failure"
    SOFTWARE_CRASH = "software_crash"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    DISK_FAILURE = "disk_failure"
    NETWORK_FAILURE = "network_failure"
    DATABASE_CORRUPTION = "database_corruption"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class MaintenanceType(Enum):
    """Types of maintenance activities"""
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"


class MaintenancePriority(Enum):
    """Maintenance priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FailurePrediction:
    """Prediction of a potential failure"""
    failure_type: FailureType
    probability: float
    time_to_failure_hours: float
    confidence_score: float
    contributing_factors: List[str]
    severity_impact: str
    recommended_actions: List[str]
    timestamp: datetime


@dataclass
class MaintenanceTask:
    """Individual maintenance task"""
    task_id: str
    task_type: MaintenanceType
    priority: MaintenancePriority
    description: str
    estimated_duration_minutes: int
    required_resources: List[str]
    prerequisites: List[str]
    expected_outcomes: List[str]
    scheduled_time: datetime
    deadline: Optional[datetime] = None


@dataclass
class MaintenancePredictions:
    """Collection of maintenance predictions"""
    predicted_issues: List[FailurePrediction]
    recommended_tasks: List[MaintenanceTask]
    overall_risk_score: float
    next_maintenance_window: datetime
    system_health_trend: str


@dataclass
class MaintenanceSchedule:
    """Scheduled maintenance activities"""
    scheduled_tasks: List[MaintenanceTask]
    maintenance_windows: List[Dict[str, Any]]
    resource_allocation: Dict[str, Any]
    total_downtime_estimate: int
    schedule_optimization_score: float


class FailurePredictor:
    """Predicts potential system failures"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failure_patterns = {}
        self.historical_data = []
        
    async def predict_hardware_failures(self, system_metrics: Dict[str, Any]) -> List[FailurePrediction]:
        """Predict potential hardware failures"""
        predictions = []
        
        # CPU temperature and usage patterns
        cpu_usage = system_metrics.get("cpu_usage", 0)
        if cpu_usage > 90:
            predictions.append(FailurePrediction(
                failure_type=FailureType.HARDWARE_FAILURE,
                probability=0.3,
                time_to_failure_hours=72.0,
                confidence_score=0.7,
                contributing_factors=["sustained_high_cpu_usage", "thermal_stress"],
                severity_impact="high",
                recommended_actions=[
                    "Monitor CPU temperature",
                    "Reduce system load",
                    "Check cooling system"
                ],
                timestamp=datetime.utcnow()
            ))
        
        # Memory degradation patterns
        memory_usage = system_metrics.get("memory_usage", 0)
        if memory_usage > 95:
            predictions.append(FailurePrediction(
                failure_type=FailureType.MEMORY_EXHAUSTION,
                probability=0.8,
                time_to_failure_hours=24.0,
                confidence_score=0.9,
                contributing_factors=["memory_pressure", "potential_memory_leak"],
                severity_impact="critical",
                recommended_actions=[
                    "Restart memory-intensive processes",
                    "Check for memory leaks",
                    "Add more memory if possible"
                ],
                timestamp=datetime.utcnow()
            ))
        
        # Disk health predictions
        disk_usage = system_metrics.get("disk_usage", 0)
        if disk_usage > 95:
            predictions.append(FailurePrediction(
                failure_type=FailureType.DISK_FAILURE,
                probability=0.6,
                time_to_failure_hours=168.0,  # 1 week
                confidence_score=0.8,
                contributing_factors=["disk_space_critical", "high_io_operations"],
                severity_impact="high",
                recommended_actions=[
                    "Clean up disk space",
                    "Move data to additional storage",
                    "Monitor disk health metrics"
                ],
                timestamp=datetime.utcnow()
            ))
        
        return predictions
    
    async def predict_software_failures(self, system_metrics: Dict[str, Any]) -> List[FailurePrediction]:
        """Predict potential software failures"""
        predictions = []
        
        # Process crash predictions based on resource usage
        if system_metrics.get("error_rate", 0) > 0.05:  # 5% error rate
            predictions.append(FailurePrediction(
                failure_type=FailureType.SOFTWARE_CRASH,
                probability=0.4,
                time_to_failure_hours=48.0,
                confidence_score=0.6,
                contributing_factors=["high_error_rate", "resource_contention"],
                severity_impact="medium",
                recommended_actions=[
                    "Review error logs",
                    "Restart affected services",
                    "Update software components"
                ],
                timestamp=datetime.utcnow()
            ))
        
        # Database corruption prediction
        db_connections = system_metrics.get("db_connections", 0)
        max_connections = system_metrics.get("max_db_connections", 100)
        if db_connections > max_connections * 0.9:
            predictions.append(FailurePrediction(
                failure_type=FailureType.DATABASE_CORRUPTION,
                probability=0.2,
                time_to_failure_hours=120.0,
                confidence_score=0.5,
                contributing_factors=["connection_pool_saturation", "high_transaction_volume"],
                severity_impact="high",
                recommended_actions=[
                    "Optimize database queries",
                    "Increase connection pool size",
                    "Monitor transaction patterns"
                ],
                timestamp=datetime.utcnow()
            ))
        
        return predictions
    
    async def predict_performance_degradation(self, performance_trends: Dict[str, List[float]]) -> List[FailurePrediction]:
        """Predict performance degradation based on trends"""
        predictions = []
        
        # Response time degradation
        response_times = performance_trends.get("response_times", [])
        if len(response_times) >= 5:
            # Calculate trend slope
            recent_avg = statistics.mean(response_times[-3:])
            older_avg = statistics.mean(response_times[:3])
            
            if recent_avg > older_avg * 1.5:  # 50% increase
                predictions.append(FailurePrediction(
                    failure_type=FailureType.PERFORMANCE_DEGRADATION,
                    probability=0.7,
                    time_to_failure_hours=96.0,
                    confidence_score=0.8,
                    contributing_factors=["response_time_increase", "system_load_growth"],
                    severity_impact="medium",
                    recommended_actions=[
                        "Profile application performance",
                        "Optimize database queries",
                        "Scale system resources"
                    ],
                    timestamp=datetime.utcnow()
                ))
        
        return predictions
    
    async def analyze_failure_patterns(self, historical_failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze historical failure patterns for better predictions"""
        patterns = {
            "common_failure_types": {},
            "failure_frequency": {},
            "seasonal_patterns": {},
            "correlation_factors": {}
        }
        
        if not historical_failures:
            return patterns
        
        # Count failure types
        for failure in historical_failures:
            failure_type = failure.get("type", "unknown")
            patterns["common_failure_types"][failure_type] = (
                patterns["common_failure_types"].get(failure_type, 0) + 1
            )
        
        # Calculate failure frequency (failures per week)
        if len(historical_failures) > 0:
            time_span_days = (datetime.utcnow() - historical_failures[0]["timestamp"]).days
            time_span_weeks = max(time_span_days / 7, 1)
            patterns["failure_frequency"]["per_week"] = len(historical_failures) / time_span_weeks
        
        return patterns


class MaintenanceScheduler:
    """Schedules maintenance activities based on predictions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.maintenance_calendar = []
        self.resource_availability = {}
        
    async def create_maintenance_tasks(self, predictions: List[FailurePrediction]) -> List[MaintenanceTask]:
        """Create maintenance tasks based on failure predictions"""
        tasks = []
        task_counter = 1
        
        for prediction in predictions:
            # Determine maintenance type and priority
            maintenance_type = self._determine_maintenance_type(prediction)
            priority = self._determine_priority(prediction)
            
            # Create task based on failure type
            task = await self._create_task_for_failure_type(
                prediction.failure_type, 
                maintenance_type, 
                priority, 
                prediction,
                task_counter
            )
            
            if task:
                tasks.append(task)
                task_counter += 1
        
        return tasks
    
    async def schedule_maintenance_window(self, tasks: List[MaintenanceTask]) -> MaintenanceSchedule:
        """Schedule maintenance tasks into optimal time windows"""
        # Sort tasks by priority and deadline
        sorted_tasks = sorted(tasks, key=lambda t: (
            self._priority_score(t.priority),
            t.deadline or datetime.max
        ))
        
        # Group tasks into maintenance windows
        maintenance_windows = self._create_maintenance_windows(sorted_tasks)
        
        # Allocate resources
        resource_allocation = self._allocate_resources(sorted_tasks)
        
        # Calculate total downtime
        total_downtime = sum(task.estimated_duration_minutes for task in sorted_tasks)
        
        # Calculate schedule optimization score
        optimization_score = self._calculate_optimization_score(sorted_tasks, maintenance_windows)
        
        return MaintenanceSchedule(
            scheduled_tasks=sorted_tasks,
            maintenance_windows=maintenance_windows,
            resource_allocation=resource_allocation,
            total_downtime_estimate=total_downtime,
            schedule_optimization_score=optimization_score
        )
    
    async def optimize_maintenance_schedule(self, schedule: MaintenanceSchedule) -> MaintenanceSchedule:
        """Optimize maintenance schedule for minimal disruption"""
        optimized_tasks = []
        
        # Group related tasks
        task_groups = self._group_related_tasks(schedule.scheduled_tasks)
        
        # Optimize each group
        for group in task_groups:
            optimized_group = self._optimize_task_group(group)
            optimized_tasks.extend(optimized_group)
        
        # Recreate schedule with optimized tasks
        return await self.schedule_maintenance_window(optimized_tasks)
    
    def _determine_maintenance_type(self, prediction: FailurePrediction) -> MaintenanceType:
        """Determine appropriate maintenance type"""
        if prediction.probability > 0.8:
            return MaintenanceType.EMERGENCY
        elif prediction.time_to_failure_hours < 24:
            return MaintenanceType.CORRECTIVE
        elif prediction.confidence_score > 0.7:
            return MaintenanceType.PREDICTIVE
        else:
            return MaintenanceType.PREVENTIVE
    
    def _determine_priority(self, prediction: FailurePrediction) -> MaintenancePriority:
        """Determine maintenance priority"""
        if prediction.severity_impact == "critical" or prediction.probability > 0.9:
            return MaintenancePriority.CRITICAL
        elif prediction.severity_impact == "high" or prediction.probability > 0.7:
            return MaintenancePriority.HIGH
        elif prediction.probability > 0.5:
            return MaintenancePriority.MEDIUM
        else:
            return MaintenancePriority.LOW
    
    async def _create_task_for_failure_type(self, failure_type: FailureType, 
                                          maintenance_type: MaintenanceType,
                                          priority: MaintenancePriority,
                                          prediction: FailurePrediction,
                                          task_id: int) -> Optional[MaintenanceTask]:
        """Create specific maintenance task for failure type"""
        base_time = datetime.utcnow()
        
        # Schedule task based on urgency
        if priority == MaintenancePriority.CRITICAL:
            scheduled_time = base_time + timedelta(hours=2)
            deadline = base_time + timedelta(hours=12)
        elif priority == MaintenancePriority.HIGH:
            scheduled_time = base_time + timedelta(hours=12)
            deadline = base_time + timedelta(hours=48)
        else:
            scheduled_time = base_time + timedelta(days=1)
            deadline = base_time + timedelta(days=7)
        
        # Create task based on failure type
        task_templates = {
            FailureType.HARDWARE_FAILURE: {
                "description": "Hardware diagnostics and replacement if needed",
                "duration": 120,
                "resources": ["hardware_technician", "replacement_parts"],
                "prerequisites": ["system_backup", "downtime_approval"]
            },
            FailureType.MEMORY_EXHAUSTION: {
                "description": "Memory optimization and cleanup",
                "duration": 60,
                "resources": ["system_administrator", "monitoring_tools"],
                "prerequisites": ["memory_analysis", "process_identification"]
            },
            FailureType.DISK_FAILURE: {
                "description": "Disk space cleanup and health check",
                "duration": 90,
                "resources": ["system_administrator", "storage_tools"],
                "prerequisites": ["data_backup", "disk_analysis"]
            },
            FailureType.SOFTWARE_CRASH: {
                "description": "Software update and configuration optimization",
                "duration": 45,
                "resources": ["software_engineer", "testing_environment"],
                "prerequisites": ["code_review", "testing_plan"]
            },
            FailureType.DATABASE_CORRUPTION: {
                "description": "Database integrity check and repair",
                "duration": 180,
                "resources": ["database_administrator", "backup_systems"],
                "prerequisites": ["database_backup", "integrity_analysis"]
            },
            FailureType.PERFORMANCE_DEGRADATION: {
                "description": "Performance tuning and optimization",
                "duration": 90,
                "resources": ["performance_engineer", "profiling_tools"],
                "prerequisites": ["performance_baseline", "bottleneck_analysis"]
            },
            FailureType.NETWORK_FAILURE: {
                "description": "Network diagnostics and repair",
                "duration": 75,
                "resources": ["network_administrator", "network_tools"],
                "prerequisites": ["network_analysis", "connectivity_test"]
            }
        }
        
        template = task_templates.get(failure_type)
        if not template:
            return None
        
        return MaintenanceTask(
            task_id=f"MAINT-{task_id:04d}",
            task_type=maintenance_type,
            priority=priority,
            description=template["description"],
            estimated_duration_minutes=template["duration"],
            required_resources=template["resources"],
            prerequisites=template["prerequisites"],
            expected_outcomes=prediction.recommended_actions,
            scheduled_time=scheduled_time,
            deadline=deadline
        )
    
    def _priority_score(self, priority: MaintenancePriority) -> int:
        """Convert priority to numeric score for sorting"""
        scores = {
            MaintenancePriority.CRITICAL: 4,
            MaintenancePriority.HIGH: 3,
            MaintenancePriority.MEDIUM: 2,
            MaintenancePriority.LOW: 1
        }
        return scores.get(priority, 0)
    
    def _create_maintenance_windows(self, tasks: List[MaintenanceTask]) -> List[Dict[str, Any]]:
        """Create maintenance windows for tasks"""
        windows = []
        current_window: Optional[Dict[str, Any]] = None
        
        for task in tasks:
            if (current_window is None or 
                task.scheduled_time - current_window["end_time"] > timedelta(hours=4)):
                # Create new window
                if current_window:
                    windows.append(current_window)
                
                current_window = {
                    "window_id": f"MW-{len(windows) + 1:03d}",
                    "start_time": task.scheduled_time,
                    "end_time": task.scheduled_time + timedelta(minutes=task.estimated_duration_minutes),
                    "tasks": [task.task_id],
                    "total_duration": task.estimated_duration_minutes
                }
            else:
                # Add to existing window
                current_window["tasks"].append(task.task_id)
                current_window["end_time"] = task.scheduled_time + timedelta(minutes=task.estimated_duration_minutes)
                current_window["total_duration"] += task.estimated_duration_minutes
        
        if current_window:
            windows.append(current_window)
        
        return windows
    
    def _allocate_resources(self, tasks: List[MaintenanceTask]) -> Dict[str, Any]:
        """Allocate resources to maintenance tasks"""
        resource_usage = {}
        
        for task in tasks:
            for resource in task.required_resources:
                if resource not in resource_usage:
                    resource_usage[resource] = []
                
                resource_usage[resource].append({
                    "task_id": task.task_id,
                    "start_time": task.scheduled_time,
                    "duration": task.estimated_duration_minutes
                })
        
        return {
            "resource_assignments": resource_usage,
            "resource_conflicts": self._detect_resource_conflicts(resource_usage),
            "utilization_rate": self._calculate_resource_utilization(resource_usage)
        }
    
    def _calculate_optimization_score(self, tasks: List[MaintenanceTask], 
                                    windows: List[Dict[str, Any]]) -> float:
        """Calculate schedule optimization score"""
        if not tasks:
            return 1.0
        
        # Factors for optimization score
        factors = []
        
        # Task grouping efficiency (fewer windows is better)
        total_duration = sum(task.estimated_duration_minutes for task in tasks)
        window_efficiency = total_duration / (len(windows) * 60) if windows else 0
        factors.append(min(window_efficiency, 1.0))
        
        # Priority handling (critical tasks scheduled earlier)
        priority_score = self._calculate_priority_adherence(tasks)
        factors.append(priority_score)
        
        # Resource utilization efficiency
        resource_efficiency = 0.8  # Placeholder
        factors.append(resource_efficiency)
        
        return statistics.mean(factors) if factors else 0.0
    
    def _group_related_tasks(self, tasks: List[MaintenanceTask]) -> List[List[MaintenanceTask]]:
        """Group related maintenance tasks"""
        groups = []
        ungrouped_tasks = tasks.copy()
        
        while ungrouped_tasks:
            current_task = ungrouped_tasks.pop(0)
            current_group = [current_task]
            
            # Find related tasks
            remaining_tasks = []
            for task in ungrouped_tasks:
                if self._are_tasks_related(current_task, task):
                    current_group.append(task)
                else:
                    remaining_tasks.append(task)
            
            ungrouped_tasks = remaining_tasks
            groups.append(current_group)
        
        return groups
    
    def _optimize_task_group(self, task_group: List[MaintenanceTask]) -> List[MaintenanceTask]:
        """Optimize a group of related tasks"""
        # Sort by priority and dependencies
        optimized = sorted(task_group, key=lambda t: (
            -self._priority_score(t.priority),
            len(t.prerequisites)
        ))
        
        return optimized
    
    def _are_tasks_related(self, task1: MaintenanceTask, task2: MaintenanceTask) -> bool:
        """Check if two tasks are related and can be grouped"""
        # Tasks are related if they share resources or have similar timing
        shared_resources = set(task1.required_resources) & set(task2.required_resources)
        time_diff = abs((task1.scheduled_time - task2.scheduled_time).total_seconds() / 3600)
        
        return len(shared_resources) > 0 or time_diff < 2  # Within 2 hours
    
    def _detect_resource_conflicts(self, resource_usage: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Detect resource scheduling conflicts"""
        conflicts = []
        
        for resource, assignments in resource_usage.items():
            sorted_assignments = sorted(assignments, key=lambda x: x["start_time"])
            
            for i in range(len(sorted_assignments) - 1):
                current = sorted_assignments[i]
                next_assignment = sorted_assignments[i + 1]
                
                current_end = current["start_time"] + timedelta(minutes=current["duration"])
                if current_end > next_assignment["start_time"]:
                    conflicts.append(f"Resource '{resource}' conflict between {current['task_id']} and {next_assignment['task_id']}")
        
        return conflicts
    
    def _calculate_resource_utilization(self, resource_usage: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate overall resource utilization rate"""
        if not resource_usage:
            return 0.0
        
        utilization_rates = []
        
        for _resource, assignments in resource_usage.items():
            if assignments:
                total_assigned_time = sum(assignment["duration"] for assignment in assignments)
                # Assume 8-hour work day for utilization calculation
                available_time = 8 * 60  # minutes
                utilization = min(total_assigned_time / available_time, 1.0)
                utilization_rates.append(utilization)
        
        return statistics.mean(utilization_rates) if utilization_rates else 0.0
    
    def _calculate_priority_adherence(self, tasks: List[MaintenanceTask]) -> float:
        """Calculate how well the schedule adheres to task priorities"""
        if not tasks:
            return 1.0
        
        # Check if higher priority tasks are scheduled earlier
        adherence_scores = []
        
        for i, task in enumerate(tasks):
            expected_position = self._priority_score(task.priority) / 4.0  # Normalize to 0-1
            actual_position = 1.0 - (i / len(tasks))  # Higher position = scheduled earlier
            
            adherence = 1.0 - abs(expected_position - actual_position)
            adherence_scores.append(adherence)
        
        return statistics.mean(adherence_scores)


class PredictiveMaintenanceSystem:
    """Predictive maintenance with preemptive issue resolution"""
    
    def __init__(self):
        self.failure_predictor = FailurePredictor()
        self.maintenance_scheduler = MaintenanceScheduler()
        self.logger = logging.getLogger(__name__)
        
    async def predict_maintenance_needs(self) -> MaintenancePredictions:
        """Predict maintenance needs before failures occur"""
        try:
            # Get system metrics (placeholder - would integrate with actual monitoring)
            system_metrics = await self._get_system_metrics()
            performance_trends = await self._get_performance_trends()
            
            # Predict different types of failures
            hardware_predictions = await self.failure_predictor.predict_hardware_failures(system_metrics)
            software_predictions = await self.failure_predictor.predict_software_failures(system_metrics)
            performance_predictions = await self.failure_predictor.predict_performance_degradation(performance_trends)
            
            # Combine all predictions
            all_predictions = hardware_predictions + software_predictions + performance_predictions
            
            # Create maintenance tasks
            maintenance_tasks = await self.maintenance_scheduler.create_maintenance_tasks(all_predictions)
            
            # Calculate overall risk score
            risk_score = self._calculate_overall_risk_score(all_predictions)
            
            # Determine next maintenance window
            next_window = self._determine_next_maintenance_window(maintenance_tasks)
            
            # Assess system health trend
            health_trend = self._assess_health_trend(all_predictions)
            
            return MaintenancePredictions(
                predicted_issues=all_predictions,
                recommended_tasks=maintenance_tasks,
                overall_risk_score=risk_score,
                next_maintenance_window=next_window,
                system_health_trend=health_trend
            )
            
        except Exception as e:
            self.logger.error("Error predicting maintenance needs: %s", str(e))
            # Return empty predictions on error
            return MaintenancePredictions(
                predicted_issues=[],
                recommended_tasks=[],
                overall_risk_score=0.0,
                next_maintenance_window=datetime.utcnow() + timedelta(days=7),
                system_health_trend="unknown"
            )
    
    async def schedule_preemptive_maintenance(self, predictions: MaintenancePredictions) -> MaintenanceSchedule:
        """Schedule preemptive maintenance to prevent failures"""
        try:
            # Create initial schedule
            initial_schedule = await self.maintenance_scheduler.schedule_maintenance_window(
                predictions.recommended_tasks
            )
            
            # Optimize schedule
            optimized_schedule = await self.maintenance_scheduler.optimize_maintenance_schedule(
                initial_schedule
            )
            
            return optimized_schedule
            
        except Exception as e:
            self.logger.error("Error scheduling preemptive maintenance: %s", str(e))
            # Return empty schedule on error
            return MaintenanceSchedule(
                scheduled_tasks=[],
                maintenance_windows=[],
                resource_allocation={},
                total_downtime_estimate=0,
                schedule_optimization_score=0.0
            )
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics for analysis"""
        # Placeholder implementation - would integrate with health monitor
        return {
            "cpu_usage": 65.0,
            "memory_usage": 78.0,
            "disk_usage": 85.0,
            "network_usage": 45.0,
            "error_rate": 0.02,
            "db_connections": 45,
            "max_db_connections": 100
        }
    
    async def _get_performance_trends(self) -> Dict[str, List[float]]:
        """Get performance trends for analysis"""
        # Placeholder implementation - would integrate with performance monitor
        return {
            "response_times": [100, 120, 135, 150, 180, 200],
            "throughput": [1000, 950, 900, 850, 800, 750],
            "error_rates": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035]
        }
    
    def _calculate_overall_risk_score(self, predictions: List[FailurePrediction]) -> float:
        """Calculate overall system risk score"""
        if not predictions:
            return 0.0
        
        # Weight predictions by probability and impact
        weighted_risks = []
        
        severity_weights = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
        for prediction in predictions:
            weight = severity_weights.get(prediction.severity_impact, 0.5)
            risk_contribution = prediction.probability * weight * prediction.confidence_score
            weighted_risks.append(risk_contribution)
        
        return min(statistics.mean(weighted_risks) if weighted_risks else 0.0, 1.0)
    
    def _determine_next_maintenance_window(self, tasks: List[MaintenanceTask]) -> datetime:
        """Determine next optimal maintenance window"""
        if not tasks:
            return datetime.utcnow() + timedelta(days=7)
        
        # Find earliest critical task
        critical_tasks = [t for t in tasks if t.priority == MaintenancePriority.CRITICAL]
        if critical_tasks:
            return min(task.scheduled_time for task in critical_tasks)
        
        # Otherwise, find earliest high priority task
        high_priority_tasks = [t for t in tasks if t.priority == MaintenancePriority.HIGH]
        if high_priority_tasks:
            return min(task.scheduled_time for task in high_priority_tasks)
        
        # Default to next business hours
        next_window = datetime.utcnow().replace(hour=2, minute=0, second=0, microsecond=0)
        if next_window <= datetime.utcnow():
            next_window += timedelta(days=1)
        
        return next_window
    
    def _assess_health_trend(self, predictions: List[FailurePrediction]) -> str:
        """Assess overall system health trend"""
        if not predictions:
            return "stable"
        
        critical_predictions = [p for p in predictions if p.severity_impact == "critical"]
        high_predictions = [p for p in predictions if p.severity_impact == "high"]
        
        if len(critical_predictions) > 2:
            return "deteriorating_rapidly"
        elif len(critical_predictions) > 0 or len(high_predictions) > 3:
            return "declining"
        elif len(high_predictions) > 0:
            return "concerning"
        else:
            return "stable"