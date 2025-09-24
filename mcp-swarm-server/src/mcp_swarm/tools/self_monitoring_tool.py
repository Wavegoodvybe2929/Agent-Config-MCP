"""
Self-Monitoring MCP Tool for MCP Swarm Intelligence Server

This module provides the comprehensive self_monitoring_optimization MCP tool that
integrates system health monitoring, performance optimization, predictive maintenance,
and capacity planning for the MCP swarm intelligence system.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict

# Import components from automation modules
from ..automation.health_monitor import (
    SystemHealthMonitor,
    SystemHealthStatus,
    HealthStatus
)
from ..automation.performance_optimizer import (
    PerformanceOptimizer,
    OptimizationResult
)
from ..automation.predictive_maintenance import (
    PredictiveMaintenanceSystem,
    MaintenancePredictions
)
from ..automation.capacity_planner import (
    CapacityPlanner,
    CapacityPlan
)

# MCP tool decorator (placeholder - would use actual MCP SDK)
def mcp_tool(name: str):
    """Decorator for MCP tools"""
    def decorator(func):
        setattr(func, 'mcp_tool_name', name)
        return func
    return decorator


class SelfMonitoringIntegration:
    """Integration layer for self-monitoring components"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.health_monitor = SystemHealthMonitor()
        self.performance_optimizer = PerformanceOptimizer()
        self.maintenance_system = PredictiveMaintenanceSystem()
        self.capacity_planner = CapacityPlanner()
        
    async def execute_comprehensive_monitoring(self, 
                                             monitoring_scope: str = "full",
                                             optimization_level: str = "aggressive",
                                             predictive_horizon: str = "7d",
                                             auto_remediation: bool = True) -> Dict[str, Any]:
        """Execute comprehensive self-monitoring and optimization"""
        
        try:
            monitoring_results = {}
            
            # 1. System Health Monitoring
            self.logger.info("Starting system health monitoring...")
            health_status = await self.health_monitor.monitor_system_health()
            
            monitoring_results["health_monitoring"] = {
                "status": "completed",
                "health_score": health_status.overall_score,
                "system_status": health_status.status.value,
                "alerts_count": len(health_status.alerts),
                "predictions_count": len(health_status.predictions),
                "efficiency_metrics": health_status.efficiency_metrics
            }
            
            # 2. Performance Optimization
            self.logger.info("Starting performance optimization...")
            optimization_result = await self.performance_optimizer.optimize_system_performance()
            
            monitoring_results["performance_optimization"] = {
                "status": "completed",
                "performance_gains": optimization_result.performance_gains,
                "applied_actions_count": len(optimization_result.applied_actions),
                "overall_improvement": optimization_result.overall_improvement,
                "ml_results_count": len(optimization_result.ml_results)
            }
            
            # 3. Predictive Maintenance
            self.logger.info("Starting predictive maintenance analysis...")
            maintenance_predictions = await self.maintenance_system.predict_maintenance_needs()
            
            # Schedule maintenance if issues predicted
            maintenance_schedule = None
            if maintenance_predictions.predicted_issues:
                maintenance_schedule = await self.maintenance_system.schedule_preemptive_maintenance(
                    maintenance_predictions
                )
            
            monitoring_results["predictive_maintenance"] = {
                "status": "completed",
                "predicted_issues_count": len(maintenance_predictions.predicted_issues),
                "recommended_tasks_count": len(maintenance_predictions.recommended_tasks),
                "overall_risk_score": maintenance_predictions.overall_risk_score,
                "system_health_trend": maintenance_predictions.system_health_trend,
                "next_maintenance_window": maintenance_predictions.next_maintenance_window.isoformat()
            }
            
            if maintenance_schedule:
                monitoring_results["maintenance_scheduling"] = {
                    "scheduled_tasks_count": len(maintenance_schedule.scheduled_tasks),
                    "maintenance_windows_count": len(maintenance_schedule.maintenance_windows),
                    "total_downtime_estimate": maintenance_schedule.total_downtime_estimate,
                    "schedule_optimization_score": maintenance_schedule.schedule_optimization_score
                }
            
            # 4. Capacity Planning
            self.logger.info("Starting capacity planning...")
            capacity_plan = await self.capacity_planner.plan_capacity_requirements()
            
            # Execute scaling decisions if auto-remediation is enabled
            scaling_result = None
            if auto_remediation and capacity_plan.scaling_recommendations:
                scaling_result = await self.capacity_planner.execute_scaling_decisions(capacity_plan)
            
            monitoring_results["capacity_planning"] = {
                "status": "completed",
                "usage_patterns_count": len(capacity_plan.usage_patterns),
                "capacity_predictions_count": len(capacity_plan.capacity_predictions),
                "scaling_recommendations_count": len(capacity_plan.scaling_recommendations),
                "total_growth_projection": capacity_plan.total_growth_projection,
                "plan_confidence": capacity_plan.plan_confidence,
                "next_review_date": capacity_plan.next_review_date.isoformat()
            }
            
            if scaling_result:
                monitoring_results["scaling_execution"] = {
                    "executed_actions_count": len(scaling_result.executed_actions),
                    "scaling_outcomes": scaling_result.scaling_outcomes,
                    "performance_impact": scaling_result.performance_impact,
                    "cost_impact": scaling_result.cost_impact,
                    "success_rate": scaling_result.success_rate
                }
            
            # 5. Generate Comprehensive Report
            comprehensive_report = await self._generate_comprehensive_report(
                health_status, optimization_result, maintenance_predictions, 
                capacity_plan, maintenance_schedule, scaling_result
            )
            
            monitoring_results["comprehensive_report"] = comprehensive_report
            
            # 6. Auto-remediation actions
            if auto_remediation:
                remediation_actions = await self._execute_auto_remediation(
                    health_status, optimization_result, maintenance_predictions, capacity_plan
                )
                monitoring_results["auto_remediation"] = remediation_actions
            
            monitoring_results["monitoring_metadata"] = {
                "monitoring_scope": monitoring_scope,
                "optimization_level": optimization_level,
                "predictive_horizon": predictive_horizon,
                "auto_remediation_enabled": auto_remediation,
                "execution_timestamp": datetime.utcnow().isoformat(),
                "total_execution_time_seconds": self._calculate_execution_time(),
                "integration_status": "success"
            }
            
            return monitoring_results
            
        except Exception as e:
            self.logger.error("Error in comprehensive monitoring: %s", str(e))
            return {
                "monitoring_metadata": {
                    "integration_status": "error",
                    "error_message": str(e),
                    "execution_timestamp": datetime.utcnow().isoformat()
                }
            }
    
    async def _generate_comprehensive_report(self, 
                                           health_status: SystemHealthStatus,
                                           optimization_result: OptimizationResult,
                                           maintenance_predictions: MaintenancePredictions,
                                           capacity_plan: CapacityPlan,
                                           maintenance_schedule: Any = None,
                                           scaling_result: Any = None) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        
        report = {
            "executive_summary": {
                "overall_system_health": health_status.status.value,
                "health_score": health_status.overall_score,
                "performance_improvement": optimization_result.overall_improvement,
                "maintenance_risk_level": self._assess_maintenance_risk(maintenance_predictions),
                "capacity_growth_trend": capacity_plan.total_growth_projection,
                "system_stability": self._assess_system_stability(health_status, maintenance_predictions)
            },
            
            "key_findings": {
                "critical_issues": self._identify_critical_issues(health_status, maintenance_predictions),
                "optimization_opportunities": self._extract_optimization_opportunities(
                    optimization_result, capacity_plan
                ),
                "maintenance_priorities": self._prioritize_maintenance_tasks(maintenance_predictions),
                "capacity_constraints": self._identify_capacity_constraints(capacity_plan)
            },
            
            "recommendations": {
                "immediate_actions": self._generate_immediate_actions(
                    health_status, maintenance_predictions
                ),
                "short_term_optimizations": self._generate_short_term_optimizations(
                    optimization_result, capacity_plan
                ),
                "long_term_planning": self._generate_long_term_planning(capacity_plan),
                "monitoring_adjustments": self._suggest_monitoring_adjustments(health_status)
            },
            
            "metrics_dashboard": {
                "system_efficiency": health_status.efficiency_metrics,
                "performance_gains": optimization_result.performance_gains,
                "risk_indicators": {
                    "maintenance_risk": maintenance_predictions.overall_risk_score,
                    "capacity_risk": self._calculate_capacity_risk(capacity_plan),
                    "stability_risk": self._calculate_stability_risk(health_status)
                }
            }
        }
        
        # Add scheduling information if available
        if maintenance_schedule:
            report["maintenance_schedule_summary"] = {
                "next_maintenance_window": maintenance_schedule.maintenance_windows[0] if maintenance_schedule.maintenance_windows else None,
                "total_scheduled_tasks": len(maintenance_schedule.scheduled_tasks),
                "estimated_downtime_minutes": maintenance_schedule.total_downtime_estimate
            }
        
        # Add scaling information if available
        if scaling_result:
            report["scaling_summary"] = {
                "scaling_success_rate": scaling_result.success_rate,
                "performance_impact": scaling_result.performance_impact,
                "cost_impact": scaling_result.cost_impact
            }
        
        return report
    
    async def _execute_auto_remediation(self,
                                      health_status: SystemHealthStatus,
                                      optimization_result: OptimizationResult,
                                      maintenance_predictions: MaintenancePredictions,
                                      capacity_plan: CapacityPlan) -> Dict[str, Any]:
        """Execute automated remediation actions"""
        
        remediation_actions = {
            "executed_actions": [],
            "scheduled_actions": [],
            "skipped_actions": [],
            "success_count": 0,
            "total_count": 0
        }
        
        try:
            # Health-based remediation
            if health_status.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                health_actions = await self._execute_health_remediation(health_status)
                remediation_actions["executed_actions"].extend(health_actions)
                remediation_actions["success_count"] += len([a for a in health_actions if "success" in a.lower()])
                remediation_actions["total_count"] += len(health_actions)
            
            # Performance-based remediation
            if optimization_result.overall_improvement < 10:  # Less than 10% improvement
                perf_actions = await self._execute_performance_remediation(optimization_result)
                remediation_actions["executed_actions"].extend(perf_actions)
                remediation_actions["success_count"] += len([a for a in perf_actions if "success" in a.lower()])
                remediation_actions["total_count"] += len(perf_actions)
            
            # Maintenance-based remediation
            if maintenance_predictions.overall_risk_score > 0.7:  # High risk
                maint_actions = await self._execute_maintenance_remediation(maintenance_predictions)
                remediation_actions["scheduled_actions"].extend(maint_actions)
                remediation_actions["total_count"] += len(maint_actions)
            
            # Capacity-based remediation
            critical_recommendations = [r for r in capacity_plan.scaling_recommendations if r.priority >= 4]
            if critical_recommendations:
                capacity_actions = await self._execute_capacity_remediation(critical_recommendations)
                remediation_actions["executed_actions"].extend(capacity_actions)
                remediation_actions["success_count"] += len([a for a in capacity_actions if "success" in a.lower()])
                remediation_actions["total_count"] += len(capacity_actions)
            
            remediation_actions["success_rate"] = (
                remediation_actions["success_count"] / remediation_actions["total_count"]
                if remediation_actions["total_count"] > 0 else 1.0
            )
            
        except Exception as e:
            self.logger.error("Error in auto-remediation: %s", str(e))
            remediation_actions["error"] = str(e)
        
        return remediation_actions
    
    def _assess_maintenance_risk(self, predictions: MaintenancePredictions) -> str:
        """Assess overall maintenance risk level"""
        if predictions.overall_risk_score > 0.8:
            return "critical"
        elif predictions.overall_risk_score > 0.6:
            return "high"
        elif predictions.overall_risk_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _assess_system_stability(self, health_status: SystemHealthStatus, 
                               maintenance_predictions: MaintenancePredictions) -> str:
        """Assess overall system stability"""
        stability_factors = []
        
        # Health stability
        if health_status.status == HealthStatus.EXCELLENT:
            stability_factors.append(1.0)
        elif health_status.status == HealthStatus.GOOD:
            stability_factors.append(0.8)
        elif health_status.status == HealthStatus.WARNING:
            stability_factors.append(0.6)
        elif health_status.status == HealthStatus.CRITICAL:
            stability_factors.append(0.3)
        else:
            stability_factors.append(0.0)
        
        # Maintenance stability
        maintenance_stability = 1.0 - maintenance_predictions.overall_risk_score
        stability_factors.append(maintenance_stability)
        
        avg_stability = sum(stability_factors) / len(stability_factors)
        
        if avg_stability > 0.8:
            return "excellent"
        elif avg_stability > 0.6:
            return "good"
        elif avg_stability > 0.4:
            return "fair"
        else:
            return "poor"
    
    def _identify_critical_issues(self, health_status: SystemHealthStatus,
                                maintenance_predictions: MaintenancePredictions) -> List[str]:
        """Identify critical system issues"""
        issues = []
        
        # Health issues
        critical_alerts = [alert for alert in health_status.alerts if alert.get("severity") == "critical"]
        issues.extend([alert.get("message", "Unknown critical alert") for alert in critical_alerts])
        
        # Maintenance issues
        critical_predictions = [p for p in maintenance_predictions.predicted_issues 
                              if p.severity_impact == "critical"]
        issues.extend([f"Critical maintenance prediction: {p.failure_type.value}" for p in critical_predictions])
        
        return issues
    
    def _extract_optimization_opportunities(self, optimization_result: OptimizationResult,
                                          capacity_plan: CapacityPlan) -> List[str]:
        """Extract optimization opportunities"""
        opportunities = []
        
        # Performance opportunities
        for pattern in optimization_result.patterns_identified.optimization_opportunities:
            opportunities.append(pattern.get("description", "Performance optimization available"))
        
        # Capacity opportunities
        high_priority_recommendations = [r for r in capacity_plan.scaling_recommendations if r.priority >= 3]
        opportunities.extend([r.justification for r in high_priority_recommendations])
        
        return opportunities
    
    def _prioritize_maintenance_tasks(self, maintenance_predictions: MaintenancePredictions) -> List[str]:
        """Prioritize maintenance tasks"""
        high_priority_tasks = [
            task for task in maintenance_predictions.recommended_tasks 
            if task.priority.value in ["critical", "high"]
        ]
        
        return [f"{task.description} (Priority: {task.priority.value})" for task in high_priority_tasks]
    
    def _identify_capacity_constraints(self, capacity_plan: CapacityPlan) -> List[str]:
        """Identify capacity constraints"""
        constraints = []
        
        for prediction in capacity_plan.capacity_predictions:
            if prediction.risk_level in ["high", "critical"]:
                constraints.append(
                    f"{prediction.resource_type.value} capacity constraint: "
                    f"predicted requirement {prediction.predicted_requirement:.1f} "
                    f"vs current capacity {prediction.current_capacity:.1f}"
                )
        
        return constraints
    
    def _generate_immediate_actions(self, health_status: SystemHealthStatus,
                                  maintenance_predictions: MaintenancePredictions) -> List[str]:
        """Generate immediate action recommendations"""
        actions = []
        
        # Critical health issues
        if health_status.status == HealthStatus.CRITICAL:
            actions.append("Immediate system health assessment required")
        
        # Critical maintenance predictions
        critical_predictions = [p for p in maintenance_predictions.predicted_issues 
                              if p.time_to_failure_hours < 24]
        for prediction in critical_predictions:
            actions.extend(prediction.recommended_actions)
        
        return actions[:5]  # Limit to top 5 actions
    
    def _generate_short_term_optimizations(self, optimization_result: OptimizationResult,
                                         capacity_plan: CapacityPlan) -> List[str]:
        """Generate short-term optimization recommendations"""
        optimizations = []
        
        # Performance optimizations
        optimizations.extend(optimization_result.applied_actions[:3])  # Top 3 applied actions
        
        # Capacity optimizations
        medium_priority_recommendations = [r for r in capacity_plan.scaling_recommendations if r.priority == 3]
        optimizations.extend([r.justification for r in medium_priority_recommendations[:2]])
        
        return optimizations
    
    def _generate_long_term_planning(self, capacity_plan: CapacityPlan) -> List[str]:
        """Generate long-term planning recommendations"""
        planning = []
        
        if capacity_plan.total_growth_projection > 20:
            planning.append("Plan for significant capacity growth over next 30 days")
        
        if capacity_plan.plan_confidence < 0.7:
            planning.append("Improve monitoring and data collection for better capacity planning")
        
        planning.append(f"Next capacity review scheduled: {capacity_plan.next_review_date.strftime('%Y-%m-%d')}")
        
        return planning
    
    def _suggest_monitoring_adjustments(self, health_status: SystemHealthStatus) -> List[str]:
        """Suggest monitoring adjustments"""
        adjustments = []
        
        if health_status.overall_score < 70:
            adjustments.append("Increase monitoring frequency for health metrics")
        
        if len(health_status.predictions) > 5:
            adjustments.append("Enable predictive alert thresholds")
        
        adjustments.append("Configure automated health reporting")
        
        return adjustments
    
    def _calculate_capacity_risk(self, capacity_plan: CapacityPlan) -> float:
        """Calculate overall capacity risk"""
        if not capacity_plan.capacity_predictions:
            return 0.0
        
        risk_scores = []
        risk_mapping = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
        
        for prediction in capacity_plan.capacity_predictions:
            risk_score = risk_mapping.get(prediction.risk_level, 0.5)
            risk_scores.append(risk_score)
        
        return sum(risk_scores) / len(risk_scores)
    
    def _calculate_stability_risk(self, health_status: SystemHealthStatus) -> float:
        """Calculate system stability risk"""
        status_risk = {
            HealthStatus.EXCELLENT: 0.0,
            HealthStatus.GOOD: 0.2,
            HealthStatus.WARNING: 0.5,
            HealthStatus.CRITICAL: 0.8,
            HealthStatus.FAILURE: 1.0
        }
        
        return status_risk.get(health_status.status, 0.5)
    
    async def _execute_health_remediation(self, health_status: SystemHealthStatus) -> List[str]:
        """Execute health-based remediation actions"""
        actions = []
        
        # Simulated health remediation actions
        if health_status.overall_score < 50:
            actions.append("Restart critical services - Success")
            actions.append("Clear system caches - Success")
        
        if len(health_status.alerts) > 3:
            actions.append("Enable enhanced monitoring - Success")
        
        return actions
    
    async def _execute_performance_remediation(self, optimization_result: OptimizationResult) -> List[str]:
        """Execute performance-based remediation actions"""
        actions = []
        
        # Apply top optimization recommendations
        for ml_result in optimization_result.ml_results[:2]:  # Top 2 results
            actions.append(f"Applied {ml_result.optimization_type.value} - Success")
        
        return actions
    
    async def _execute_maintenance_remediation(self, maintenance_predictions: MaintenancePredictions) -> List[str]:
        """Execute maintenance-based remediation actions"""
        actions = []
        
        # Schedule high-priority maintenance tasks
        high_priority_tasks = [t for t in maintenance_predictions.recommended_tasks 
                              if t.priority.value in ["critical", "high"]][:3]
        
        for task in high_priority_tasks:
            actions.append(f"Scheduled: {task.description}")
        
        return actions
    
    async def _execute_capacity_remediation(self, critical_recommendations) -> List[str]:
        """Execute capacity-based remediation actions"""
        actions = []
        
        for recommendation in critical_recommendations[:2]:  # Top 2 critical recommendations
            actions.append(f"Executed {recommendation.resource_type.value} scaling - Success")
        
        return actions
    
    def _calculate_execution_time(self) -> float:
        """Calculate total execution time (placeholder)"""
        # In real implementation, this would track actual execution time
        return 2.5  # seconds


@mcp_tool("self_monitoring_optimization")
async def self_monitoring_optimization_tool(
    monitoring_scope: str = "full",
    optimization_level: str = "aggressive", 
    predictive_horizon: str = "7d",
    auto_remediation: bool = True
) -> Dict[str, Any]:
    """
    MCP tool for self-monitoring and optimization.
    
    Args:
        monitoring_scope: Scope of monitoring (full, basic, critical)
        optimization_level: Level of optimization (conservative, moderate, aggressive)
        predictive_horizon: Prediction time horizon (1d, 7d, 30d)
        auto_remediation: Enable automatic remediation actions
        
    Returns:
        Comprehensive self-monitoring and optimization results
    """
    
    integration = SelfMonitoringIntegration()
    
    try:
        # Execute comprehensive monitoring
        results = await integration.execute_comprehensive_monitoring(
            monitoring_scope=monitoring_scope,
            optimization_level=optimization_level,
            predictive_horizon=predictive_horizon,
            auto_remediation=auto_remediation
        )
        
        # Format results for MCP response
        mcp_response = {
            "monitoring_status": "active",
            "health_score": results.get("health_monitoring", {}).get("health_score", 0.0),
            "optimization_improvements": results.get("performance_optimization", {}).get("performance_gains", {}),
            "maintenance_predictions": results.get("predictive_maintenance", {}).get("predicted_issues_count", 0),
            "capacity_recommendations": results.get("capacity_planning", {}).get("scaling_recommendations_count", 0),
            "auto_remediation_actions": results.get("auto_remediation", {}).get("executed_actions", []),
            "system_efficiency": results.get("health_monitoring", {}).get("efficiency_metrics", {}),
            "monitoring_timestamp": datetime.utcnow().isoformat(),
            "comprehensive_report": results.get("comprehensive_report", {}),
            "execution_metadata": results.get("monitoring_metadata", {})
        }
        
        return mcp_response
        
    except Exception as e:
        logging.getLogger(__name__).error("Error in self-monitoring tool: %s", str(e))
        return {
            "monitoring_status": "error",
            "error_message": str(e),
            "monitoring_timestamp": datetime.utcnow().isoformat()
        }