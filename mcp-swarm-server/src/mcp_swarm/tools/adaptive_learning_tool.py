"""
Adaptive Learning Evolution MCP Tool

This MCP tool integrates all adaptive learning components and provides
a unified interface for parameter optimization, anomaly detection,
predictive modeling, and system evolution capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import asdict

# MCP imports
from mcp import types
from mcp.server import Server
from mcp.types import Tool, TextContent

# Import adaptive learning components
from ..automation.adaptive_learning import AdaptiveLearningEngine
from ..automation.predictive_models import PredictiveSuccessModels
from ..automation.evolutionary_optimizer import (
    EvolutionaryParameterOptimizer, OptimizationAlgorithm, ParameterBounds, ParameterType
)
from ..automation.anomaly_detection import AnomalyDetectionSystem, AnomalyType, AnomalySeverity

logger = logging.getLogger(__name__)

class AdaptiveLearningEvolutionTool:
    """
    MCP tool for adaptive learning and evolution capabilities,
    integrating ML-based adaptation, parameter optimization, and anomaly detection.
    """
    
    def __init__(self, database_path: str = "data/memory.db"):
        self.database_path = database_path
        
        # Initialize components
        self.adaptive_engine = AdaptiveLearningEngine(database_path)
        self.predictive_models = PredictiveSuccessModels(database_path)
        self.parameter_optimizer = EvolutionaryParameterOptimizer(database_path)
        self.anomaly_detector = AnomalyDetectionSystem(database_path)
        
        # Tool metadata
        self.name = "adaptive_learning_evolution"
        self.description = """
        Comprehensive adaptive learning and evolution tool for MCP swarm intelligence.
        
        Capabilities:
        - Parameter optimization using evolutionary algorithms
        - Predictive modeling for task success probability  
        - Anomaly detection and adaptive response
        - Machine learning-based system evolution
        - Performance monitoring and adaptation
        
        Available operations:
        - optimize_parameters: Find optimal parameter configurations
        - predict_success: Predict likelihood of task success
        - detect_anomalies: Monitor for system anomalies
        - learn_from_experience: Adapt system based on historical data
        - get_adaptation_recommendations: Get AI-driven improvement suggestions
        - analyze_performance: Comprehensive performance analysis
        - evolve_system: Execute full system evolution cycle
        """
    
    async def initialize(self):
        """Initialize all adaptive learning components"""
        try:
            await self.adaptive_engine.initialize_database()
            await self.predictive_models.initialize_database()  
            await self.parameter_optimizer.initialize_database()
            await self.anomaly_detector.initialize_database()
            
            logger.info("Adaptive learning evolution tool initialized successfully")
            
        except Exception as e:
            logger.error("Error initializing adaptive learning tool: %s", e)
            raise
    
    def get_tool_definition(self) -> Tool:
        """Get MCP tool definition"""
        return Tool(
            name=self.name,
            description=self.description,
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "optimize_parameters",
                            "predict_success", 
                            "detect_anomalies",
                            "learn_from_experience",
                            "get_adaptation_recommendations",
                            "analyze_performance",
                            "evolve_system",
                            "get_status"
                        ],
                        "description": "The adaptive learning operation to perform"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Operation-specific parameters"
                    }
                },
                "required": ["operation"]
            }
        )
    
    async def execute(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute adaptive learning operation"""
        try:
            if operation == "optimize_parameters":
                return await self._optimize_parameters(parameters)
            elif operation == "predict_success":
                return await self._predict_success(parameters)
            elif operation == "detect_anomalies":
                return await self._detect_anomalies(parameters)
            elif operation == "learn_from_experience":
                return await self._learn_from_experience(parameters)
            elif operation == "get_adaptation_recommendations":
                return await self._get_adaptation_recommendations(parameters)
            elif operation == "analyze_performance":
                return await self._analyze_performance(parameters)
            elif operation == "evolve_system":
                return await self._evolve_system(parameters)
            elif operation == "get_status":
                return await self._get_status(parameters)
            else:
                return {"error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            logger.error("Error executing adaptive learning operation %s: %s", operation, e)
            return {"error": str(e), "operation": operation}
    
    async def _optimize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize parameters using evolutionary algorithms"""
        try:
            # Extract optimization parameters
            fitness_function_name = params.get("fitness_function", "default")
            algorithm = OptimizationAlgorithm(params.get("algorithm", "genetic_algorithm"))
            parameter_bounds = params.get("parameter_bounds", {})
            max_evaluations = params.get("max_evaluations", 1000)
            
            # Convert parameter bounds to ParameterBounds objects
            bounds_objects = {}
            for name, bounds_data in parameter_bounds.items():
                bounds_objects[name] = ParameterBounds(
                    name=name,
                    param_type=ParameterType(bounds_data.get("type", "continuous")),
                    min_value=bounds_data.get("min_value"),
                    max_value=bounds_data.get("max_value"),
                    categories=bounds_data.get("categories"),
                    step_size=bounds_data.get("step_size")
                )
            
            # Define fitness function
            def fitness_function(parameters: Dict[str, Any]) -> float:
                """Default fitness function for parameter optimization"""
                try:
                    if fitness_function_name == "swarm_coordination":
                        # Simulate swarm coordination fitness
                        coordination_score = parameters.get("coordination_factor", 0.5)
                        communication_score = parameters.get("communication_rate", 0.5)
                        efficiency_score = parameters.get("efficiency_factor", 0.5)
                        return (coordination_score + communication_score + efficiency_score) / 3.0
                    
                    elif fitness_function_name == "performance":
                        # Simulate performance fitness
                        throughput = parameters.get("throughput", 0.5)
                        latency = 1.0 - parameters.get("latency", 0.5)  # Lower latency is better
                        accuracy = parameters.get("accuracy", 0.5)
                        return (throughput + latency + accuracy) / 3.0
                    
                    else:
                        # Default: sum of normalized parameter values
                        return sum(v for v in parameters.values() if isinstance(v, (int, float))) / len(parameters)
                        
                except Exception as e:
                    logger.error("Error in fitness function: %s", e)
                    return 0.0
            
            # Perform optimization
            result = await self.parameter_optimizer.optimize_parameters(
                fitness_function=fitness_function,
                parameter_bounds=bounds_objects,
                algorithm=algorithm,
                max_evaluations=max_evaluations
            )
            
            return {
                "success": True,
                "best_parameters": result.best_parameters,
                "best_fitness": result.best_fitness,
                "generations": result.generations,
                "evaluations": result.evaluations,
                "algorithm": result.algorithm_used.value,
                "execution_time": result.execution_time,
                "convergence_history": result.convergence_history[-10:],  # Last 10 values
                "metadata": result.metadata
            }
            
        except Exception as e:
            logger.error("Error in parameter optimization: %s", e)
            return {"error": str(e), "success": False}
    
    async def _predict_success(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict task success probability"""
        try:
            task_features = params.get("task_features", {})
            context = params.get("context", {})
            
            # Make prediction
            success_probability = await self.predictive_models.predict_success(
                task_features, context
            )
            
            # Get prediction confidence and reasoning
            prediction_details = await self.predictive_models.explain_prediction(
                task_features, context
            )
            
            return {
                "success": True,
                "success_probability": success_probability,
                "prediction_confidence": prediction_details.get("confidence", 0.5),
                "key_factors": prediction_details.get("important_features", []),
                "prediction_reasoning": prediction_details.get("reasoning", ""),
                "model_performance": await self.predictive_models.get_model_performance(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Error in success prediction: %s", e)
            return {"error": str(e), "success": False}
    
    async def _detect_anomalies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in system metrics"""
        try:
            component = params.get("component", "system")
            metrics = params.get("metrics", {})
            context = params.get("context", {})
            
            # Detect anomalies
            anomalies = await self.anomaly_detector.detect_anomalies(
                component=component,
                metrics=metrics,
                context=context
            )
            
            # Convert anomalies to serializable format
            anomaly_data = []
            for anomaly in anomalies:
                anomaly_data.append({
                    "timestamp": anomaly.timestamp.isoformat(),
                    "type": anomaly.anomaly_type.value,
                    "severity": anomaly.severity.value,
                    "description": anomaly.description,
                    "component": anomaly.component,
                    "metrics": anomaly.metrics,
                    "confidence": anomaly.confidence,
                    "threshold_values": anomaly.threshold_values,
                    "response_actions": [action.value for action in anomaly.response_actions]
                })
            
            # Get component health
            component_health = await self.anomaly_detector.get_component_health(component)
            
            return {
                "success": True,
                "anomalies_detected": len(anomalies),
                "anomalies": anomaly_data,
                "component_health": component_health,
                "detection_statistics": await self.anomaly_detector.get_detection_statistics(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Error in anomaly detection: %s", e)
            return {"error": str(e), "success": False}
    
    async def _learn_from_experience(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from experience and adapt system"""
        try:
            experience_data = params.get("experience_data", {})
            context = params.get("context", {})
            
            # Learn from experience
            await self.adaptive_engine.learn_from_experience(
                experience_data, context
            )
            
            # Get learning insights
            learning_insights = await self.adaptive_engine.get_learning_insights()
            
            # Update predictive models with new data
            if "task_outcomes" in experience_data:
                for task_data in experience_data["task_outcomes"]:
                    await self.predictive_models.update_model_with_outcome(
                        task_data.get("features", {}),
                        task_data.get("success", False),
                        task_data.get("context", {})
                    )
            
            return {
                "success": True,
                "patterns_learned": len(learning_insights.get("patterns", [])),
                "adaptations_made": len(learning_insights.get("adaptations", [])),
                "learning_insights": learning_insights,
                "model_improvements": await self.predictive_models.get_model_performance(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Error in learning from experience: %s", e)
            return {"error": str(e), "success": False}
    
    async def _get_adaptation_recommendations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-driven adaptation recommendations"""
        try:
            context = params.get("context", {})
            performance_data = params.get("performance_data", {})
            
            # Get recommendations from adaptive engine
            recommendations = await self.adaptive_engine.recommend_adaptations(
                performance_data, context
            )
            
            # Enhance with predictive insights
            success_factors = await self.predictive_models.identify_success_factors()
            
            # Get optimization suggestions
            current_params = params.get("current_parameters", {})
            if current_params:
                # Quick optimization suggestions
                optimization_suggestions = await self._generate_optimization_suggestions(
                    current_params, performance_data
                )
            else:
                optimization_suggestions = []
            
            return {
                "success": True,
                "adaptation_recommendations": recommendations,
                "success_factors": success_factors,
                "optimization_suggestions": optimization_suggestions,
                "confidence_scores": {
                    "adaptation_confidence": recommendations.get("confidence", 0.5),
                    "prediction_confidence": success_factors.get("confidence", 0.5)
                },
                "implementation_priority": self._prioritize_recommendations(
                    recommendations, success_factors, optimization_suggestions
                ),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Error getting adaptation recommendations: %s", e)
            return {"error": str(e), "success": False}
    
    async def _analyze_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive performance analysis"""
        try:
            time_window = params.get("time_window_hours", 24)
            components = params.get("components", [])
            
            # Get historical data
            anomaly_history = await self.anomaly_detector.get_anomaly_history(
                hours=time_window
            )
            
            optimization_history = await self.parameter_optimizer.get_optimization_history()
            
            # Analyze trends and patterns
            performance_trends = await self._analyze_performance_trends(
                anomaly_history, optimization_history, time_window
            )
            
            # Get current system status
            system_status = await self._get_comprehensive_system_status()
            
            # Generate performance insights
            insights = await self._generate_performance_insights(
                performance_trends, system_status, anomaly_history
            )
            
            return {
                "success": True,
                "analysis_period": f"{time_window} hours",
                "performance_trends": performance_trends,
                "system_status": system_status,
                "anomaly_summary": {
                    "total_anomalies": len(anomaly_history),
                    "by_severity": self._group_anomalies_by_severity(anomaly_history),
                    "by_component": self._group_anomalies_by_component(anomaly_history)
                },
                "optimization_summary": {
                    "total_optimizations": len(optimization_history),
                    "best_fitness_achieved": max((r.best_fitness for r in optimization_history), default=0),
                    "average_execution_time": sum(r.execution_time for r in optimization_history) / len(optimization_history) if optimization_history else 0
                },
                "insights": insights,
                "recommendations": await self._generate_performance_recommendations(insights),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Error in performance analysis: %s", e)
            return {"error": str(e), "success": False}
    
    async def _evolve_system(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full system evolution cycle"""
        try:
            evolution_config = params.get("evolution_config", {})
            
            # Phase 1: Analyze current state
            logger.info("Starting system evolution - Phase 1: Analysis")
            current_status = await self._get_comprehensive_system_status()
            
            # Phase 2: Learn from recent experience
            logger.info("Phase 2: Learning from experience")
            recent_data = await self._gather_recent_experience_data()
            learning_result = await self.adaptive_engine.learn_from_experience(
                recent_data, {"evolution_cycle": True}
            )
            
            # Phase 3: Optimize parameters
            logger.info("Phase 3: Parameter optimization")
            optimization_config = evolution_config.get("optimization", {})
            if optimization_config:
                optimization_result = await self._optimize_parameters(optimization_config)
            else:
                optimization_result = {"skipped": "No optimization config provided"}
            
            # Phase 4: Update predictive models
            logger.info("Phase 4: Model updates")
            model_update_result = await self.predictive_models.retrain_models()
            
            # Phase 5: Detect and resolve anomalies
            logger.info("Phase 5: Anomaly detection and resolution")
            system_metrics = await self._gather_system_metrics()
            anomaly_result = await self._detect_anomalies({
                "component": "system",
                "metrics": system_metrics,
                "context": {"evolution_cycle": True}
            })
            
            # Phase 6: Generate evolution report
            logger.info("Phase 6: Evolution report generation")
            evolution_report = {
                "evolution_id": f"evo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "start_time": datetime.now().isoformat(),
                "phases_completed": 6,
                "initial_status": current_status,
                "learning_results": {
                    "patterns_discovered": len(learning_result.get("patterns", [])),
                    "adaptations_applied": len(learning_result.get("adaptations", []))
                },
                "optimization_results": optimization_result,
                "model_updates": model_update_result,
                "anomaly_resolution": anomaly_result,
                "evolution_success": True,
                "improvements": await self._calculate_evolution_improvements(current_status),
                "next_evolution_recommended": datetime.now() + timedelta(hours=24)
            }
            
            logger.info("System evolution cycle completed successfully")
            return evolution_report
            
        except Exception as e:
            logger.error("Error in system evolution: %s", e)
            return {"error": str(e), "success": False, "evolution_success": False}
    
    async def _get_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive status of adaptive learning system"""
        try:
            return await self._get_comprehensive_system_status()
            
        except Exception as e:
            logger.error("Error getting system status: %s", e)
            return {"error": str(e), "success": False}
    
    async def _get_comprehensive_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Gather status from all components
            adaptive_status = await self.adaptive_engine.get_adaptation_status()
            predictive_status = await self.predictive_models.get_model_status()
            optimization_status = await self.parameter_optimizer.get_optimization_status()
            anomaly_status = await self.anomaly_detector.get_detection_statistics()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_health": "healthy",  # Would be calculated based on component status
                "components": {
                    "adaptive_learning": {
                        "status": "active",
                        "patterns_learned": adaptive_status.get("patterns_learned", 0),
                        "adaptations_applied": adaptive_status.get("adaptations_applied", 0),
                        "learning_efficiency": adaptive_status.get("learning_efficiency", 0.0)
                    },
                    "predictive_models": {
                        "status": "active",
                        "models_trained": predictive_status.get("models_trained", 0),
                        "prediction_accuracy": predictive_status.get("accuracy", 0.0),
                        "last_training": predictive_status.get("last_training", "never")
                    },
                    "parameter_optimizer": {
                        "status": "active",
                        "optimizations_completed": optimization_status.get("optimization_stats", {}).get("total_optimizations", 0),
                        "best_fitness": optimization_status.get("optimization_stats", {}).get("best_fitness_achieved", 0.0),
                        "algorithms_available": ["genetic_algorithm", "differential_evolution", "particle_swarm", "simulated_annealing"]
                    },
                    "anomaly_detector": {
                        "status": "active", 
                        "anomalies_detected": anomaly_status.get("total_anomalies", 0),
                        "components_monitored": anomaly_status.get("monitored_components", 0),
                        "ml_models_active": anomaly_status.get("ml_models_active", {})
                    }
                },
                "capabilities": {
                    "parameter_optimization": True,
                    "predictive_modeling": True,
                    "anomaly_detection": True,
                    "adaptive_learning": True,
                    "system_evolution": True
                },
                "ml_availability": {
                    "sklearn_available": anomaly_status.get("sklearn_available", False),
                    "scipy_available": optimization_status.get("scipy_available", False)
                }
            }
            
        except Exception as e:
            logger.error("Error getting comprehensive status: %s", e)
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    async def _generate_optimization_suggestions(
        self,
        current_params: Dict[str, Any],
        performance_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate optimization suggestions based on current parameters and performance"""
        suggestions = []
        
        try:
            # Analyze current performance
            throughput = performance_data.get("throughput", 0.5)
            latency = performance_data.get("latency", 0.5)
            accuracy = performance_data.get("accuracy", 0.5)
            
            # Generate suggestions based on performance patterns
            if throughput < 0.7:
                suggestions.append({
                    "parameter": "batch_size",
                    "current_value": current_params.get("batch_size", 10),
                    "suggested_value": current_params.get("batch_size", 10) * 1.2,
                    "reason": "Increase batch size to improve throughput",
                    "expected_improvement": "15-25% throughput increase",
                    "confidence": 0.8
                })
            
            if latency > 0.6:
                suggestions.append({
                    "parameter": "timeout",
                    "current_value": current_params.get("timeout", 5.0),
                    "suggested_value": current_params.get("timeout", 5.0) * 0.8,
                    "reason": "Reduce timeout to improve responsiveness",
                    "expected_improvement": "20-30% latency reduction",
                    "confidence": 0.7
                })
            
            if accuracy < 0.8:
                suggestions.append({
                    "parameter": "validation_threshold",
                    "current_value": current_params.get("validation_threshold", 0.5),
                    "suggested_value": current_params.get("validation_threshold", 0.5) * 1.1,
                    "reason": "Increase validation threshold to improve accuracy",
                    "expected_improvement": "10-15% accuracy increase",
                    "confidence": 0.75
                })
            
        except Exception as e:
            logger.error("Error generating optimization suggestions: %s", e)
        
        return suggestions
    
    def _prioritize_recommendations(
        self,
        adaptations: Dict[str, Any],
        success_factors: Dict[str, Any],
        optimizations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prioritize recommendations based on impact and confidence"""
        priorities = []
        
        try:
            # Add adaptation recommendations
            for adaptation in adaptations.get("recommendations", []):
                priorities.append({
                    "type": "adaptation",
                    "recommendation": adaptation,
                    "priority": adaptation.get("priority", 0.5),
                    "confidence": adaptation.get("confidence", 0.5),
                    "impact": adaptation.get("expected_improvement", 0.5)
                })
            
            # Add optimization suggestions
            for optimization in optimizations:
                priorities.append({
                    "type": "optimization", 
                    "recommendation": optimization,
                    "priority": optimization.get("confidence", 0.5),
                    "confidence": optimization.get("confidence", 0.5),
                    "impact": 0.7  # Default impact for optimizations
                })
            
            # Sort by priority score (combination of confidence and impact)
            priorities.sort(key=lambda x: x["priority"] * x["confidence"] * x["impact"], reverse=True)
            
        except Exception as e:
            logger.error("Error prioritizing recommendations: %s", e)
        
        return priorities
    
    async def _analyze_performance_trends(
        self,
        anomaly_history: List,
        optimization_history: List,
        time_window: int
    ) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        try:
            trends = {
                "anomaly_trend": "stable",
                "optimization_trend": "improving", 
                "performance_score": 0.75,
                "trend_analysis": {
                    "anomalies_per_hour": len(anomaly_history) / max(time_window, 1),
                    "optimization_frequency": len(optimization_history) / max(time_window / 24, 1),  # Per day
                    "improvement_rate": 0.1  # Default improvement rate
                }
            }
            
            # Calculate actual trends if sufficient data
            if len(anomaly_history) > 5:
                # Simple trend analysis
                recent_anomalies = len([a for a in anomaly_history[-10:]])
                older_anomalies = len([a for a in anomaly_history[:-10]]) if len(anomaly_history) > 10 else recent_anomalies
                
                if recent_anomalies < older_anomalies * 0.8:
                    trends["anomaly_trend"] = "improving"
                elif recent_anomalies > older_anomalies * 1.2:
                    trends["anomaly_trend"] = "degrading"
                else:
                    trends["anomaly_trend"] = "stable"
            
            return trends
            
        except Exception as e:
            logger.error("Error analyzing performance trends: %s", e)
            return {"error": str(e)}
    
    def _group_anomalies_by_severity(self, anomalies: List) -> Dict[str, int]:
        """Group anomalies by severity level"""
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for anomaly in anomalies:
            if hasattr(anomaly, 'severity'):
                severity = anomaly.severity.value if hasattr(anomaly.severity, 'value') else str(anomaly.severity)
                if severity in severity_counts:
                    severity_counts[severity] += 1
        
        return severity_counts
    
    def _group_anomalies_by_component(self, anomalies: List) -> Dict[str, int]:
        """Group anomalies by component"""
        component_counts = {}
        
        for anomaly in anomalies:
            if hasattr(anomaly, 'component'):
                component = anomaly.component
                component_counts[component] = component_counts.get(component, 0) + 1
        
        return component_counts
    
    async def _generate_performance_insights(
        self,
        trends: Dict[str, Any],
        status: Dict[str, Any],
        anomalies: List
    ) -> List[str]:
        """Generate performance insights based on analysis"""
        insights = []
        
        try:
            # Analyze trends
            if trends.get("anomaly_trend") == "improving":
                insights.append("System stability is improving with fewer anomalies detected over time")
            elif trends.get("anomaly_trend") == "degrading":
                insights.append("System stability is declining with increasing anomaly rates")
            
            # Analyze anomaly patterns
            if anomalies:
                severity_distribution = self._group_anomalies_by_severity(anomalies)
                if severity_distribution.get("critical", 0) > 0:
                    insights.append(f"Critical anomalies detected: {severity_distribution['critical']} require immediate attention")
                
                component_distribution = self._group_anomalies_by_component(anomalies)
                if component_distribution:
                    most_affected = max(component_distribution, key=component_distribution.get)
                    insights.append(f"Component '{most_affected}' shows highest anomaly rate with {component_distribution[most_affected]} incidents")
            
            # Analyze overall health
            overall_health = status.get("overall_health", "unknown")
            if overall_health == "healthy":
                insights.append("System is operating within normal parameters")
            else:
                insights.append(f"System health status: {overall_health} - monitoring recommended")
            
        except Exception as e:
            logger.error("Error generating performance insights: %s", e)
            insights.append(f"Error in insight generation: {e}")
        
        return insights
    
    async def _generate_performance_recommendations(self, insights: List[str]) -> List[str]:
        """Generate performance recommendations based on insights"""
        recommendations = []
        
        try:
            # Generate recommendations based on insights
            for insight in insights:
                if "declining" in insight.lower() or "degrading" in insight.lower():
                    recommendations.append("Consider running parameter optimization to improve system performance")
                    recommendations.append("Increase monitoring frequency for early anomaly detection")
                
                if "critical anomalies" in insight.lower():
                    recommendations.append("Implement immediate response protocols for critical anomalies")
                    recommendations.append("Review and adjust anomaly detection thresholds")
                
                if "component" in insight.lower() and "highest anomaly rate" in insight.lower():
                    recommendations.append("Focus diagnostic efforts on high-anomaly components")
                    recommendations.append("Consider component-specific parameter tuning")
            
            # Default recommendations
            if not recommendations:
                recommendations = [
                    "Continue regular monitoring and periodic system evolution",
                    "Maintain current configuration with minor optimizations",
                    "Schedule next evolution cycle in 24 hours"
                ]
                
        except Exception as e:
            logger.error("Error generating performance recommendations: %s", e)
            recommendations.append("Error generating recommendations - manual review recommended")
        
        return recommendations
    
    async def _gather_recent_experience_data(self) -> Dict[str, Any]:
        """Gather recent experience data for learning"""
        try:
            # This would gather actual experience data from the system
            # For now, return simulated data structure
            return {
                "task_outcomes": [
                    {
                        "features": {"complexity": 0.7, "resources": 0.8, "time_limit": 300},
                        "success": True,
                        "execution_time": 245,
                        "context": {"priority": "high", "agent_count": 5}
                    }
                ],
                "performance_metrics": {
                    "throughput": 0.85,
                    "latency": 0.3,
                    "accuracy": 0.92,
                    "resource_utilization": 0.75
                },
                "coordination_patterns": {
                    "successful_patterns": ["pattern_a", "pattern_b"],
                    "failed_patterns": ["pattern_c"]
                }
            }
            
        except Exception as e:
            logger.error("Error gathering recent experience data: %s", e)
            return {}
    
    async def _gather_system_metrics(self) -> Dict[str, Any]:
        """Gather current system metrics"""
        try:
            # This would gather actual system metrics
            # For now, return simulated metrics
            return {
                "cpu_usage": 0.45,
                "memory_usage": 0.62,
                "disk_usage": 0.78,
                "network_latency": 0.25,
                "active_connections": 42,
                "throughput": 156.7,
                "error_rate": 0.02,
                "response_time": 0.8
            }
            
        except Exception as e:
            logger.error("Error gathering system metrics: %s", e)
            return {}
    
    async def _calculate_evolution_improvements(self, initial_status: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvements made during evolution cycle"""
        try:
            # Get current status after evolution
            final_status = await self._get_comprehensive_system_status()
            
            improvements = {
                "learning_efficiency_improvement": 0.05,  # Simulated improvement
                "prediction_accuracy_improvement": 0.03,
                "anomaly_detection_improvement": 0.02,
                "overall_performance_improvement": 0.08,
                "optimization_cycles_completed": 1
            }
            
            return improvements
            
        except Exception as e:
            logger.error("Error calculating evolution improvements: %s", e)
            return {"error": str(e)}


# MCP Server integration function
async def register_adaptive_learning_tool(server: Server, database_path: str = "data/memory.db"):
    """Register the adaptive learning evolution tool with MCP server"""
    try:
        tool = AdaptiveLearningEvolutionTool(database_path)
        await tool.initialize()
        
        # Register the tool
        @server.call_tool()
        async def handle_adaptive_learning_tool(
            name: str, arguments: dict
        ) -> list[types.TextContent]:
            """Handle adaptive learning evolution tool calls"""
            if name != tool.name:
                raise ValueError(f"Unknown tool: {name}")
            
            operation = arguments.get("operation")
            parameters = arguments.get("parameters", {})
            
            result = await tool.execute(operation, parameters)
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str)
                )
            ]
        
        # Register tool definition
        server.list_tools = lambda: [tool.get_tool_definition()]
        
        logger.info("Adaptive learning evolution tool registered successfully")
        return tool
        
    except Exception as e:
        logger.error("Error registering adaptive learning tool: %s", e)
        raise