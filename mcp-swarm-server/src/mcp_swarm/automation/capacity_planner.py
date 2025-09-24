"""
Capacity Planner for MCP Swarm Intelligence Server

This module provides automated capacity planning and resource scaling based on
usage trends and predictive modeling for the MCP swarm intelligence system.
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CONNECTIONS = "connections"


class ScalingDirection(Enum):
    """Direction of scaling operations"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingStrategy(Enum):
    """Scaling strategy types"""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    PROACTIVE = "proactive"
    SCHEDULED = "scheduled"


@dataclass
class UsagePattern:
    """System usage pattern"""
    resource_type: ResourceType
    current_usage: float
    peak_usage: float
    average_usage: float
    trend_direction: str
    growth_rate: float
    seasonality_factor: float
    volatility: float


@dataclass
class CapacityPrediction:
    """Capacity requirement prediction"""
    resource_type: ResourceType
    current_capacity: float
    predicted_requirement: float
    time_horizon_days: int
    confidence_level: float
    growth_factors: List[str]
    risk_level: str


@dataclass
class ScalingRecommendation:
    """Resource scaling recommendation"""
    resource_type: ResourceType
    scaling_direction: ScalingDirection
    scaling_factor: float
    target_capacity: float
    justification: str
    priority: int
    estimated_cost: float
    implementation_complexity: str


@dataclass
class CapacityPlan:
    """Complete capacity planning result"""
    usage_patterns: List[UsagePattern]
    capacity_predictions: List[CapacityPrediction]
    scaling_recommendations: List[ScalingRecommendation]
    total_growth_projection: float
    planning_horizon_days: int
    plan_confidence: float
    next_review_date: datetime


@dataclass
class ScalingResult:
    """Result of scaling operations"""
    executed_actions: List[str]
    scaling_outcomes: Dict[str, float]
    performance_impact: Dict[str, float]
    cost_impact: float
    success_rate: float
    timestamp: datetime


class UsageAnalyzer:
    """Analyzes system usage patterns and trends"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.historical_data = {}
        
    async def analyze_cpu_usage_patterns(self, historical_data: List[Dict[str, Any]]) -> UsagePattern:
        """Analyze CPU usage patterns"""
        cpu_values = [entry.get("cpu_usage", 0) for entry in historical_data]
        
        if not cpu_values:
            return UsagePattern(
                resource_type=ResourceType.CPU,
                current_usage=0.0,
                peak_usage=0.0,
                average_usage=0.0,
                trend_direction="stable",
                growth_rate=0.0,
                seasonality_factor=1.0,
                volatility=0.0
            )
        
        current_usage = cpu_values[-1] if cpu_values else 0.0
        peak_usage = max(cpu_values)
        average_usage = statistics.mean(cpu_values)
        
        # Calculate trend
        trend_direction, growth_rate = self._calculate_trend(cpu_values)
        
        # Calculate seasonality
        seasonality_factor = self._calculate_seasonality(cpu_values)
        
        # Calculate volatility
        volatility = self._calculate_volatility(cpu_values)
        
        return UsagePattern(
            resource_type=ResourceType.CPU,
            current_usage=current_usage,
            peak_usage=peak_usage,
            average_usage=average_usage,
            trend_direction=trend_direction,
            growth_rate=growth_rate,
            seasonality_factor=seasonality_factor,
            volatility=volatility
        )
    
    async def analyze_memory_usage_patterns(self, historical_data: List[Dict[str, Any]]) -> UsagePattern:
        """Analyze memory usage patterns"""
        memory_values = [entry.get("memory_usage", 0) for entry in historical_data]
        
        if not memory_values:
            return UsagePattern(
                resource_type=ResourceType.MEMORY,
                current_usage=0.0,
                peak_usage=0.0,
                average_usage=0.0,
                trend_direction="stable",
                growth_rate=0.0,
                seasonality_factor=1.0,
                volatility=0.0
            )
        
        current_usage = memory_values[-1] if memory_values else 0.0
        peak_usage = max(memory_values)
        average_usage = statistics.mean(memory_values)
        
        trend_direction, growth_rate = self._calculate_trend(memory_values)
        seasonality_factor = self._calculate_seasonality(memory_values)
        volatility = self._calculate_volatility(memory_values)
        
        return UsagePattern(
            resource_type=ResourceType.MEMORY,
            current_usage=current_usage,
            peak_usage=peak_usage,
            average_usage=average_usage,
            trend_direction=trend_direction,
            growth_rate=growth_rate,
            seasonality_factor=seasonality_factor,
            volatility=volatility
        )
    
    async def analyze_storage_usage_patterns(self, historical_data: List[Dict[str, Any]]) -> UsagePattern:
        """Analyze storage usage patterns"""
        storage_values = [entry.get("disk_usage", 0) for entry in historical_data]
        
        if not storage_values:
            return UsagePattern(
                resource_type=ResourceType.STORAGE,
                current_usage=0.0,
                peak_usage=0.0,
                average_usage=0.0,
                trend_direction="stable",
                growth_rate=0.0,
                seasonality_factor=1.0,
                volatility=0.0
            )
        
        current_usage = storage_values[-1] if storage_values else 0.0
        peak_usage = max(storage_values)
        average_usage = statistics.mean(storage_values)
        
        trend_direction, growth_rate = self._calculate_trend(storage_values)
        seasonality_factor = self._calculate_seasonality(storage_values)
        volatility = self._calculate_volatility(storage_values)
        
        return UsagePattern(
            resource_type=ResourceType.STORAGE,
            current_usage=current_usage,
            peak_usage=peak_usage,
            average_usage=average_usage,
            trend_direction=trend_direction,
            growth_rate=growth_rate,
            seasonality_factor=seasonality_factor,
            volatility=volatility
        )
    
    async def analyze_network_usage_patterns(self, historical_data: List[Dict[str, Any]]) -> UsagePattern:
        """Analyze network usage patterns"""
        network_values = [entry.get("network_io", 0) for entry in historical_data]
        
        if not network_values:
            return UsagePattern(
                resource_type=ResourceType.NETWORK,
                current_usage=0.0,
                peak_usage=0.0,
                average_usage=0.0,
                trend_direction="stable",
                growth_rate=0.0,
                seasonality_factor=1.0,
                volatility=0.0
            )
        
        current_usage = network_values[-1] if network_values else 0.0
        peak_usage = max(network_values)
        average_usage = statistics.mean(network_values)
        
        trend_direction, growth_rate = self._calculate_trend(network_values)
        seasonality_factor = self._calculate_seasonality(network_values)
        volatility = self._calculate_volatility(network_values)
        
        return UsagePattern(
            resource_type=ResourceType.NETWORK,
            current_usage=current_usage,
            peak_usage=peak_usage,
            average_usage=average_usage,
            trend_direction=trend_direction,
            growth_rate=growth_rate,
            seasonality_factor=seasonality_factor,
            volatility=volatility
        )
    
    def _calculate_trend(self, values: List[float]) -> tuple[str, float]:
        """Calculate trend direction and growth rate"""
        if len(values) < 2:
            return "stable", 0.0
        
        # Simple linear trend calculation
        n = len(values)
        x_values = list(range(n))
        
        # Calculate linear regression slope
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return "stable", 0.0
        
        slope = numerator / denominator
        
        # Determine trend direction
        if slope > 0.5:
            trend = "increasing"
        elif slope < -0.5:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Convert slope to growth rate percentage
        growth_rate = (slope / y_mean) * 100 if y_mean != 0 else 0
        
        return trend, growth_rate
    
    def _calculate_seasonality(self, values: List[float]) -> float:
        """Calculate seasonality factor"""
        if len(values) < 7:  # Need at least a week of data
            return 1.0
        
        # Simple seasonality detection based on weekly patterns
        try:
            # Group by day of week (assuming daily data points)
            weekly_averages = []
            for i in range(7):
                day_values = [values[j] for j in range(i, len(values), 7)]
                if day_values:
                    weekly_averages.append(statistics.mean(day_values))
            
            if not weekly_averages:
                return 1.0
            
            overall_mean = statistics.mean(weekly_averages)
            if overall_mean == 0:
                return 1.0
            
            # Calculate variation coefficient
            variation = statistics.stdev(weekly_averages) / overall_mean
            seasonality_factor = 1.0 + variation
            
            return min(seasonality_factor, 2.0)  # Cap at 2.0
            
        except (statistics.StatisticsError, ZeroDivisionError):
            return 1.0
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility of usage patterns"""
        if len(values) < 2:
            return 0.0
        
        try:
            mean_value = statistics.mean(values)
            if mean_value == 0:
                return 0.0
            
            # Calculate coefficient of variation
            std_dev = statistics.stdev(values)
            volatility = std_dev / mean_value
            
            return min(volatility, 2.0)  # Cap at 2.0
            
        except (statistics.StatisticsError, ZeroDivisionError):
            return 0.0


class ScalingEngine:
    """Executes scaling decisions and resource adjustments"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaling_history = []
        
    async def execute_cpu_scaling(self, recommendation: ScalingRecommendation) -> Dict[str, Any]:
        """Execute CPU scaling recommendations"""
        try:
            # Simulate CPU scaling operation
            current_cores = 4  # Placeholder
            
            if recommendation.scaling_direction == ScalingDirection.UP:
                new_cores = int(current_cores * recommendation.scaling_factor)
                action = f"Scale CPU from {current_cores} to {new_cores} cores"
            elif recommendation.scaling_direction == ScalingDirection.DOWN:
                new_cores = max(1, int(current_cores / recommendation.scaling_factor))
                action = f"Scale CPU down from {current_cores} to {new_cores} cores"
            else:
                new_cores = current_cores
                action = "Maintain current CPU allocation"
            
            # Record scaling action
            scaling_result = {
                "resource_type": "cpu",
                "action": action,
                "previous_value": current_cores,
                "new_value": new_cores,
                "success": True,
                "timestamp": datetime.utcnow()
            }
            
            return scaling_result
            
        except Exception as e:
            self.logger.error("Error executing CPU scaling: %s", str(e))
            return {
                "resource_type": "cpu",
                "action": "CPU scaling failed",
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow()
            }
    
    async def execute_memory_scaling(self, recommendation: ScalingRecommendation) -> Dict[str, Any]:
        """Execute memory scaling recommendations"""
        try:
            # Simulate memory scaling operation
            current_memory_gb = 8  # Placeholder
            
            if recommendation.scaling_direction == ScalingDirection.UP:
                new_memory_gb = int(current_memory_gb * recommendation.scaling_factor)
                action = f"Scale memory from {current_memory_gb}GB to {new_memory_gb}GB"
            elif recommendation.scaling_direction == ScalingDirection.DOWN:
                new_memory_gb = max(2, int(current_memory_gb / recommendation.scaling_factor))
                action = f"Scale memory down from {current_memory_gb}GB to {new_memory_gb}GB"
            else:
                new_memory_gb = current_memory_gb
                action = "Maintain current memory allocation"
            
            scaling_result = {
                "resource_type": "memory",
                "action": action,
                "previous_value": current_memory_gb,
                "new_value": new_memory_gb,
                "success": True,
                "timestamp": datetime.utcnow()
            }
            
            return scaling_result
            
        except Exception as e:
            self.logger.error("Error executing memory scaling: %s", str(e))
            return {
                "resource_type": "memory",
                "action": "Memory scaling failed",
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow()
            }
    
    async def execute_storage_scaling(self, recommendation: ScalingRecommendation) -> Dict[str, Any]:
        """Execute storage scaling recommendations"""
        try:
            # Simulate storage scaling operation
            current_storage_gb = 100  # Placeholder
            
            if recommendation.scaling_direction == ScalingDirection.UP:
                new_storage_gb = int(current_storage_gb * recommendation.scaling_factor)
                action = f"Scale storage from {current_storage_gb}GB to {new_storage_gb}GB"
            elif recommendation.scaling_direction == ScalingDirection.DOWN:
                new_storage_gb = max(50, int(current_storage_gb / recommendation.scaling_factor))
                action = f"Scale storage down from {current_storage_gb}GB to {new_storage_gb}GB"
            else:
                new_storage_gb = current_storage_gb
                action = "Maintain current storage allocation"
            
            scaling_result = {
                "resource_type": "storage",
                "action": action,
                "previous_value": current_storage_gb,
                "new_value": new_storage_gb,
                "success": True,
                "timestamp": datetime.utcnow()
            }
            
            return scaling_result
            
        except Exception as e:
            self.logger.error("Error executing storage scaling: %s", str(e))
            return {
                "resource_type": "storage",
                "action": "Storage scaling failed",
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow()
            }
    
    async def execute_connection_scaling(self, recommendation: ScalingRecommendation) -> Dict[str, Any]:
        """Execute connection pool scaling recommendations"""
        try:
            # Simulate connection pool scaling
            current_connections = 50  # Placeholder
            
            if recommendation.scaling_direction == ScalingDirection.UP:
                new_connections = int(current_connections * recommendation.scaling_factor)
                action = f"Scale connection pool from {current_connections} to {new_connections}"
            elif recommendation.scaling_direction == ScalingDirection.DOWN:
                new_connections = max(10, int(current_connections / recommendation.scaling_factor))
                action = f"Scale connection pool down from {current_connections} to {new_connections}"
            else:
                new_connections = current_connections
                action = "Maintain current connection pool size"
            
            scaling_result = {
                "resource_type": "connections",
                "action": action,
                "previous_value": current_connections,
                "new_value": new_connections,
                "success": True,
                "timestamp": datetime.utcnow()
            }
            
            return scaling_result
            
        except Exception as e:
            self.logger.error("Error executing connection scaling: %s", str(e))
            return {
                "resource_type": "connections",
                "action": "Connection scaling failed",
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow()
            }
    
    async def validate_scaling_operation(self, scaling_result: Dict[str, Any]) -> bool:
        """Validate that scaling operation was successful"""
        # In a real implementation, this would check actual system resources
        return scaling_result.get("success", False)
    
    async def rollback_scaling_operation(self, scaling_result: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback a scaling operation if needed"""
        try:
            rollback_result = {
                "resource_type": scaling_result["resource_type"],
                "action": f"Rollback: {scaling_result['action']}",
                "previous_value": scaling_result["new_value"],
                "new_value": scaling_result["previous_value"],
                "success": True,
                "timestamp": datetime.utcnow()
            }
            
            self.logger.info("Rolled back scaling operation for %s", scaling_result["resource_type"])
            return rollback_result
            
        except Exception as e:
            self.logger.error("Error rolling back scaling operation: %s", str(e))
            return {
                "resource_type": scaling_result.get("resource_type", "unknown"),
                "action": "Rollback failed",
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow()
            }


class CapacityPlanner:
    """Automated capacity planning and resource scaling"""
    
    def __init__(self):
        self.usage_analyzer = UsageAnalyzer()
        self.scaling_engine = ScalingEngine()
        self.logger = logging.getLogger(__name__)
        
    async def plan_capacity_requirements(self) -> CapacityPlan:
        """Plan future capacity requirements based on trends"""
        try:
            # Get historical usage data (placeholder implementation)
            historical_data = await self._get_historical_usage_data()
            
            # Analyze usage patterns for different resources
            cpu_pattern = await self.usage_analyzer.analyze_cpu_usage_patterns(historical_data)
            memory_pattern = await self.usage_analyzer.analyze_memory_usage_patterns(historical_data)
            storage_pattern = await self.usage_analyzer.analyze_storage_usage_patterns(historical_data)
            network_pattern = await self.usage_analyzer.analyze_network_usage_patterns(historical_data)
            
            usage_patterns = [cpu_pattern, memory_pattern, storage_pattern, network_pattern]
            
            # Generate capacity predictions
            capacity_predictions = []
            for pattern in usage_patterns:
                prediction = await self._generate_capacity_prediction(pattern)
                capacity_predictions.append(prediction)
            
            # Generate scaling recommendations
            scaling_recommendations = []
            for prediction in capacity_predictions:
                recommendation = await self._generate_scaling_recommendation(prediction)
                if recommendation:
                    scaling_recommendations.append(recommendation)
            
            # Calculate overall growth projection
            total_growth = self._calculate_total_growth_projection(capacity_predictions)
            
            # Calculate plan confidence
            plan_confidence = self._calculate_plan_confidence(usage_patterns, capacity_predictions)
            
            # Set planning horizon and next review date
            planning_horizon_days = 30
            next_review_date = datetime.utcnow() + timedelta(days=7)
            
            return CapacityPlan(
                usage_patterns=usage_patterns,
                capacity_predictions=capacity_predictions,
                scaling_recommendations=scaling_recommendations,
                total_growth_projection=total_growth,
                planning_horizon_days=planning_horizon_days,
                plan_confidence=plan_confidence,
                next_review_date=next_review_date
            )
            
        except Exception as e:
            self.logger.error("Error planning capacity requirements: %s", str(e))
            # Return empty plan on error
            return CapacityPlan(
                usage_patterns=[],
                capacity_predictions=[],
                scaling_recommendations=[],
                total_growth_projection=0.0,
                planning_horizon_days=30,
                plan_confidence=0.0,
                next_review_date=datetime.utcnow() + timedelta(days=7)
            )
    
    async def execute_scaling_decisions(self, capacity_plan: CapacityPlan) -> ScalingResult:
        """Execute automated scaling decisions"""
        try:
            executed_actions = []
            scaling_outcomes = {}
            performance_impact = {}
            total_cost = 0.0
            successful_operations = 0
            
            # Execute scaling recommendations
            for recommendation in capacity_plan.scaling_recommendations:
                if recommendation.priority >= 3:  # Execute high priority recommendations
                    scaling_result = await self._execute_scaling_recommendation(recommendation)
                    
                    if scaling_result["success"]:
                        successful_operations += 1
                        executed_actions.append(scaling_result["action"])
                        scaling_outcomes[recommendation.resource_type.value] = recommendation.scaling_factor
                        performance_impact[recommendation.resource_type.value] = self._estimate_performance_impact(recommendation)
                        total_cost += recommendation.estimated_cost
                    else:
                        executed_actions.append(f"Failed: {scaling_result.get('action', 'Unknown action')}")
            
            # Calculate success rate
            total_operations = len([r for r in capacity_plan.scaling_recommendations if r.priority >= 3])
            success_rate = successful_operations / total_operations if total_operations > 0 else 1.0
            
            return ScalingResult(
                executed_actions=executed_actions,
                scaling_outcomes=scaling_outcomes,
                performance_impact=performance_impact,
                cost_impact=total_cost,
                success_rate=success_rate,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error("Error executing scaling decisions: %s", str(e))
            return ScalingResult(
                executed_actions=[],
                scaling_outcomes={},
                performance_impact={},
                cost_impact=0.0,
                success_rate=0.0,
                timestamp=datetime.utcnow()
            )
    
    async def _get_historical_usage_data(self) -> List[Dict[str, Any]]:
        """Get historical usage data for analysis"""
        # Placeholder implementation - would integrate with monitoring system
        return [
            {
                "timestamp": datetime.utcnow() - timedelta(days=i),
                "cpu_usage": 50 + (i * 2) + (i % 7 * 5),  # Simulated growth with weekly pattern
                "memory_usage": 60 + (i * 1.5) + (i % 7 * 3),
                "disk_usage": 70 + (i * 0.5),
                "network_io": 40 + (i * 3) + (i % 7 * 8)
            }
            for i in range(30, 0, -1)  # Last 30 days
        ]
    
    async def _generate_capacity_prediction(self, pattern: UsagePattern) -> CapacityPrediction:
        """Generate capacity prediction from usage pattern"""
        # Project future requirements based on trend
        days_ahead = 30
        projected_growth = pattern.growth_rate * (days_ahead / 30)  # Monthly growth rate
        
        # Apply seasonality factor
        seasonal_adjustment = pattern.seasonality_factor
        
        # Calculate predicted requirement
        base_requirement = pattern.current_usage
        growth_factor = 1 + (projected_growth / 100)
        predicted_requirement = base_requirement * growth_factor * seasonal_adjustment
        
        # Add buffer for volatility
        volatility_buffer = 1 + (pattern.volatility * 0.2)
        predicted_requirement *= volatility_buffer
        
        # Determine confidence level based on data quality
        confidence_level = self._calculate_prediction_confidence(pattern)
        
        # Assess risk level
        risk_level = self._assess_risk_level(pattern, predicted_requirement)
        
        # Identify growth factors
        growth_factors = self._identify_growth_factors(pattern)
        
        return CapacityPrediction(
            resource_type=pattern.resource_type,
            current_capacity=pattern.current_usage * 1.2,  # Assume 20% headroom
            predicted_requirement=predicted_requirement,
            time_horizon_days=days_ahead,
            confidence_level=confidence_level,
            growth_factors=growth_factors,
            risk_level=risk_level
        )
    
    async def _generate_scaling_recommendation(self, prediction: CapacityPrediction) -> Optional[ScalingRecommendation]:
        """Generate scaling recommendation from capacity prediction"""
        if prediction.predicted_requirement <= prediction.current_capacity * 0.8:
            # Scale down
            scaling_direction = ScalingDirection.DOWN
            scaling_factor = prediction.current_capacity / prediction.predicted_requirement
            target_capacity = prediction.predicted_requirement * 1.1  # 10% buffer
            justification = "Predicted usage is below current capacity"
            priority = 2
        elif prediction.predicted_requirement > prediction.current_capacity * 0.9:
            # Scale up
            scaling_direction = ScalingDirection.UP
            scaling_factor = prediction.predicted_requirement / prediction.current_capacity
            target_capacity = prediction.predicted_requirement * 1.2  # 20% buffer
            justification = "Predicted usage exceeds current capacity"
            priority = 4 if prediction.risk_level in ["high", "critical"] else 3
        else:
            # No scaling needed
            return None
        
        # Estimate cost and complexity
        estimated_cost = self._estimate_scaling_cost(prediction.resource_type, scaling_factor)
        implementation_complexity = self._assess_implementation_complexity(prediction.resource_type, scaling_factor)
        
        return ScalingRecommendation(
            resource_type=prediction.resource_type,
            scaling_direction=scaling_direction,
            scaling_factor=scaling_factor,
            target_capacity=target_capacity,
            justification=justification,
            priority=priority,
            estimated_cost=estimated_cost,
            implementation_complexity=implementation_complexity
        )
    
    async def _execute_scaling_recommendation(self, recommendation: ScalingRecommendation) -> Dict[str, Any]:
        """Execute a specific scaling recommendation"""
        if recommendation.resource_type == ResourceType.CPU:
            return await self.scaling_engine.execute_cpu_scaling(recommendation)
        elif recommendation.resource_type == ResourceType.MEMORY:
            return await self.scaling_engine.execute_memory_scaling(recommendation)
        elif recommendation.resource_type == ResourceType.STORAGE:
            return await self.scaling_engine.execute_storage_scaling(recommendation)
        elif recommendation.resource_type == ResourceType.CONNECTIONS:
            return await self.scaling_engine.execute_connection_scaling(recommendation)
        else:
            return {
                "resource_type": recommendation.resource_type.value,
                "action": "Unsupported resource type",
                "success": False,
                "timestamp": datetime.utcnow()
            }
    
    def _calculate_total_growth_projection(self, predictions: List[CapacityPrediction]) -> float:
        """Calculate overall growth projection"""
        if not predictions:
            return 0.0
        
        growth_rates = []
        for prediction in predictions:
            current = prediction.current_capacity
            predicted = prediction.predicted_requirement
            if current > 0:
                growth_rate = ((predicted - current) / current) * 100
                growth_rates.append(growth_rate)
        
        return statistics.mean(growth_rates) if growth_rates else 0.0
    
    def _calculate_plan_confidence(self, patterns: List[UsagePattern], 
                                 predictions: List[CapacityPrediction]) -> float:
        """Calculate overall plan confidence"""
        confidence_factors = []
        
        # Pattern confidence based on volatility
        for pattern in patterns:
            pattern_confidence = max(0.0, 1.0 - pattern.volatility)
            confidence_factors.append(pattern_confidence)
        
        # Prediction confidence
        for prediction in predictions:
            confidence_factors.append(prediction.confidence_level)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.0
    
    def _calculate_prediction_confidence(self, pattern: UsagePattern) -> float:
        """Calculate confidence level for a prediction"""
        confidence = 0.8  # Base confidence
        
        # Reduce confidence for high volatility
        confidence -= pattern.volatility * 0.3
        
        # Reduce confidence for unstable trends
        if pattern.trend_direction == "stable":
            confidence += 0.1
        
        # Adjust for growth rate extremes
        if abs(pattern.growth_rate) > 50:  # Very high growth rate
            confidence -= 0.2
        
        return max(0.1, min(confidence, 1.0))
    
    def _assess_risk_level(self, pattern: UsagePattern, predicted_requirement: float) -> str:
        """Assess risk level for capacity prediction"""
        current = pattern.current_usage
        
        if predicted_requirement > current * 2:
            return "critical"
        elif predicted_requirement > current * 1.5:
            return "high"
        elif predicted_requirement > current * 1.2:
            return "medium"
        else:
            return "low"
    
    def _identify_growth_factors(self, pattern: UsagePattern) -> List[str]:
        """Identify factors contributing to growth"""
        factors = []
        
        if pattern.growth_rate > 10:
            factors.append("high_growth_trend")
        
        if pattern.seasonality_factor > 1.2:
            factors.append("seasonal_peaks")
        
        if pattern.volatility > 0.5:
            factors.append("usage_volatility")
        
        if pattern.trend_direction == "increasing":
            factors.append("increasing_demand")
        
        return factors if factors else ["stable_usage"]
    
    def _estimate_scaling_cost(self, resource_type: ResourceType, scaling_factor: float) -> float:
        """Estimate cost of scaling operation"""
        # Simplified cost estimation
        base_costs = {
            ResourceType.CPU: 100.0,
            ResourceType.MEMORY: 80.0,
            ResourceType.STORAGE: 50.0,
            ResourceType.NETWORK: 60.0,
            ResourceType.DATABASE: 120.0,
            ResourceType.CONNECTIONS: 20.0
        }
        
        base_cost = base_costs.get(resource_type, 50.0)
        scaling_multiplier = max(0.5, min(scaling_factor, 3.0))  # Cap scaling factor
        
        return base_cost * scaling_multiplier
    
    def _assess_implementation_complexity(self, _resource_type: ResourceType, scaling_factor: float) -> str:
        """Assess implementation complexity"""
        if scaling_factor > 2.0:
            return "high"
        elif scaling_factor > 1.5:
            return "medium"
        else:
            return "low"
    
    def _estimate_performance_impact(self, recommendation: ScalingRecommendation) -> float:
        """Estimate performance impact of scaling"""
        if recommendation.scaling_direction == ScalingDirection.UP:
            return recommendation.scaling_factor * 20  # 20% improvement per scaling factor
        elif recommendation.scaling_direction == ScalingDirection.DOWN:
            return -(100 / recommendation.scaling_factor) * 10  # 10% degradation per scale down factor
        else:
            return 0.0