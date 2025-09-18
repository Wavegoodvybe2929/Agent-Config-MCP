"""
Real-time Strategy Selector for Dynamic Coordination

This module provides intelligent strategy selection algorithms that analyze
task context, team composition, and performance history to recommend optimal
coordination strategies for multi-agent workflows.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime

from .coordination_strategies import (
    CoordinationPattern, 
    BaseCoordinationStrategy, 
    CoordinationStrategyLibrary,
    Task, 
    Agent
)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ContextualFactors:
    """Factors that influence strategy selection."""
    task_complexity_score: float
    team_size: int
    dependency_density: float
    time_pressure: float
    resource_constraints: float
    collaboration_history: float
    risk_tolerance: float


@dataclass
class StrategyRecommendation:
    """Complete strategy recommendation with reasoning."""
    primary_strategy: BaseCoordinationStrategy
    confidence_score: float
    reasoning: List[str]
    fallback_strategies: List[BaseCoordinationStrategy]
    expected_performance: Dict[str, float]
    risk_factors: List[str]


class StrategySelector:
    """Intelligent strategy selector using multi-criteria analysis."""
    
    def __init__(self, coordination_engine=None):
        self.coordination_engine = coordination_engine
        self.strategy_library = CoordinationStrategyLibrary(coordination_engine)
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.selection_weights = self._initialize_selection_weights()
    
    def _initialize_selection_weights(self) -> Dict[str, float]:
        """Initialize weights for different selection criteria."""
        return {
            "task_complexity": 0.25,
            "team_dynamics": 0.20,
            "time_constraints": 0.15,
            "historical_performance": 0.20,
            "resource_availability": 0.10,
            "risk_assessment": 0.10
        }
    
    async def select_optimal_strategy(
        self, 
        tasks: List[Task], 
        agents: List[Agent],
        constraints: Optional[Dict[str, Any]] = None
    ) -> StrategyRecommendation:
        """Select the best coordination strategy based on comprehensive analysis."""
        
        if not tasks or not agents:
            logger.warning("Cannot select strategy: empty tasks or agents list")
            return self._create_fallback_recommendation()
        
        try:
            # Analyze contextual factors
            context = self._analyze_context(tasks, agents, constraints or {})
            
            # Evaluate all available strategies
            strategy_scores = await self._evaluate_all_strategies(tasks, agents, context)
            
            # Select best strategy
            best_strategy, confidence = self._select_best_strategy(strategy_scores)
            
            if not best_strategy:
                return self._create_fallback_recommendation()
            
            # Generate comprehensive recommendation
            recommendation = await self._generate_recommendation(
                best_strategy, confidence, tasks, agents, context, strategy_scores
            )
            
            # Record selection for learning
            self._record_selection(recommendation, context)
            
            return recommendation
            
        except RuntimeError as e:
            logger.error("Strategy selection failed: %s", str(e))
            return self._create_fallback_recommendation()
    
    def _analyze_context(self, tasks: List[Task], agents: List[Agent], constraints: Dict[str, Any]) -> ContextualFactors:
        """Analyze the coordination context to extract relevant factors."""
        
        # Task complexity analysis
        task_complexity = self._analyze_task_complexity(tasks)
        
        # Team dynamics analysis
        team_size = len(agents)
        
        # Dependency analysis
        dependency_density = self._calculate_dependency_density(tasks)
        
        # Time pressure analysis
        time_pressure = self._analyze_time_pressure(tasks, constraints)
        
        # Resource constraints analysis
        resource_constraints = self._analyze_resource_constraints(agents, constraints)
        
        # Collaboration history analysis
        collaboration_history = self._analyze_collaboration_history(agents)
        
        # Risk tolerance from constraints
        risk_tolerance = constraints.get('risk_tolerance', 0.5)
        
        return ContextualFactors(
            task_complexity_score=task_complexity,
            team_size=team_size,
            dependency_density=dependency_density,
            time_pressure=time_pressure,
            resource_constraints=resource_constraints,
            collaboration_history=collaboration_history,
            risk_tolerance=risk_tolerance
        )
    
    def _analyze_task_complexity(self, tasks: List[Task]) -> float:
        """Calculate overall task complexity score."""
        if not tasks:
            return 0.0
        
        # Base complexity from task attributes
        complexities = [task.complexity for task in tasks]
        avg_complexity = np.mean(complexities)
        complexity_variance = np.var(complexities)
        
        # Additional complexity from requirements diversity
        all_capabilities = set()
        for task in tasks:
            all_capabilities.update(task.required_capabilities)
        
        capability_diversity = len(all_capabilities) / max(len(tasks), 1)
        
        # Priority spread analysis
        priorities = [task.priority for task in tasks]
        priority_spread = (max(priorities) - min(priorities)) / max(priorities[0] if priorities else 1, 1) if priorities else 0
        
        # Deadline pressure
        current_time = datetime.now()
        deadline_pressures = []
        for task in tasks:
            if task.deadline:
                time_remaining = (task.deadline - current_time).total_seconds() / 3600  # hours
                pressure = max(0, 1 - time_remaining / (task.estimated_duration / 60))  # normalize
                deadline_pressures.append(pressure)
        
        avg_deadline_pressure = np.mean(deadline_pressures) if deadline_pressures else 0.0
        
        # Combine factors
        complexity_score = (
            avg_complexity * 0.4 +
            complexity_variance * 0.2 +
            capability_diversity * 0.2 +
            priority_spread * 0.1 +
            avg_deadline_pressure * 0.1
        )
        
        return float(min(complexity_score, 1.0))
    
    def _calculate_dependency_density(self, tasks: List[Task]) -> float:
        """Calculate how interconnected the tasks are."""
        if not tasks:
            return 0.0
        
        total_dependencies = sum(len(task.dependencies) for task in tasks)
        max_possible_dependencies = len(tasks) * (len(tasks) - 1)
        
        return total_dependencies / max(max_possible_dependencies, 1)
    
    def _analyze_time_pressure(self, tasks: List[Task], constraints: Dict[str, Any]) -> float:
        """Analyze time pressure based on deadlines and constraints."""
        current_time = datetime.now()
        
        # Timeline constraints
        timeline_constraints = constraints.get('timeline_constraints', {})
        overall_deadline = timeline_constraints.get('deadline')
        
        pressure_factors = []
        
        # Individual task deadline pressure
        for task in tasks:
            if task.deadline:
                time_remaining = (task.deadline - current_time).total_seconds() / 3600
                estimated_hours = task.estimated_duration / 60
                pressure = max(0, 1 - time_remaining / max(estimated_hours, 1))
                pressure_factors.append(pressure)
        
        # Overall project deadline pressure
        if overall_deadline:
            if isinstance(overall_deadline, str):
                overall_deadline = datetime.fromisoformat(overall_deadline)
            
            total_estimated_time = sum(task.estimated_duration for task in tasks) / 60  # hours
            time_remaining = (overall_deadline - current_time).total_seconds() / 3600
            overall_pressure = max(0, 1 - time_remaining / max(total_estimated_time, 1))
            pressure_factors.append(overall_pressure)
        
        # Urgency from constraints
        urgency = constraints.get('urgency', 'medium')
        urgency_score = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}.get(urgency, 0.5)
        pressure_factors.append(urgency_score)
        
        return float(np.mean(pressure_factors)) if pressure_factors else 0.5
    
    def _analyze_resource_constraints(self, agents: List[Agent], constraints: Dict[str, Any]) -> float:
        """Analyze resource availability and constraints."""
        if not agents:
            return 1.0  # Maximum constraint
        
        # Agent availability analysis
        available_agents = sum(1 for agent in agents if agent.availability)
        availability_ratio = available_agents / len(agents)
        
        # Agent load analysis
        current_loads = [agent.current_load for agent in agents if agent.availability]
        avg_load = np.mean(current_loads) if current_loads else 0.0
        
        # Capability coverage analysis
        all_capabilities = set()
        for agent in agents:
            all_capabilities.update(agent.capabilities)
        
        capability_coverage = len(all_capabilities) / max(len(agents), 1)
        
        # Resource constraints from input
        resource_limits = constraints.get('resource_constraints', {})
        max_agents = resource_limits.get('max_agents', len(agents))
        max_parallel_tasks = resource_limits.get('max_parallel_tasks', len(agents))
        
        agent_constraint = len(agents) / max(max_agents, 1)
        parallel_constraint = len(agents) / max(max_parallel_tasks, 1)
        
        # Combine factors (higher value = more constrained)
        constraint_score = (
            (1 - availability_ratio) * 0.3 +
            avg_load * 0.3 +
            (1 - capability_coverage) * 0.2 +
            min(agent_constraint, 1.0) * 0.1 +
            min(parallel_constraint, 1.0) * 0.1
        )
        
        return min(constraint_score, 1.0)
    
    def _analyze_collaboration_history(self, agents: List[Agent]) -> float:
        """Analyze how well the team has collaborated in the past."""
        if not agents:
            return 0.5
        
        # Average performance scores
        performance_scores = []
        for agent in agents:
            success_rate = agent.performance_history.get('success_rate', 0.5)
            collaboration_score = agent.performance_history.get('collaboration_score', 0.5)
            performance_scores.append((success_rate + collaboration_score) / 2)
        
        avg_performance = np.mean(performance_scores)
        
        # Team diversity (can be positive for collaboration)
        capabilities = set()
        for agent in agents:
            capabilities.update(agent.capabilities)
        
        diversity_score = len(capabilities) / max(len(agents) * 2, 1)  # Normalize
        
        # Combine factors
        collaboration_score = avg_performance * 0.7 + diversity_score * 0.3
        
        return float(min(collaboration_score, 1.0))
    
    async def _evaluate_all_strategies(
        self, 
        tasks: List[Task], 
        agents: List[Agent], 
        context: ContextualFactors
    ) -> Dict[BaseCoordinationStrategy, float]:
        """Evaluate all available strategies and assign scores."""
        
        strategy_scores = {}
        
        for strategy in self.strategy_library.get_all_strategies():
            try:
                score = await self._evaluate_strategy(strategy, tasks, agents, context)
                strategy_scores[strategy] = score
            except RuntimeError as e:
                logger.warning("Failed to evaluate strategy %s: %s", strategy.name, str(e))
                strategy_scores[strategy] = 0.0
        
        return strategy_scores
    
    async def _evaluate_strategy(
        self, 
        strategy: BaseCoordinationStrategy, 
        tasks: List[Task], 
        agents: List[Agent], 
        context: ContextualFactors
    ) -> float:
        """Evaluate a single strategy based on multiple criteria."""
        
        scores = {}
        
        # Task complexity fit
        complexity_fit = self._evaluate_complexity_fit(strategy, context.task_complexity_score)
        scores['complexity'] = complexity_fit * self.selection_weights['task_complexity']
        
        # Team dynamics fit
        team_fit = self._evaluate_team_fit(strategy, context.team_size, context.collaboration_history)
        scores['team'] = team_fit * self.selection_weights['team_dynamics']
        
        # Time constraints fit
        time_fit = self._evaluate_time_fit(strategy, context.time_pressure, tasks, agents)
        scores['time'] = time_fit * self.selection_weights['time_constraints']
        
        # Historical performance
        historical_fit = self._evaluate_historical_performance(strategy, tasks, agents)
        scores['historical'] = historical_fit * self.selection_weights['historical_performance']
        
        # Resource availability fit
        resource_fit = self._evaluate_resource_fit(strategy, context.resource_constraints)
        scores['resources'] = resource_fit * self.selection_weights['resource_availability']
        
        # Risk assessment fit
        risk_fit = self._evaluate_risk_fit(strategy, context.risk_tolerance)
        scores['risk'] = risk_fit * self.selection_weights['risk_assessment']
        
        # Base suitability from strategy itself
        base_suitability = strategy.assess_suitability(tasks, agents)
        
        # Combined score
        weighted_score = sum(scores.values())
        final_score = (weighted_score * 0.7 + base_suitability * 0.3)
        
        logger.debug("Strategy %s scored: %f (breakdown: %s)", strategy.name, final_score, scores)
        
        return final_score
    
    def _evaluate_complexity_fit(self, strategy: BaseCoordinationStrategy, complexity: float) -> float:
        """Evaluate how well a strategy handles task complexity."""
        strategy_complexity_preferences = {
            'Sequential': (0.0, 0.6),  # Good for low-medium complexity
            'Parallel': (0.3, 0.8),   # Good for medium-high complexity
            'Swarm-Based': (0.5, 1.0) # Excellent for high complexity
        }
        
        min_complexity, max_complexity = strategy_complexity_preferences.get(
            strategy.name, (0.0, 1.0)
        )
        
        if min_complexity <= complexity <= max_complexity:
            return 1.0
        elif complexity < min_complexity:
            return max(0.0, 1.0 - (min_complexity - complexity) * 2)
        else:
            return max(0.0, 1.0 - (complexity - max_complexity) * 2)
    
    def _evaluate_team_fit(self, strategy: BaseCoordinationStrategy, team_size: int, collaboration_history: float) -> float:
        """Evaluate strategy fit for team characteristics."""
        strategy_team_preferences = {
            'Sequential': (1, 5),     # Good for small teams
            'Parallel': (3, 10),     # Good for medium-large teams  
            'Swarm-Based': (3, 20)   # Scales well with larger teams
        }
        
        min_team, max_team = strategy_team_preferences.get(strategy.name, (1, 20))
        
        # Team size fit
        if min_team <= team_size <= max_team:
            size_fit = 1.0
        elif team_size < min_team:
            size_fit = team_size / min_team
        else:
            size_fit = max_team / team_size
        
        # Collaboration history bonus/penalty
        collaboration_bonus = collaboration_history * 0.2
        
        return min(1.0, size_fit + collaboration_bonus)
    
    def _evaluate_time_fit(self, strategy: BaseCoordinationStrategy, time_pressure: float, tasks: List[Task], agents: List[Agent]) -> float:
        """Evaluate strategy efficiency under time pressure."""
        
        # Estimate execution time for this strategy
        estimated_time = strategy.estimate_execution_time(tasks, agents)
        
        # Normalize against sequential execution time
        sequential_time = sum(task.estimated_duration for task in tasks)
        efficiency_ratio = sequential_time / max(estimated_time, 1)
        
        # Higher time pressure requires more efficient strategies
        efficiency_score = min(1.0, efficiency_ratio)
        
        # Time pressure adjustment
        if time_pressure > 0.7:  # High pressure
            return efficiency_score
        elif time_pressure < 0.3:  # Low pressure
            return 0.7 + efficiency_score * 0.3  # Less emphasis on efficiency
        else:  # Medium pressure
            return 0.5 + efficiency_score * 0.5
    
    def _evaluate_historical_performance(self, strategy: BaseCoordinationStrategy, tasks: List[Task], agents: List[Agent]) -> float:
        """Evaluate strategy based on historical performance."""
        
        strategy_name = strategy.name
        if strategy_name not in self.performance_history:
            return 0.5  # Neutral score for new strategies
        
        history = self.performance_history[strategy_name]
        
        if not history:
            return 0.5
        
        # Find similar contexts in history
        current_context_signature = self._create_context_signature(tasks, agents)
        similar_executions = []
        
        for execution in history[-10:]:  # Last 10 executions
            execution_signature = execution.get('context_signature', {})
            similarity = self._calculate_context_similarity(current_context_signature, execution_signature)
            
            if similarity > 0.7:  # Similar enough context
                performance = execution.get('performance_score', 0.0)
                similar_executions.append(performance)
        
        if similar_executions:
            return float(np.mean(similar_executions))
        else:
            # Fall back to overall average
            all_performances = [exec.get('performance_score', 0.0) for exec in history]
            return float(np.mean(all_performances)) if all_performances else 0.5
    
    def _evaluate_resource_fit(self, strategy: BaseCoordinationStrategy, resource_constraints: float) -> float:
        """Evaluate how well strategy handles resource constraints."""
        
        strategy_resource_efficiency = {
            'Sequential': 0.9,    # Very efficient with resources
            'Parallel': 0.6,     # Requires more resources
            'Swarm-Based': 0.7   # Moderate resource usage
        }
        
        efficiency = strategy_resource_efficiency.get(strategy.name, 0.5)
        
        # Higher constraints require more efficient strategies
        if resource_constraints > 0.7:
            return efficiency
        elif resource_constraints < 0.3:
            return 0.7 + efficiency * 0.3  # Less emphasis on efficiency
        else:
            return 0.5 + efficiency * 0.5
    
    def _evaluate_risk_fit(self, strategy: BaseCoordinationStrategy, risk_tolerance: float) -> float:
        """Evaluate strategy risk profile against tolerance."""
        
        strategy_risk_levels = {
            'Sequential': 0.2,    # Low risk, predictable
            'Parallel': 0.5,     # Medium risk, coordination challenges
            'Swarm-Based': 0.7   # Higher risk, complex coordination
        }
        
        strategy_risk = strategy_risk_levels.get(strategy.name, 0.5)
        
        # Calculate fit based on risk tolerance
        risk_diff = abs(strategy_risk - risk_tolerance)
        return max(0.0, 1.0 - risk_diff * 2)
    
    def _create_context_signature(self, tasks: List[Task], agents: List[Agent]) -> Dict[str, float]:
        """Create a signature for the current context for similarity matching."""
        return {
            'task_count': float(len(tasks)),
            'agent_count': float(len(agents)),
            'avg_complexity': float(np.mean([task.complexity for task in tasks])) if tasks else 0.0,
            'dependency_density': self._calculate_dependency_density(tasks),
            'team_avg_load': float(np.mean([agent.current_load for agent in agents])) if agents else 0.0
        }
    
    def _calculate_context_similarity(self, sig1: Dict[str, float], sig2: Dict[str, float]) -> float:
        """Calculate similarity between two context signatures."""
        if not sig1 or not sig2:
            return 0.0
        
        similarities = []
        for key in sig1:
            if key in sig2:
                val1, val2 = sig1[key], sig2[key]
                if max(val1, val2) > 0:
                    similarity = 1.0 - abs(val1 - val2) / max(val1, val2)
                    similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _select_best_strategy(self, strategy_scores: Dict[BaseCoordinationStrategy, float]) -> Tuple[Optional[BaseCoordinationStrategy], float]:
        """Select the best strategy from scored options."""
        
        if not strategy_scores:
            logger.warning("No strategies available for selection")
            return None, 0.0
        
        best_strategy = max(strategy_scores.keys(), key=lambda s: strategy_scores[s])
        confidence = strategy_scores[best_strategy]
        
        logger.info("Selected strategy: %s with confidence: %f", best_strategy.name, confidence)
        
        return best_strategy, confidence
    
    async def _generate_recommendation(
        self,
        best_strategy: BaseCoordinationStrategy,
        confidence: float,
        tasks: List[Task],
        agents: List[Agent],
        context: ContextualFactors,
        all_scores: Dict[BaseCoordinationStrategy, float]
    ) -> StrategyRecommendation:
        """Generate a comprehensive strategy recommendation."""
        
        # Generate reasoning
        reasoning = self._generate_reasoning(best_strategy, context, confidence)
        
        # Select fallback strategies
        sorted_strategies = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        fallback_strategies = [strategy for strategy, _ in sorted_strategies[1:3]]  # Top 2 alternatives
        
        # Predict expected performance
        expected_performance = await self._predict_performance(best_strategy, tasks, agents)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(best_strategy, context, tasks, agents)
        
        return StrategyRecommendation(
            primary_strategy=best_strategy,
            confidence_score=confidence,
            reasoning=reasoning,
            fallback_strategies=fallback_strategies,
            expected_performance=expected_performance,
            risk_factors=risk_factors
        )
    
    def _generate_reasoning(self, strategy: BaseCoordinationStrategy, context: ContextualFactors, confidence: float) -> List[str]:
        """Generate human-readable reasoning for strategy selection."""
        reasoning = []
        
        reasoning.append(f"Selected {strategy.name} strategy with {confidence:.1%} confidence")
        
        if context.task_complexity_score > 0.7:
            reasoning.append("High task complexity detected - strategy provides good optimization")
        elif context.task_complexity_score < 0.3:
            reasoning.append("Low task complexity - strategy offers simplicity and reliability")
        
        if context.team_size >= 5:
            reasoning.append(f"Large team size ({context.team_size} agents) - strategy scales well")
        elif context.team_size <= 2:
            reasoning.append(f"Small team size ({context.team_size} agents) - strategy minimizes coordination overhead")
        
        if context.time_pressure > 0.7:
            reasoning.append("High time pressure - strategy optimizes for speed")
        elif context.time_pressure < 0.3:
            reasoning.append("Low time pressure - strategy focuses on quality and thoroughness")
        
        if context.dependency_density > 0.6:
            reasoning.append("High task interdependency - strategy handles dependencies well")
        elif context.dependency_density < 0.2:
            reasoning.append("Independent tasks - strategy enables parallel execution")
        
        if context.resource_constraints > 0.6:
            reasoning.append("Resource constraints detected - strategy is resource-efficient")
        
        return reasoning
    
    async def _predict_performance(self, strategy: BaseCoordinationStrategy, tasks: List[Task], agents: List[Agent]) -> Dict[str, float]:
        """Predict expected performance metrics for the strategy."""
        
        # Base predictions from strategy assessment
        base_suitability = strategy.assess_suitability(tasks, agents)
        estimated_time = strategy.estimate_execution_time(tasks, agents)
        
        # Historical performance if available
        strategy_name = strategy.name
        historical_success = 0.75  # Default
        
        if strategy_name in self.performance_history:
            history = self.performance_history[strategy_name]
            if history:
                recent_performances = [exec.get('performance_score', 0.0) for exec in history[-5:]]
                historical_success = np.mean(recent_performances)
        
        return {
            'success_probability': float(min(0.95, base_suitability * 0.6 + historical_success * 0.4)),
            'estimated_completion_time_hours': float(estimated_time / 60),
            'resource_efficiency': float(1.0 - (estimated_time / sum(task.estimated_duration for task in tasks))),
            'coordination_overhead': 0.1 if strategy.name == 'Sequential' else 0.2,
            'quality_score': float(base_suitability * 0.8 + 0.2)
        }
    
    def _identify_risk_factors(self, strategy: BaseCoordinationStrategy, context: ContextualFactors, tasks: List[Task], agents: List[Agent]) -> List[str]:
        """Identify potential risk factors for the selected strategy."""
        risks = []
        
        if context.time_pressure > 0.8:
            risks.append("Extremely tight deadlines may compromise quality")
        
        if context.resource_constraints > 0.7:
            risks.append("Resource constraints may impact strategy effectiveness")
        
        if strategy.name == 'Parallel' and context.dependency_density > 0.5:
            risks.append("High task dependencies may create coordination bottlenecks")
        
        if context.team_size > 10 and strategy.name != 'Swarm-Based':
            risks.append("Large team size may overwhelm coordination capacity")
        
        if context.collaboration_history < 0.4:
            risks.append("Poor team collaboration history may impact execution")
        
        # Agent-specific risks
        available_agents = [agent for agent in agents if agent.availability]
        if len(available_agents) < len(tasks) / 2:
            risks.append("Agent shortage may create execution bottlenecks")
        
        high_load_agents = [agent for agent in agents if agent.current_load > 0.8]
        if len(high_load_agents) > len(agents) / 2:
            risks.append("High agent workload may reduce performance quality")
        
        return risks
    
    def _record_selection(self, recommendation: StrategyRecommendation, context: ContextualFactors):
        """Record strategy selection for future learning."""
        strategy_name = recommendation.primary_strategy.name
        
        if strategy_name not in self.performance_history:
            self.performance_history[strategy_name] = []
        
        record = {
            'timestamp': datetime.now(),
            'context_signature': {
                'complexity': context.task_complexity_score,
                'team_size': context.team_size,
                'time_pressure': context.time_pressure,
                'resource_constraints': context.resource_constraints
            },
            'confidence': recommendation.confidence_score,
            'expected_performance': recommendation.expected_performance,
            'risk_factors': recommendation.risk_factors
        }
        
        self.performance_history[strategy_name].append(record)
        
        # Keep only recent history (last 50 records)
        if len(self.performance_history[strategy_name]) > 50:
            self.performance_history[strategy_name] = self.performance_history[strategy_name][-50:]
    
    def _create_fallback_recommendation(self) -> StrategyRecommendation:
        """Create a fallback recommendation when selection fails."""
        # Default to sequential strategy
        sequential_strategy = self.strategy_library.get_strategy(CoordinationPattern.SEQUENTIAL)
        
        if not sequential_strategy:
            # This should never happen, but create a minimal fallback
            logger.error("No fallback strategy available")
            # Create a mock strategy for the fallback
            class FallbackStrategy:
                name = "Manual Coordination"
                
            fallback_strategy = FallbackStrategy()
            
            return StrategyRecommendation(
                primary_strategy=fallback_strategy,  # type: ignore
                confidence_score=0.0,
                reasoning=["Strategy selection failed - manual coordination required"],
                fallback_strategies=[],
                expected_performance={},
                risk_factors=["System error - unable to recommend strategy"]
            )
        
        return StrategyRecommendation(
            primary_strategy=sequential_strategy,
            confidence_score=0.5,
            reasoning=["Fallback to sequential strategy due to selection error"],
            fallback_strategies=[],
            expected_performance={
                'success_probability': 0.7,
                'estimated_completion_time_hours': 2.0,
                'resource_efficiency': 0.8,
                'coordination_overhead': 0.1,
                'quality_score': 0.8
            },
            risk_factors=["Using fallback strategy - may not be optimal"]
        )
    
    def update_performance_feedback(self, strategy_name: str, actual_performance: Dict[str, Any]):
        """Update performance history with actual execution results."""
        if strategy_name not in self.performance_history:
            return
        
        # Find the most recent selection for this strategy
        recent_records = self.performance_history[strategy_name]
        if recent_records:
            latest_record = recent_records[-1]
            latest_record['actual_performance'] = actual_performance
            latest_record['performance_score'] = actual_performance.get('performance_score', 0.0)
            
            logger.info("Updated performance feedback for strategy %s: %f", 
                       strategy_name, actual_performance.get('performance_score', 0.0))
    
    def get_strategy_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all strategies."""
        summary = {}
        
        for strategy_name, history in self.performance_history.items():
            if history:
                performances = [record.get('performance_score', 0.0) for record in history if 'performance_score' in record]
                
                if performances:
                    summary[strategy_name] = {
                        'average_performance': np.mean(performances),
                        'best_performance': max(performances),
                        'worst_performance': min(performances),
                        'consistency': 1.0 - np.std(performances),
                        'execution_count': len(performances)
                    }
        
        return summary