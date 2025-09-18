"""
Dynamic Coordination Strategy Tool for MCP Swarm Intelligence Server

This tool provides real-time coordination strategy selection, adaptation, and optimization
for multi-agent workflows using intelligent analysis of task complexity, team composition,
and performance constraints.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import asdict, dataclass

from ..swarm.coordinator import SwarmCoordinator
from ..agents.manager import AgentManager
from .coordination_strategies import CoordinationStrategyLibrary, CoordinationStrategy, BaseCoordinationStrategy, CoordinationPattern, Task, Agent
from .strategy_selector import StrategySelector
from .risk_assessment import RiskAssessmentEngine
from .adaptive_coordination import AdaptiveCoordinationEngine


logger = logging.getLogger(__name__)


@dataclass
class DynamicCoordinationRequest:
    """Request for dynamic coordination strategy selection."""
    
    task_types: List[str]
    agent_count: int
    complexity_level: str  # "low", "medium", "high"
    time_constraints: Optional[float] = None  # in minutes
    quality_requirements: Optional[float] = None  # 0.0 to 1.0
    resource_constraints: Optional[Dict[str, Any]] = None
    preferred_strategy: Optional[str] = None
    risk_tolerance: str = "medium"  # "low", "medium", "high"


@dataclass
class CoordinationResponse:
    """Response containing selected strategy and analysis."""
    
    strategy_name: str
    strategy_config: Dict[str, Any]
    confidence_score: float
    risk_assessment: Dict[str, Any]
    estimated_duration: float
    resource_allocation: Dict[str, Any]
    adaptation_triggers: List[Dict[str, Any]]
    monitoring_metrics: List[str]


class DynamicCoordinationTool:
    """
    MCP Tool for dynamic coordination strategy management.
    
    Provides interface for:
    - Strategy selection based on context
    - Real-time strategy adaptation
    - Performance monitoring and optimization
    - Risk assessment and mitigation
    """
    
    def __init__(self, swarm_coordinator: Optional[SwarmCoordinator] = None, agent_manager: Optional[AgentManager] = None):
        """Initialize the dynamic coordination tool."""
        self.coordinator = swarm_coordinator
        self.agent_manager = agent_manager
        
        # Initialize coordination components
        self.strategy_library = CoordinationStrategyLibrary()
        self.strategy_selector = StrategySelector(self.strategy_library)
        self.risk_assessor = RiskAssessmentEngine()
        self.adaptive_engine = AdaptiveCoordinationEngine()
        
        # Active coordination sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("DynamicCoordinationTool initialized successfully")
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get the MCP tool definition."""
        return {
            "name": "dynamic_coordination",
            "description": "Dynamically select and adapt coordination strategies for multi-agent tasks",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "select_strategy",
                            "adapt_strategy", 
                            "monitor_performance",
                            "assess_risks",
                            "get_active_sessions",
                            "optimize_allocation"
                        ],
                        "description": "Action to perform"
                    },
                    "task_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Types of tasks to coordinate"
                    },
                    "agent_count": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Number of agents available"
                    },
                    "complexity_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Task complexity level"
                    },
                    "time_constraints": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Time constraint in minutes (optional)"
                    },
                    "quality_requirements": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Quality requirement (0.0 to 1.0, optional)"
                    },
                    "resource_constraints": {
                        "type": "object",
                        "description": "Resource constraints (optional)"
                    },
                    "preferred_strategy": {
                        "type": "string",
                        "description": "Preferred strategy name (optional)"
                    },
                    "risk_tolerance": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Risk tolerance level"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID for ongoing coordination (required for some actions)"
                    }
                },
                "required": ["action"]
            }
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the dynamic coordination tool."""
        try:
            action = arguments.get("action")
            
            if action == "select_strategy":
                return await self._select_strategy(arguments)
            elif action == "adapt_strategy":
                return await self._adapt_strategy(arguments)
            elif action == "monitor_performance":
                return await self._monitor_performance(arguments)
            elif action == "assess_risks":
                return await self._assess_risks(arguments)
            elif action == "get_active_sessions":
                return await self._get_active_sessions(arguments)
            elif action == "optimize_allocation":
                return await self._optimize_allocation(arguments)
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error("Error executing dynamic coordination tool: %s", e)
            return [{
                "type": "text",
                "text": f"Error: {str(e)}"
            }]
    
    async def _select_strategy(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select optimal coordination strategy based on context."""
        try:
            # Validate required parameters
            task_types = arguments.get("task_types", [])
            agent_count = arguments.get("agent_count")
            complexity_level = arguments.get("complexity_level", "medium")
            
            if not task_types:
                raise ValueError("task_types is required for strategy selection")
            if agent_count is None:
                raise ValueError("agent_count is required for strategy selection")
            
            # Create request
            request = DynamicCoordinationRequest(
                task_types=task_types,
                agent_count=agent_count,
                complexity_level=complexity_level,
                time_constraints=arguments.get("time_constraints"),
                quality_requirements=arguments.get("quality_requirements"),
                resource_constraints=arguments.get("resource_constraints"),
                preferred_strategy=arguments.get("preferred_strategy"),
                risk_tolerance=arguments.get("risk_tolerance", "medium")
            )
            
            # Create mock tasks and agents for strategy selection
            tasks = [
                Task(
                    id=f"task_{i}",
                    description=f"Task of type {task_type}",
                    complexity=0.5,  # Medium complexity
                    estimated_duration=30,  # 30 minutes default
                    dependencies=[],
                    required_capabilities=[task_type],
                    priority=1
                )
                for i, task_type in enumerate(task_types)
            ]
            
            agents = [
                Agent(
                    id=f"agent_{i}",
                    name=f"Agent {i}",
                    capabilities=task_types,  # All agents can handle all task types
                    current_load=0.5,
                    expertise_scores={task_type: 0.8 for task_type in task_types},
                    availability=True,
                    performance_history={"success_rate": 0.8, "avg_completion_time": 25.0}
                )
                for i in range(agent_count)
            ]
            
            # Create context for strategy selector
            constraints = {
                "time_constraint": request.time_constraints,
                "quality_requirement": request.quality_requirements,
                "resources": request.resource_constraints or {},
                "preferred_strategy": request.preferred_strategy,
                "risk_tolerance": request.risk_tolerance
            }
            
            # Select strategy using actual method
            recommendation = await self.strategy_selector.select_optimal_strategy(
                tasks, agents, constraints
            )
            
            # Assess risks using actual method
            risk_assessment = await self.risk_assessor.assess_coordination_risks(
                recommendation.primary_strategy, tasks, agents, constraints
            )
            
            # Create session
            session_id = f"coord_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Start adaptive monitoring
            await self.adaptive_engine.start_execution_monitoring(
                recommendation.primary_strategy, tasks, agents
            )
            
            # Create response
            response = CoordinationResponse(
                strategy_name=recommendation.primary_strategy.name,
                strategy_config={"pattern": recommendation.primary_strategy.pattern.value},
                confidence_score=recommendation.confidence_score,
                risk_assessment=asdict(risk_assessment),
                estimated_duration=self._estimate_duration(recommendation.primary_strategy, constraints),
                resource_allocation=self._calculate_resource_allocation(
                    recommendation.primary_strategy, constraints
                ),
                adaptation_triggers=self._get_adaptation_triggers(recommendation.primary_strategy),
                monitoring_metrics=self._get_monitoring_metrics(recommendation.primary_strategy)
            )
            
            # Store session
            self.active_sessions[session_id] = {
                "request": asdict(request),
                "response": asdict(response),
                "strategy": recommendation.primary_strategy,
                "context": constraints,
                "start_time": datetime.now().isoformat(),
                "status": "active",
                "tasks": tasks,
                "agents": agents
            }
            
            result = {
                "session_id": session_id,
                "selected_strategy": asdict(response),
                "recommendations": recommendation.reasoning
            }
            
            return [{
                "type": "text",
                "text": f"Strategy Selection Complete:\n\n{self._format_response(result)}"
            }]
            
        except Exception as e:
            logger.error("Error in strategy selection: %s", e)
            raise
    
    async def _adapt_strategy(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adapt coordination strategy based on current performance."""
        try:
            session_id = arguments.get("session_id")
            if not session_id or session_id not in self.active_sessions:
                raise ValueError("Valid session_id is required for strategy adaptation")
            
            session = self.active_sessions[session_id]
            
            # Use the adaptive engine to check if adaptation is needed
            tasks = session.get("tasks", [])
            agents = session.get("agents", [])
            
            # Check for performance issues (simplified approach)
            needs_adaptation = self._check_adaptation_needs(session)
            
            if needs_adaptation["needs_adaptation"]:
                # Select new strategy
                new_recommendation = await self.strategy_selector.select_optimal_strategy(
                    tasks, agents, session["context"]
                )
                
                # Update session
                session["strategy"] = new_recommendation.primary_strategy
                session["adaptations"] = session.get("adaptations", [])
                session["adaptations"].append({
                    "timestamp": datetime.now().isoformat(),
                    "triggers": needs_adaptation["triggers"],
                    "new_strategy": new_recommendation.primary_strategy.name,
                    "reason": "Performance optimization"
                })
                
                result = {
                    "session_id": session_id,
                    "adaptation_applied": True,
                    "new_strategy": new_recommendation.primary_strategy.name,
                    "triggers": needs_adaptation["triggers"],
                    "performance_improvement": 0.1  # Estimated improvement
                }
            else:
                result = {
                    "session_id": session_id,
                    "adaptation_applied": False,
                    "current_strategy": session["strategy"].name,
                    "performance_status": "optimal"
                }
            
            return [{
                "type": "text",
                "text": f"Strategy Adaptation Result:\n\n{self._format_response(result)}"
            }]
            
        except Exception as e:
            logger.error("Error in strategy adaptation: %s", e)
            raise
    
    async def _monitor_performance(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Monitor performance of active coordination strategies."""
        try:
            session_id = arguments.get("session_id")
            
            if session_id:
                # Monitor specific session
                if session_id not in self.active_sessions:
                    raise ValueError(f"Session {session_id} not found")
                
                metrics = self._get_session_metrics(session_id)
                result = {
                    "session_id": session_id,
                    "performance_metrics": metrics,
                    "status": self.active_sessions[session_id]["status"]
                }
            else:
                # Monitor all active sessions
                all_metrics = {}
                for sid in self.active_sessions:
                    if self.active_sessions[sid]["status"] == "active":
                        metrics = self._get_session_metrics(sid)
                        all_metrics[sid] = metrics
                
                result = {
                    "active_sessions": len(all_metrics),
                    "session_metrics": all_metrics,
                    "summary": self._generate_performance_summary(all_metrics)
                }
            
            return [{
                "type": "text",
                "text": f"Performance Monitoring:\n\n{self._format_response(result)}"
            }]
            
        except Exception as e:
            logger.error("Error in performance monitoring: %s", e)
            raise
    
    async def _assess_risks(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Assess risks for coordination strategies."""
        try:
            session_id = arguments.get("session_id")
            
            if session_id and session_id in self.active_sessions:
                # Assess risks for specific session
                session = self.active_sessions[session_id]
                
                risk_assessment = await self.risk_assessor.assess_coordination_risks(
                    session["strategy"], 
                    session.get("tasks", []), 
                    session.get("agents", []), 
                    session["context"]
                )
                
                result = {
                    "session_id": session_id,
                    "strategy": session["strategy"].name,
                    "risk_assessment": asdict(risk_assessment),
                    "mitigation_recommendations": risk_assessment.recommendations
                }
            else:
                # General risk assessment based on parameters
                task_types = arguments.get("task_types", ["general"])
                agent_count = arguments.get("agent_count", 1)
                
                # Create mock tasks and agents
                tasks = [Task(
                    id=f"task_{i}", 
                    description=f"Task of type {task_type}",
                    complexity=0.5,
                    estimated_duration=30, 
                    dependencies=[], 
                    required_capabilities=[task_type],
                    priority=1
                ) for i, task_type in enumerate(task_types)]
                
                agents = [Agent(
                    id=f"agent_{i}", 
                    name=f"Agent {i}",
                    capabilities=task_types, 
                    current_load=0.5,
                    expertise_scores={task_type: 0.8 for task_type in task_types},
                    availability=True,
                    performance_history={"success_rate": 0.8}
                ) for i in range(agent_count)]
                
                context = {
                    "complexity": arguments.get("complexity_level", "medium"),
                    "time_constraint": arguments.get("time_constraints"),
                    "quality_requirement": arguments.get("quality_requirements"),
                    "risk_tolerance": arguments.get("risk_tolerance", "medium")
                }
                
                # Use a default strategy for assessment
                default_strategy = self.strategy_library.get_strategy(CoordinationPattern.PARALLEL)
                
                if default_strategy is None:
                    raise RuntimeError("Could not create default strategy for risk assessment")
                
                risk_assessment = await self.risk_assessor.assess_coordination_risks(
                    default_strategy, tasks, agents, context
                )
                
                result = {
                    "context": context,
                    "risk_assessment": asdict(risk_assessment),
                    "general_recommendations": risk_assessment.recommendations
                }
            
            return [{
                "type": "text",
                "text": f"Risk Assessment:\n\n{self._format_response(result)}"
            }]
            
        except Exception as e:
            logger.error("Error in risk assessment: %s", e)
            raise
    
    async def _get_active_sessions(self, _arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get information about active coordination sessions."""
        try:
            active_sessions = {
                sid: {
                    "strategy": session["strategy"].name,
                    "start_time": session["start_time"],
                    "status": session["status"],
                    "task_types": session["request"]["task_types"],
                    "agent_count": session["request"]["agent_count"],
                    "adaptations": len(session.get("adaptations", []))
                }
                for sid, session in self.active_sessions.items()
                if session["status"] == "active"
            }
            
            result = {
                "total_active_sessions": len(active_sessions),
                "sessions": active_sessions,
                "summary": {
                    "most_used_strategy": self._get_most_used_strategy(),
                    "average_duration": self._get_average_session_duration(),
                    "total_adaptations": sum(
                        len(session.get("adaptations", []))
                        for session in self.active_sessions.values()
                    )
                }
            }
            
            return [{
                "type": "text",
                "text": f"Active Coordination Sessions:\n\n{self._format_response(result)}"
            }]
            
        except Exception as e:
            logger.error("Error getting active sessions: %s", e)
            raise
    
    async def _optimize_allocation(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize resource allocation for coordination strategies."""
        try:
            session_id = arguments.get("session_id")
            
            if session_id and session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                
                # Get current performance
                current_metrics = self._get_session_metrics(session_id)
                
                # Optimize allocation
                optimized_allocation = await self._calculate_optimized_allocation(
                    session["strategy"], session["context"], current_metrics
                )
                
                result = {
                    "session_id": session_id,
                    "current_allocation": session["response"]["resource_allocation"],
                    "optimized_allocation": optimized_allocation,
                    "expected_improvement": self._calculate_improvement_estimate(
                        session["response"]["resource_allocation"], optimized_allocation
                    ),
                    "implementation_cost": self._calculate_implementation_cost(
                        session["response"]["resource_allocation"], optimized_allocation
                    )
                }
            else:
                raise ValueError("Valid session_id is required for allocation optimization")
            
            return [{
                "type": "text",
                "text": f"Resource Allocation Optimization:\n\n{self._format_response(result)}"
            }]
            
        except Exception as e:
            logger.error("Error in allocation optimization: %s", e)
            raise
    
    # Helper methods
    
    def _check_adaptation_needs(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Check if strategy adaptation is needed."""
        # Simple heuristic for demonstration
        session_duration = (
            datetime.now() - datetime.fromisoformat(session["start_time"])
        ).total_seconds() / 60
        
        triggers = []
        
        if session_duration > 60:  # More than 1 hour
            triggers.append("long_execution_time")
        
        if len(session.get("adaptations", [])) == 0 and session_duration > 30:
            triggers.append("potential_optimization")
        
        return {
            "needs_adaptation": len(triggers) > 0,
            "triggers": triggers
        }
    
    def _get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get performance metrics for a session."""
        session = self.active_sessions[session_id]
        session_duration = (
            datetime.now() - datetime.fromisoformat(session["start_time"])
        ).total_seconds() / 60
        
        # Simulated metrics
        return {
            "duration_minutes": session_duration,
            "throughput": 0.8,  # Tasks per minute
            "resource_utilization": 0.7,
            "error_rate": 0.05,
            "quality_score": 0.85,
            "overall_performance": 0.8
        }
    
    def _estimate_duration(self, strategy: Union[BaseCoordinationStrategy, CoordinationStrategy], context: Dict[str, Any]) -> float:
        """Estimate coordination duration in minutes."""
        base_duration = 30.0  # Base 30 minutes
        
        # Adjust for complexity (if available in context)
        complexity = context.get("complexity", "medium")
        complexity_multiplier = {
            "low": 0.5,
            "medium": 1.0,
            "high": 1.8
        }.get(complexity, 1.0)
        
        # Adjust for strategy type - handle both BaseCoordinationStrategy and CoordinationStrategy
        if isinstance(strategy, BaseCoordinationStrategy):
            strategy_name = strategy.name
        else:
            strategy_name = strategy.pattern.value
            
        strategy_multiplier = {
            "sequential": 1.5,
            "parallel": 0.8,
            "swarm_based": 1.2
        }.get(strategy_name, 1.0)
        
        return base_duration * complexity_multiplier * strategy_multiplier
    
    def _calculate_resource_allocation(self, strategy: Union[BaseCoordinationStrategy, CoordinationStrategy], _context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resource allocation for the strategy."""
        # Handle both BaseCoordinationStrategy and CoordinationStrategy
        if isinstance(strategy, BaseCoordinationStrategy):
            strategy_name = strategy.name
        else:
            strategy_name = strategy.pattern.value
        
        if strategy_name == "sequential":
            return {
                "coordination_overhead": 0.1,
                "communication_overhead": 0.05,
                "agent_utilization": 0.6
            }
        elif strategy_name == "parallel":
            return {
                "coordination_overhead": 0.2,
                "communication_overhead": 0.15,
                "synchronization_overhead": 0.1,
                "agent_utilization": 0.8
            }
        elif strategy_name == "swarm_based":
            return {
                "coordination_overhead": 0.3,
                "communication_overhead": 0.25,
                "consensus_overhead": 0.2,
                "optimization_overhead": 0.15,
                "agent_utilization": 0.9
            }
        else:
            return {
                "coordination_overhead": 0.15,
                "communication_overhead": 0.1,
                "agent_utilization": 0.7
            }
    
    def _get_adaptation_triggers(self, _strategy: Union[BaseCoordinationStrategy, CoordinationStrategy]) -> List[Dict[str, Any]]:
        """Get adaptation triggers for the strategy."""
        return [
            {
                "type": "performance_degradation",
                "threshold": 0.7,
                "description": "Adapt when performance drops below 70%"
            },
            {
                "type": "resource_exhaustion",
                "threshold": 0.9,
                "description": "Adapt when resource utilization exceeds 90%"
            },
            {
                "type": "timeline_deviation",
                "threshold": 1.5,
                "description": "Adapt when timeline deviates by more than 50%"
            }
        ]
    
    def _get_monitoring_metrics(self, strategy: Union[BaseCoordinationStrategy, CoordinationStrategy]) -> List[str]:
        """Get monitoring metrics for the strategy."""
        base_metrics = [
            "throughput",
            "latency",
            "resource_utilization",
            "error_rate",
            "quality_score"
        ]
        
        # Handle both BaseCoordinationStrategy and CoordinationStrategy
        if isinstance(strategy, BaseCoordinationStrategy):
            strategy_name = strategy.name
        else:
            strategy_name = strategy.pattern.value
        
        if strategy_name == "swarm_based":
            base_metrics.extend([
                "consensus_time",
                "convergence_rate"
            ])
        elif strategy_name == "parallel":
            base_metrics.extend([
                "synchronization_time",
                "load_balance"
            ])
        
        return base_metrics
    
    def _get_most_used_strategy(self) -> str:
        """Get the most frequently used strategy."""
        strategy_counts = {}
        for session in self.active_sessions.values():
            strategy_name = session["strategy"].name
            strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
        
        if strategy_counts:
            return max(strategy_counts.keys(), key=lambda x: strategy_counts[x])
        return "none"
    
    def _get_average_session_duration(self) -> float:
        """Get average session duration in minutes."""
        durations = []
        current_time = datetime.now()
        
        for session in self.active_sessions.values():
            start_time = datetime.fromisoformat(session["start_time"])
            duration = (current_time - start_time).total_seconds() / 60
            durations.append(duration)
        
        return sum(durations) / len(durations) if durations else 0.0
    
    async def _calculate_optimized_allocation(self, strategy: Union[BaseCoordinationStrategy, CoordinationStrategy], context: Dict[str, Any], current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimized resource allocation."""
        current_allocation = self._calculate_resource_allocation(strategy, context)
        
        # Simple optimization based on current performance
        optimization_factor = 1.0
        if current_metrics.get("resource_utilization", 0.5) > 0.8:
            optimization_factor = 0.9  # Reduce overhead
        elif current_metrics.get("throughput", 0.5) < 0.5:
            optimization_factor = 1.1  # Increase resources
        
        optimized = {}
        for key, value in current_allocation.items():
            if isinstance(value, (int, float)):
                optimized[key] = value * optimization_factor
            else:
                optimized[key] = value
        
        return optimized
    
    def _calculate_improvement_estimate(self, current: Dict[str, Any], optimized: Dict[str, Any]) -> float:
        """Calculate expected improvement from optimization."""
        # Simple heuristic: average change in numeric values
        changes = []
        for key in current:
            if key in optimized and isinstance(current[key], (int, float)):
                change = abs(optimized[key] - current[key]) / max(current[key], 0.01)
                changes.append(change)
        
        return sum(changes) / len(changes) if changes else 0.0
    
    def _calculate_implementation_cost(self, current: Dict[str, Any], optimized: Dict[str, Any]) -> float:
        """Calculate cost of implementing optimization."""
        # Simplified cost model based on magnitude of changes
        return self._calculate_improvement_estimate(current, optimized) * 0.5
    
    def _generate_performance_summary(self, all_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance summary across all sessions."""
        if not all_metrics:
            return {"average_performance": 0.0, "total_throughput": 0.0}
        
        total_throughput = sum(
            metrics.get("throughput", 0.0) for metrics in all_metrics.values()
        )
        average_performance = sum(
            metrics.get("overall_performance", 0.0) for metrics in all_metrics.values()
        ) / len(all_metrics)
        
        return {
            "average_performance": average_performance,
            "total_throughput": total_throughput,
            "session_count": len(all_metrics)
        }
    
    def _format_response(self, data: Dict[str, Any]) -> str:
        """Format response data for display."""
        return json.dumps(data, indent=2, default=str)
    
