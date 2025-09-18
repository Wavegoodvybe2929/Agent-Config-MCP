"""
Dynamic Coordination Strategy Library for MCP Swarm Intelligence Server

This module implements multiple coordination patterns for multi-agent task execution,
providing a comprehensive strategy pattern implementation for different coordination
scenarios based on task complexity, team size, and performance requirements.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import asyncio
import logging
from dataclasses import dataclass, asdict
import time
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)


class CoordinationPattern(Enum):
    """Enumeration of available coordination patterns."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    HIERARCHICAL = "hierarchical"
    SWARM_BASED = "swarm_based"
    HYBRID = "hybrid"
    DEMOCRATIC = "democratic"
    EMERGENCY = "emergency"


@dataclass
class Task:
    """Represents a task to be coordinated."""
    id: str
    description: str
    complexity: float
    estimated_duration: int  # minutes
    dependencies: List[str]
    required_capabilities: List[str]
    priority: int
    deadline: Optional[datetime] = None


@dataclass
class Agent:
    """Represents an agent in the coordination system."""
    id: str
    name: str
    capabilities: List[str]
    current_load: float
    expertise_scores: Dict[str, float]
    availability: bool
    performance_history: Dict[str, float]


@dataclass
class CoordinationResult:
    """Result of coordination strategy execution."""
    success: bool
    execution_time: float
    tasks_completed: int
    agents_utilized: int
    performance_score: float
    issues_encountered: List[str]
    recommendations: List[str]


@dataclass
class CoordinationStrategy:
    """Configuration for a coordination strategy."""
    pattern: CoordinationPattern
    complexity_threshold: float
    team_size_range: Tuple[int, int]
    success_rate: float
    estimated_time_factor: float
    risk_level: float
    adaptability: float
    resource_efficiency: float


class BaseCoordinationStrategy(ABC):
    """Abstract base class for coordination strategies."""
    
    def __init__(self, name: str, pattern: CoordinationPattern):
        self.name = name
        self.pattern = pattern
        self.execution_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    async def execute(self, tasks: List[Task], agents: List[Agent]) -> CoordinationResult:
        """Execute the coordination strategy."""
        raise NotImplementedError
    
    @abstractmethod
    def estimate_execution_time(self, tasks: List[Task], agents: List[Agent]) -> float:
        """Estimate execution time for the strategy."""
        raise NotImplementedError
    
    @abstractmethod
    def assess_suitability(self, tasks: List[Task], agents: List[Agent]) -> float:
        """Assess how suitable this strategy is for the given context."""
        raise NotImplementedError
    
    def record_execution(self, result: CoordinationResult, context: Dict[str, Any]):
        """Record execution results for learning."""
        self.execution_history.append({
            "timestamp": datetime.now(),
            "result": asdict(result),
            "context": context
        })


class SequentialStrategy(BaseCoordinationStrategy):
    """Linear task execution with handoffs."""
    
    def __init__(self):
        super().__init__("Sequential", CoordinationPattern.SEQUENTIAL)
    
    async def execute(self, tasks: List[Task], agents: List[Agent]) -> CoordinationResult:
        """Execute tasks sequentially with optimal agent assignment."""
        start_time = time.time()
        completed_tasks = 0
        utilized_agents = set()
        issues = []
        
        try:
            # Sort tasks by priority and dependencies
            sorted_tasks = self._sort_tasks_by_dependencies(tasks)
            
            for task in sorted_tasks:
                # Find best available agent
                best_agent = self._find_best_agent(task, agents)
                if not best_agent:
                    issues.append(f"No suitable agent found for task {task.id}")
                    continue
                
                # Execute task
                await self._execute_task_with_agent(task, best_agent)
                completed_tasks += 1
                utilized_agents.add(best_agent.id)
                
                # Update agent availability
                best_agent.current_load += task.complexity
                
        except RuntimeError as e:
            issues.append(f"Sequential execution error: {str(e)}")
            logger.error("Sequential strategy execution failed: %s", str(e))
        
        execution_time = time.time() - start_time
        performance_score = self._calculate_performance_score(
            completed_tasks, len(tasks), execution_time, issues
        )
        
        return CoordinationResult(
            success=completed_tasks > 0,
            execution_time=execution_time,
            tasks_completed=completed_tasks,
            agents_utilized=len(utilized_agents),
            performance_score=performance_score,
            issues_encountered=issues,
            recommendations=self._generate_recommendations(tasks, agents, issues)
        )
    
    def estimate_execution_time(self, tasks: List[Task], agents: List[Agent]) -> float:
        """Estimate sequential execution time."""
        total_time = sum(task.estimated_duration for task in tasks)
        # Add 10% overhead for handoffs
        return total_time * 1.1
    
    def assess_suitability(self, tasks: List[Task], agents: List[Agent]) -> float:
        """Assess suitability for sequential execution."""
        # Good for tasks with strong dependencies
        dependency_score = self._calculate_dependency_density(tasks)
        # Good for small to medium teams
        team_size_score = 1.0 if len(agents) <= 5 else 0.5
        
        return (dependency_score * 0.7 + team_size_score * 0.3)
    
    def _sort_tasks_by_dependencies(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks respecting dependencies."""
        # Topological sort implementation
        sorted_tasks = []
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            # Find tasks with no unresolved dependencies
            ready_tasks = [
                task for task in remaining_tasks
                if all(dep_id in [t.id for t in sorted_tasks] for dep_id in task.dependencies)
            ]
            
            if not ready_tasks:
                # Break circular dependencies by priority
                ready_tasks = [min(remaining_tasks, key=lambda t: t.priority)]
            
            # Sort ready tasks by priority
            ready_tasks.sort(key=lambda t: t.priority)
            next_task = ready_tasks[0]
            
            sorted_tasks.append(next_task)
            remaining_tasks.remove(next_task)
        
        return sorted_tasks
    
    def _find_best_agent(self, task: Task, agents: List[Agent]) -> Optional[Agent]:
        """Find the best agent for a task."""
        suitable_agents = [
            agent for agent in agents
            if agent.availability and
            any(cap in task.required_capabilities for cap in agent.capabilities)
        ]
        
        if not suitable_agents:
            return None
        
        # Score agents based on capability match and current load
        def score_agent(agent: Agent) -> float:
            capability_score = sum(
                agent.expertise_scores.get(cap, 0.0)
                for cap in task.required_capabilities
            ) / max(len(task.required_capabilities), 1)
            
            load_score = 1.0 - min(agent.current_load, 1.0)
            
            return capability_score * 0.7 + load_score * 0.3
        
        return max(suitable_agents, key=score_agent)
    
    async def _execute_task_with_agent(self, task: Task, agent: Agent):
        """Simulate task execution with an agent."""
        # Simulate execution time
        execution_time = task.estimated_duration / 60.0  # Convert to seconds
        await asyncio.sleep(min(execution_time, 0.1))  # Cap simulation time
        
        logger.info("Task %s executed by agent %s", task.id, agent.id)
    
    def _calculate_dependency_density(self, tasks: List[Task]) -> float:
        """Calculate how many tasks have dependencies."""
        if not tasks:
            return 0.0
        
        dependent_tasks = sum(1 for task in tasks if task.dependencies)
        return dependent_tasks / len(tasks)
    
    def _calculate_performance_score(self, completed: int, total: int, exec_time: float, issues: List[str]) -> float:
        """Calculate overall performance score."""
        completion_rate = completed / max(total, 1)
        efficiency_score = 1.0 / (1.0 + exec_time / max(completed, 1))
        issue_penalty = len(issues) * 0.1
        
        return max(0.0, completion_rate * 0.6 + efficiency_score * 0.4 - issue_penalty)
    
    def _generate_recommendations(self, tasks: List[Task], agents: List[Agent], issues: List[str]) -> List[str]:
        """Generate recommendations based on execution results."""
        recommendations = []
        
        if issues:
            recommendations.append("Consider parallel execution for independent tasks")
        
        avg_complexity = sum(task.complexity for task in tasks) / max(len(tasks), 1)
        if avg_complexity > 0.7:
            recommendations.append("High complexity tasks detected - consider hierarchical coordination")
        
        if len(agents) > 3:
            recommendations.append("Large team available - parallel strategy might be more efficient")
        
        return recommendations


class ParallelStrategy(BaseCoordinationStrategy):
    """Concurrent task execution with synchronization."""
    
    def __init__(self):
        super().__init__("Parallel", CoordinationPattern.PARALLEL)
    
    async def execute(self, tasks: List[Task], agents: List[Agent]) -> CoordinationResult:
        """Execute tasks in parallel where possible."""
        start_time = time.time()
        completed_tasks = 0
        utilized_agents = set()
        issues = []
        
        try:
            # Group tasks by dependency levels
            task_groups = self._group_tasks_by_dependency_level(tasks)
            
            for task_group in task_groups:
                # Execute tasks in this group in parallel
                task_results = await self._execute_task_group_parallel(task_group, agents)
                
                for success, agent_id in task_results:
                    if success:
                        completed_tasks += 1
                        if agent_id:
                            utilized_agents.add(agent_id)
                    else:
                        issues.append("Failed to execute task in parallel group")
                
        except RuntimeError as e:
            issues.append(f"Parallel execution error: {str(e)}")
            logger.error("Parallel strategy execution failed: %s", str(e))
        
        execution_time = time.time() - start_time
        performance_score = self._calculate_performance_score(
            completed_tasks, len(tasks), execution_time, issues
        )
        
        return CoordinationResult(
            success=completed_tasks > 0,
            execution_time=execution_time,
            tasks_completed=completed_tasks,
            agents_utilized=len(utilized_agents),
            performance_score=performance_score,
            issues_encountered=issues,
            recommendations=self._generate_recommendations(tasks, agents, issues)
        )
    
    def estimate_execution_time(self, tasks: List[Task], agents: List[Agent]) -> float:
        """Estimate parallel execution time."""
        task_groups = self._group_tasks_by_dependency_level(tasks)
        
        total_time = 0.0
        for group in task_groups:
            # Time for this group is max task time (since parallel)
            group_time = max(task.estimated_duration for task in group)
            total_time += group_time
        
        # Add 5% overhead for synchronization
        return total_time * 1.05
    
    def assess_suitability(self, tasks: List[Task], agents: List[Agent]) -> float:
        """Assess suitability for parallel execution."""
        # Good for independent tasks
        independence_score = 1.0 - self._calculate_dependency_density(tasks)
        # Good for larger teams
        team_size_score = min(len(agents) / 5.0, 1.0)
        # Good for tasks with similar complexity
        complexity_variance = self._calculate_complexity_variance(tasks)
        
        return (independence_score * 0.5 + team_size_score * 0.3 + (1.0 - complexity_variance) * 0.2)
    
    def _group_tasks_by_dependency_level(self, tasks: List[Task]) -> List[List[Task]]:
        """Group tasks by their dependency level for parallel execution."""
        task_dict = {task.id: task for task in tasks}
        levels = []
        remaining_tasks = set(task.id for task in tasks)
        
        while remaining_tasks:
            # Find tasks with no dependencies in remaining tasks
            current_level = []
            for task_id in list(remaining_tasks):
                task = task_dict[task_id]
                if all(dep_id not in remaining_tasks for dep_id in task.dependencies):
                    current_level.append(task)
                    remaining_tasks.remove(task_id)
            
            if current_level:
                levels.append(current_level)
            else:
                # Break circular dependencies
                next_task = min(
                    (task_dict[tid] for tid in remaining_tasks),
                    key=lambda t: t.priority
                )
                levels.append([next_task])
                remaining_tasks.remove(next_task.id)
        
        return levels
    
    async def _execute_task_group_parallel(self, task_group: List[Task], agents: List[Agent]) -> List[Tuple[bool, Optional[str]]]:
        """Execute a group of tasks in parallel."""
        tasks_to_execute = []
        
        for task in task_group:
            agent = self._find_best_available_agent(task, agents)
            if agent:
                tasks_to_execute.append(self._execute_task_with_agent(task, agent))
            else:
                tasks_to_execute.append(self._handle_no_agent_available(task))
        
        results = await asyncio.gather(*tasks_to_execute, return_exceptions=True)
        
        return [(not isinstance(result, Exception), None) for result in results]
    
    def _find_best_available_agent(self, task: Task, agents: List[Agent]) -> Optional[Agent]:
        """Find the best available agent for a task."""
        available_agents = [
            agent for agent in agents
            if agent.availability and agent.current_load < 0.8 and
            any(cap in task.required_capabilities for cap in agent.capabilities)
        ]
        
        if not available_agents:
            return None
        
        # Simple scoring based on capability match
        def score_agent(agent: Agent) -> float:
            capability_score = sum(
                agent.expertise_scores.get(cap, 0.0)
                for cap in task.required_capabilities
            ) / max(len(task.required_capabilities), 1)
            
            return capability_score
        
        return max(available_agents, key=score_agent)
    
    async def _execute_task_with_agent(self, task: Task, agent: Agent):
        """Execute a task with an agent."""
        execution_time = task.estimated_duration / 60.0
        await asyncio.sleep(min(execution_time, 0.1))
        
        # Update agent load
        agent.current_load += task.complexity * 0.5  # Parallel tasks add less load
        
        logger.info("Task %s executed in parallel by agent %s", task.id, agent.id)
    
    async def _handle_no_agent_available(self, task: Task):
        """Handle case where no agent is available for a task."""
        logger.warning("No agent available for task %s", task.id)
        raise RuntimeError(f"No agent available for task {task.id}")
    
    def _calculate_dependency_density(self, tasks: List[Task]) -> float:
        """Calculate dependency density."""
        if not tasks:
            return 0.0
        
        dependent_tasks = sum(1 for task in tasks if task.dependencies)
        return dependent_tasks / len(tasks)
    
    def _calculate_complexity_variance(self, tasks: List[Task]) -> float:
        """Calculate variance in task complexity."""
        if not tasks:
            return 0.0
        
        complexities = [task.complexity for task in tasks]
        mean_complexity = sum(complexities) / len(complexities)
        variance = sum((c - mean_complexity) ** 2 for c in complexities) / len(complexities)
        
        return min(variance, 1.0)  # Normalize to 0-1
    
    def _calculate_performance_score(self, completed: int, total: int, exec_time: float, issues: List[str]) -> float:
        """Calculate performance score for parallel execution."""
        completion_rate = completed / max(total, 1)
        efficiency_score = 1.0 / (1.0 + exec_time / max(completed, 1))
        issue_penalty = len(issues) * 0.1
        
        return max(0.0, completion_rate * 0.6 + efficiency_score * 0.4 - issue_penalty)
    
    def _generate_recommendations(self, tasks: List[Task], agents: List[Agent], issues: List[str]) -> List[str]:
        """Generate recommendations for parallel execution."""
        recommendations = []
        
        if issues:
            recommendations.append("Consider reducing parallel task load or adding more agents")
        
        dependency_density = self._calculate_dependency_density(tasks)
        if dependency_density > 0.5:
            recommendations.append("High dependency density - consider sequential or pipeline strategy")
        
        if len(agents) < len(tasks) / 2:
            recommendations.append("Agent shortage detected - consider hybrid strategy")
        
        return recommendations


class SwarmBasedStrategy(BaseCoordinationStrategy):
    """ACO/PSO coordinated execution with swarm intelligence."""
    
    def __init__(self, swarm_coordinator=None):
        super().__init__("Swarm-Based", CoordinationPattern.SWARM_BASED)
        self.swarm_coordinator = swarm_coordinator
    
    async def execute(self, tasks: List[Task], agents: List[Agent]) -> CoordinationResult:
        """Execute tasks using swarm intelligence coordination."""
        start_time = time.time()
        completed_tasks = 0
        utilized_agents = set()
        issues = []
        
        try:
            if not self.swarm_coordinator:
                issues.append("Swarm coordinator not available")
                return self._create_failed_result(issues)
            
            # Use ACO for task assignment optimization
            assignments = await self._optimize_task_assignments(tasks, agents)
            
            # Execute assignments using swarm coordination
            for task, agent in assignments:
                success = await self._execute_swarm_task(task, agent)
                if success:
                    completed_tasks += 1
                    utilized_agents.add(agent.id)
                else:
                    issues.append(f"Swarm execution failed for task {task.id}")
                
        except RuntimeError as e:
            issues.append(f"Swarm execution error: {str(e)}")
            logger.error("Swarm strategy execution failed: %s", str(e))
        
        execution_time = time.time() - start_time
        performance_score = self._calculate_performance_score(
            completed_tasks, len(tasks), execution_time, issues
        )
        
        return CoordinationResult(
            success=completed_tasks > 0,
            execution_time=execution_time,
            tasks_completed=completed_tasks,
            agents_utilized=len(utilized_agents),
            performance_score=performance_score,
            issues_encountered=issues,
            recommendations=self._generate_swarm_recommendations(tasks, agents, issues)
        )
    
    def estimate_execution_time(self, tasks: List[Task], agents: List[Agent]) -> float:
        """Estimate swarm-based execution time."""
        # Swarm coordination is very efficient for optimal assignment
        base_time = sum(task.estimated_duration for task in tasks) / max(len(agents), 1)
        # Add overhead for swarm coordination
        return base_time * 1.15
    
    def assess_suitability(self, tasks: List[Task], agents: List[Agent]) -> float:
        """Assess suitability for swarm-based execution."""
        # Excellent for complex optimization problems
        complexity_score = sum(task.complexity for task in tasks) / max(len(tasks), 1)
        # Good for medium to large teams
        team_size_score = min(len(agents) / 3.0, 1.0) if len(agents) >= 3 else 0.3
        # Benefits from diverse agent capabilities
        diversity_score = self._calculate_agent_diversity(agents)
        
        return (complexity_score * 0.4 + team_size_score * 0.4 + diversity_score * 0.2)
    
    async def _optimize_task_assignments(self, tasks: List[Task], agents: List[Agent]) -> List[Tuple[Task, Agent]]:
        """Use ACO to optimize task assignments."""
        # Simulate ACO optimization
        assignments = []
        
        for task in tasks:
            best_agent = self._find_optimal_agent(task, agents)
            if best_agent:
                assignments.append((task, best_agent))
        
        return assignments
    
    def _find_optimal_agent(self, task: Task, agents: List[Agent]) -> Optional[Agent]:
        """Find optimal agent using swarm intelligence metrics."""
        suitable_agents = [
            agent for agent in agents
            if agent.availability and
            any(cap in task.required_capabilities for cap in agent.capabilities)
        ]
        
        if not suitable_agents:
            return None
        
        # Use more sophisticated scoring including swarm intelligence
        def swarm_score(agent: Agent) -> float:
            capability_match = sum(
                agent.expertise_scores.get(cap, 0.0)
                for cap in task.required_capabilities
            ) / max(len(task.required_capabilities), 1)
            
            load_balance = 1.0 - min(agent.current_load, 1.0)
            
            # Historical performance factor
            historical_performance = agent.performance_history.get('success_rate', 0.5)
            
            return capability_match * 0.5 + load_balance * 0.3 + historical_performance * 0.2
        
        return max(suitable_agents, key=swarm_score)
    
    async def _execute_swarm_task(self, task: Task, agent: Agent) -> bool:
        """Execute task with swarm coordination."""
        try:
            execution_time = task.estimated_duration / 60.0
            await asyncio.sleep(min(execution_time, 0.1))
            
            # Update agent metrics
            agent.current_load += task.complexity
            success_rate = agent.performance_history.get('success_rate', 0.8)
            agent.performance_history['success_rate'] = min(success_rate + 0.05, 1.0)
            
            logger.info("Swarm task %s executed by agent %s", task.id, agent.id)
            return True
            
        except RuntimeError as e:
            logger.error("Swarm task execution failed: %s", str(e))
            return False
    
    def _calculate_agent_diversity(self, agents: List[Agent]) -> float:
        """Calculate diversity of agent capabilities."""
        if not agents:
            return 0.0
        
        all_capabilities = set()
        for agent in agents:
            all_capabilities.update(agent.capabilities)
        
        if not all_capabilities:
            return 0.0
        
        # Calculate how well capabilities are distributed
        capability_counts = {}
        for cap in all_capabilities:
            capability_counts[cap] = sum(1 for agent in agents if cap in agent.capabilities)
        
        # Shannon diversity index
        total_agents = len(agents)
        diversity = 0.0
        for count in capability_counts.values():
            if count > 0:
                p = count / total_agents
                diversity -= p * (p.bit_length() - 1) if p > 0 else 0
        
        return min(diversity / len(all_capabilities), 1.0) if all_capabilities else 0.0
    
    def _create_failed_result(self, issues: List[str]) -> CoordinationResult:
        """Create a failed coordination result."""
        return CoordinationResult(
            success=False,
            execution_time=0.0,
            tasks_completed=0,
            agents_utilized=0,
            performance_score=0.0,
            issues_encountered=issues,
            recommendations=["Fix swarm coordinator issues", "Consider alternative strategies"]
        )
    
    def _calculate_performance_score(self, completed: int, total: int, exec_time: float, issues: List[str]) -> float:
        """Calculate performance score for swarm execution."""
        completion_rate = completed / max(total, 1)
        efficiency_score = 1.0 / (1.0 + exec_time / max(completed, 1))
        issue_penalty = len(issues) * 0.1
        
        # Swarm intelligence bonus for high completion rates
        swarm_bonus = 0.1 if completion_rate > 0.9 else 0.0
        
        return max(0.0, completion_rate * 0.6 + efficiency_score * 0.4 - issue_penalty + swarm_bonus)
    
    def _generate_swarm_recommendations(self, tasks: List[Task], agents: List[Agent], issues: List[str]) -> List[str]:
        """Generate recommendations for swarm execution."""
        recommendations = []
        
        if issues:
            recommendations.append("Check swarm coordinator configuration and agent connectivity")
        
        diversity_score = self._calculate_agent_diversity(agents)
        if diversity_score < 0.5:
            recommendations.append("Consider adding agents with diverse capabilities for better swarm performance")
        
        avg_complexity = sum(task.complexity for task in tasks) / max(len(tasks), 1)
        if avg_complexity < 0.3:
            recommendations.append("Simple tasks detected - parallel strategy might be more efficient")
        
        return recommendations


class CoordinationStrategyLibrary:
    """Library of coordination strategies with selection and management capabilities."""
    
    def __init__(self, swarm_coordinator=None):
        self.swarm_coordinator = swarm_coordinator
        self.strategies = self._initialize_strategies()
    
    def _initialize_strategies(self) -> Dict[CoordinationPattern, BaseCoordinationStrategy]:
        """Initialize all available coordination strategies."""
        return {
            CoordinationPattern.SEQUENTIAL: SequentialStrategy(),
            CoordinationPattern.PARALLEL: ParallelStrategy(),
            CoordinationPattern.SWARM_BASED: SwarmBasedStrategy(self.swarm_coordinator)
        }
    
    def get_strategy(self, pattern: CoordinationPattern) -> Optional[BaseCoordinationStrategy]:
        """Get a strategy by pattern type."""
        return self.strategies.get(pattern)
    
    def get_all_strategies(self) -> List[BaseCoordinationStrategy]:
        """Get all available strategies."""
        return list(self.strategies.values())
    
    def assess_all_strategies(self, tasks: List[Task], agents: List[Agent]) -> Dict[CoordinationPattern, float]:
        """Assess suitability of all strategies for given context."""
        assessments = {}
        
        for pattern, strategy in self.strategies.items():
            try:
                suitability = strategy.assess_suitability(tasks, agents)
                assessments[pattern] = suitability
            except RuntimeError as e:
                logger.error("Error assessing strategy %s: %s", pattern, str(e))
                assessments[pattern] = 0.0
        
        return assessments
    
    def recommend_best_strategy(self, tasks: List[Task], agents: List[Agent]) -> Optional[BaseCoordinationStrategy]:
        """Recommend the best strategy for given context."""
        assessments = self.assess_all_strategies(tasks, agents)
        
        if not assessments:
            return None
        
        best_pattern = max(assessments.keys(), key=lambda p: assessments[p])
        return self.strategies.get(best_pattern)