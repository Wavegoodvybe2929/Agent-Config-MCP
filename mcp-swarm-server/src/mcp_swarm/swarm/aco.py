"""
Ant Colony Optimization (ACO) Engine for Optimal Agent-Task Assignment

This module implements ant colony optimization algorithms for finding optimal
assignments between agents and tasks in the MCP Swarm Intelligence Server.
The implementation includes pheromone trail management, heuristic calculations,
and multi-objective optimization capabilities.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import asyncio
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """Represents an agent in the swarm with capabilities and current state."""
    id: str
    capabilities: List[str]
    current_load: float  # 0.0 to 1.0
    success_rate: float  # Historical success rate (0.0 to 1.0)
    availability: bool
    specialty_weights: Dict[str, float] = field(default_factory=dict)
    last_task_completion: Optional[datetime] = None
    total_tasks_completed: int = 0
    average_completion_time: float = 0.0


@dataclass
class Task:
    """Represents a task that needs to be assigned to an agent."""
    id: str
    requirements: List[str]
    complexity: float  # 0.0 to 1.0
    priority: int  # 1 (low) to 5 (critical)
    deadline: Optional[datetime] = None
    estimated_duration: float = 1.0  # Hours
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)


@dataclass
class Assignment:
    """Represents an agent-task assignment with metadata."""
    agent_id: str
    task_id: str
    assignment_time: datetime
    confidence: float
    estimated_completion: datetime
    cost: float


class AntColonyOptimizer:
    """
    Ant Colony Optimization for optimal agent-task assignment.
    
    This implementation uses ACO principles to find near-optimal assignments
    between available agents and pending tasks, considering multiple objectives:
    - Load balancing across agents
    - Capability matching between agents and tasks
    - Historical success rates and performance
    - Task priorities and deadlines
    """
    
    def __init__(
        self,
        num_ants: int = 50,
        alpha: float = 1.0,      # Pheromone importance
        beta: float = 2.0,       # Heuristic importance
        rho: float = 0.1,        # Evaporation rate
        q: float = 100.0,        # Pheromone deposit amount
        max_iterations: int = 100,
        convergence_threshold: float = 0.001
    ):
        """
        Initialize ACO parameters.
        
        Args:
            num_ants: Number of ants in the colony
            alpha: Pheromone trail importance factor
            beta: Heuristic information importance factor
            rho: Pheromone evaporation rate (0 < rho < 1)
            q: Pheromone deposit constant
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence detection threshold
        """
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Pheromone matrix: [agent_index][task_index] -> pheromone_level
        self.pheromone_matrix: Optional[np.ndarray] = None
        self.heuristic_matrix: Optional[np.ndarray] = None
        
        # Optimization history
        self.best_solution: Optional[Dict[str, str]] = None
        self.best_cost: float = float('inf')
        self.iteration_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.convergence_iteration: Optional[int] = None
        self.total_evaluations: int = 0
    
    async def find_optimal_assignment(
        self,
        agents: List[Agent],
        tasks: List[Task],
        existing_assignments: Optional[Dict[str, str]] = None
    ) -> Tuple[Dict[str, str], float, Dict[str, Any]]:
        """
        Find optimal agent-task assignments using ACO.
        
        Args:
            agents: List of available agents
            tasks: List of tasks to be assigned
            existing_assignments: Current assignments to preserve
            
        Returns:
            Tuple of (assignments, total_cost, optimization_metadata)
        """
        if not agents or not tasks:
            return {}, 0.0, {"status": "no_work_available"}
        
        # Filter available agents and unassigned tasks
        available_agents = [a for a in agents if a.availability]
        if existing_assignments:
            unassigned_tasks = [t for t in tasks if t.id not in existing_assignments.values()]
        else:
            unassigned_tasks = tasks
            
        if not available_agents or not unassigned_tasks:
            return existing_assignments or {}, 0.0, {"status": "no_optimization_needed"}
        
        # Initialize pheromone and heuristic matrices
        self._initialize_matrices(available_agents, unassigned_tasks)
        
        logger.info(f"Starting ACO optimization: {len(available_agents)} agents, {len(unassigned_tasks)} tasks")
        
        # Run ACO optimization
        best_assignment, best_cost = await self._optimize_assignments(
            available_agents, unassigned_tasks
        )
        
        # Combine with existing assignments
        final_assignments = existing_assignments.copy() if existing_assignments else {}
        final_assignments.update(best_assignment)
        
        # Calculate final cost
        final_cost = self._calculate_assignment_cost(final_assignments, agents, tasks)
        
        # Prepare optimization metadata
        metadata = {
            "status": "optimized",
            "iterations": len(self.iteration_history),
            "convergence_iteration": self.convergence_iteration,
            "total_evaluations": self.total_evaluations,
            "improvement": (self.iteration_history[0]["best_cost"] - best_cost) if self.iteration_history else 0,
            "final_cost": final_cost,
            "optimization_time": sum(h.get("duration", 0) for h in self.iteration_history)
        }
        
        return final_assignments, final_cost, metadata
    
    def _initialize_matrices(self, agents: List[Agent], tasks: List[Task]) -> None:
        """Initialize pheromone and heuristic matrices."""
        num_agents = len(agents)
        num_tasks = len(tasks)
        
        # Initialize pheromone matrix with small positive values
        self.pheromone_matrix = np.ones((num_agents, num_tasks)) * 0.1
        
        # Calculate heuristic matrix based on agent-task compatibility
        self.heuristic_matrix = np.zeros((num_agents, num_tasks))
        
        for i, agent in enumerate(agents):
            for j, task in enumerate(tasks):
                heuristic = self._calculate_heuristic(agent, task)
                self.heuristic_matrix[i, j] = heuristic
    
    def _calculate_heuristic(self, agent: Agent, task: Task) -> float:
        """
        Calculate heuristic value for agent-task pair.
        
        Higher values indicate better matches. Considers:
        - Capability matching
        - Agent load and availability
        - Historical success rate
        - Task complexity vs agent experience
        - Deadline pressure
        """
        # Capability matching score
        capability_score = 0.0
        if task.requirements:
            matched_capabilities = len(set(agent.capabilities) & set(task.requirements))
            capability_score = matched_capabilities / len(task.requirements)
        else:
            capability_score = 1.0  # No specific requirements
        
        # Load balancing factor (prefer less loaded agents)
        load_factor = 1.0 - agent.current_load
        
        # Success rate factor
        success_factor = agent.success_rate
        
        # Complexity matching (experienced agents for complex tasks)
        complexity_factor = 1.0
        if task.complexity > 0.7 and agent.total_tasks_completed > 10:
            complexity_factor = 1.2  # Bonus for experienced agents on complex tasks
        elif task.complexity < 0.3:
            complexity_factor = 1.1  # Simple tasks can go to any agent
        
        # Deadline urgency factor
        urgency_factor = 1.0
        if task.deadline:
            now = datetime.now()
            time_to_deadline = (task.deadline - now).total_seconds() / 3600  # Hours
            if time_to_deadline < task.estimated_duration * 1.5:
                urgency_factor = 1.3  # Urgent tasks get priority
        
        # Priority factor
        priority_factor = 1.0 + (task.priority - 1) * 0.1  # Small boost for higher priority
        
        # Combine all factors
        heuristic = (
            capability_score * 0.4 +
            load_factor * 0.2 +
            success_factor * 0.2 +
            complexity_factor * 0.1 +
            urgency_factor * 0.05 +
            priority_factor * 0.05
        )
        
        return max(heuristic, 0.01)  # Minimum positive value
    
    async def _optimize_assignments(
        self,
        agents: List[Agent],
        tasks: List[Task]
    ) -> Tuple[Dict[str, str], float]:
        """Run the main ACO optimization loop."""
        self.best_solution = None
        self.best_cost = float('inf')
        self.iteration_history = []
        self.total_evaluations = 0
        
        for iteration in range(self.max_iterations):
            iteration_start = datetime.now()
            
            # Generate solutions using ant colony
            solutions = []
            costs = []
            
            for ant in range(self.num_ants):
                solution = self._construct_solution(agents, tasks)
                cost = self._calculate_solution_cost(solution, agents, tasks)
                
                solutions.append(solution)
                costs.append(cost)
                self.total_evaluations += 1
                
                # Update best solution
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = solution.copy()
            
            # Update pheromone trails
            self._update_pheromones(solutions, costs, agents, tasks)
            
            # Record iteration statistics
            iteration_duration = (datetime.now() - iteration_start).total_seconds()
            self.iteration_history.append({
                "iteration": iteration,
                "best_cost": self.best_cost,
                "average_cost": np.mean(costs),
                "std_cost": np.std(costs),
                "duration": iteration_duration
            })
            
            # Check for convergence
            if self._check_convergence():
                self.convergence_iteration = iteration
                logger.info(f"ACO converged at iteration {iteration}")
                break
        
        return self.best_solution or {}, self.best_cost
    
    def _construct_solution(self, agents: List[Agent], tasks: List[Task]) -> Dict[str, str]:
        """Construct a solution using probabilistic ant decision making."""
        solution = {}
        available_agents = set(range(len(agents)))
        available_tasks = set(range(len(tasks)))
        
        # Assign tasks one by one
        while available_tasks and available_agents:
            # Choose a task (can be random or based on priority)
            task_idx = self._select_task(tasks, available_tasks)
            
            # Choose an agent for this task based on pheromone and heuristic
            agent_idx = self._select_agent(task_idx, available_agents)
            
            if agent_idx is not None:
                solution[agents[agent_idx].id] = tasks[task_idx].id
                available_agents.remove(agent_idx)
                available_tasks.remove(task_idx)
                
                # Update agent load for this construction
                agents[agent_idx].current_load += tasks[task_idx].complexity * 0.1
            else:
                # No suitable agent found, skip this task
                available_tasks.remove(task_idx)
        
        # Reset agent loads (this was just for construction)
        for agent in agents:
            agent.current_load = max(0, agent.current_load - sum(
                tasks[j].complexity * 0.1 for j in range(len(tasks))
                if agent.id in solution and solution[agent.id] == tasks[j].id
            ))
        
        return solution
    
    def _select_task(self, tasks: List[Task], available_tasks: Set[int]) -> int:
        """Select next task to assign (priority-based selection)."""
        if not available_tasks:
            return -1
            
        # Prefer higher priority and closer deadline tasks
        task_scores = {}
        for task_idx in available_tasks:
            task = tasks[task_idx]
            score = task.priority
            
            # Add urgency factor
            if task.deadline:
                now = datetime.now()
                time_to_deadline = (task.deadline - now).total_seconds() / 3600
                urgency = max(0, 1.0 - (time_to_deadline / (task.estimated_duration * 2)))
                score += urgency * 2
            
            task_scores[task_idx] = score
        
        # Select task with highest score
        return max(task_scores.keys(), key=lambda k: task_scores[k])
    
    def _select_agent(self, task_idx: int, available_agents: Set[int]) -> Optional[int]:
        """Select agent for task using ACO probability rules."""
        if not available_agents:
            return None
        
        # Calculate selection probabilities
        probabilities = {}
        total_attractiveness = 0.0
        
        for agent_idx in available_agents:
            if self.pheromone_matrix is None or self.heuristic_matrix is None:
                return np.random.choice(list(available_agents)) if available_agents else None
                
            pheromone = self.pheromone_matrix[agent_idx, task_idx]
            heuristic = self.heuristic_matrix[agent_idx, task_idx]
            
            attractiveness = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities[agent_idx] = attractiveness
            total_attractiveness += attractiveness
        
        if total_attractiveness == 0:
            # Random selection if no attractiveness
            return np.random.choice(list(available_agents))
        
        # Normalize probabilities
        for agent_idx in probabilities:
            probabilities[agent_idx] /= total_attractiveness
        
        # Roulette wheel selection
        rand = np.random.random()
        cumulative_prob = 0.0
        
        for agent_idx, prob in probabilities.items():
            cumulative_prob += prob
            if rand <= cumulative_prob:
                return agent_idx
        
        # Fallback to last agent
        return list(available_agents)[-1]
    
    def _calculate_solution_cost(
        self, 
        solution: Dict[str, str], 
        agents: List[Agent], 
        tasks: List[Task]
    ) -> float:
        """Calculate the total cost of a solution."""
        if not solution:
            return float('inf')
        
        total_cost = 0.0
        agent_map = {a.id: a for a in agents}
        task_map = {t.id: t for t in tasks}
        
        # Calculate costs for each assignment
        for agent_id, task_id in solution.items():
            agent = agent_map.get(agent_id)
            task = task_map.get(task_id)
            
            if not agent or not task:
                continue
            
            # Capability mismatch penalty
            if task.requirements:
                matched_caps = len(set(agent.capabilities) & set(task.requirements))
                capability_penalty = (len(task.requirements) - matched_caps) * 10
                total_cost += capability_penalty
            
            # Load imbalance penalty
            load_penalty = agent.current_load * 5
            total_cost += load_penalty
            
            # Success rate penalty (prefer higher success rates)
            success_penalty = (1.0 - agent.success_rate) * 8
            total_cost += success_penalty
            
            # Priority penalty (unassigned high priority tasks)
            priority_penalty = (6 - task.priority) * 2
            total_cost += priority_penalty
            
            # Deadline penalty
            if task.deadline:
                now = datetime.now()
                time_available = (task.deadline - now).total_seconds() / 3600
                if time_available < task.estimated_duration:
                    deadline_penalty = (task.estimated_duration - time_available) * 15
                    total_cost += deadline_penalty
        
        # Penalty for unassigned tasks
        assigned_tasks = set(solution.values())
        unassigned_penalty = len([t for t in tasks if t.id not in assigned_tasks]) * 20
        total_cost += unassigned_penalty
        
        return total_cost
    
    def _update_pheromones(
        self, 
        solutions: List[Dict[str, str]], 
        costs: List[float],
        agents: List[Agent],
        tasks: List[Task]
    ) -> None:
        """Update pheromone trails based on solution quality."""
        if self.pheromone_matrix is None:
            return
            
        # Evaporate existing pheromones
        self.pheromone_matrix *= (1.0 - self.rho)
        
        # Add pheromones from solutions
        agent_map = {a.id: i for i, a in enumerate(agents)}
        task_map = {t.id: i for i, t in enumerate(tasks)}
        
        for solution, cost in zip(solutions, costs):
            if not solution or cost == float('inf'):
                continue
                
            # Calculate pheromone deposit (better solutions deposit more)
            max_cost = max(costs) if costs else 1.0
            min_cost = min(costs) if costs else 0.0
            
            if max_cost > min_cost:
                quality = (max_cost - cost) / (max_cost - min_cost)
            else:
                quality = 1.0
            
            deposit = self.q * quality
            
            # Deposit pheromones for this solution
            for agent_id, task_id in solution.items():
                agent_idx = agent_map.get(agent_id)
                task_idx = task_map.get(task_id)
                
                if agent_idx is not None and task_idx is not None:
                    self.pheromone_matrix[agent_idx, task_idx] += deposit
        
        # Ensure minimum pheromone levels
        self.pheromone_matrix = np.maximum(self.pheromone_matrix, 0.01)
    
    def _check_convergence(self) -> bool:
        """Check if the algorithm has converged."""
        if len(self.iteration_history) < 10:
            return False
        
        # Check if best cost hasn't improved significantly in last iterations
        recent_costs = [h["best_cost"] for h in self.iteration_history[-10:]]
        cost_variance = np.var(recent_costs)
        
        return bool(cost_variance < self.convergence_threshold)
    
    def _calculate_assignment_cost(
        self, 
        assignments: Dict[str, str], 
        agents: List[Agent], 
        tasks: List[Task]
    ) -> float:
        """Calculate total cost for the final assignment dictionary."""
        return self._calculate_solution_cost(assignments, agents, tasks)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the optimization process."""
        if not self.iteration_history:
            return {"status": "not_run"}
        
        costs = [h["best_cost"] for h in self.iteration_history]
        durations = [h["duration"] for h in self.iteration_history]
        
        return {
            "total_iterations": len(self.iteration_history),
            "convergence_iteration": self.convergence_iteration,
            "best_cost": self.best_cost,
            "initial_cost": costs[0] if costs else None,
            "final_cost": costs[-1] if costs else None,
            "improvement": (costs[0] - costs[-1]) if len(costs) > 1 else 0,
            "average_iteration_time": np.mean(durations),
            "total_optimization_time": sum(durations),
            "cost_reduction_rate": self._calculate_improvement_rate(),
            "convergence_efficiency": self._calculate_convergence_efficiency()
        }
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate the rate of cost improvement over iterations."""
        if len(self.iteration_history) < 2:
            return 0.0
        
        costs = [h["best_cost"] for h in self.iteration_history]
        initial_cost = costs[0]
        final_cost = costs[-1]
        iterations = len(costs)
        
        if initial_cost > final_cost:
            return (initial_cost - final_cost) / (initial_cost * iterations)
        return 0.0
    
    def _calculate_convergence_efficiency(self) -> float:
        """Calculate how efficiently the algorithm converged."""
        if not self.convergence_iteration:
            return 0.0
        
        return 1.0 - (self.convergence_iteration / self.max_iterations)