"""
Swarm Intelligence Coordination Module

This module provides the main coordination engine for swarm intelligence
behaviors in the MCP Swarm Intelligence Server. It integrates all swarm
components including ACO, PSO, pheromone trails, and collective decision-making.
"""

from .aco import AntColonyOptimizer, Agent, Task
from .pso import ParticleSwarmConsensus, ConsensusOption, ConflictResolutionStrategy as PSOConflictStrategy
from .pheromones import PheromoneTrail, PheromoneEntry, TrailPattern, TrailStatistics
from .decisions import (
    CollectiveDecisionMaker, 
    DecisionType, 
    ConflictResolutionStrategy, 
    Vote, 
    RankedVote, 
    ApprovalVote,
    DecisionOption, 
    DecisionResult, 
    DecisionContext
)

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CoordinationMode(Enum):
    """Coordination modes for swarm behavior."""
    QUEEN_LED = "queen_led"
    DEMOCRATIC = "democratic"
    EXPERTISE_BASED = "expertise_based"
    HYBRID = "hybrid"
    PHEROMONE_GUIDED = "pheromone_guided"


@dataclass
class SwarmConfiguration:
    """Configuration for swarm coordination."""
    coordination_mode: CoordinationMode = CoordinationMode.HYBRID
    aco_enabled: bool = True
    pso_enabled: bool = True
    pheromone_enabled: bool = True
    collective_decisions_enabled: bool = True
    
    # ACO parameters
    aco_num_ants: int = 50
    aco_alpha: float = 1.0
    aco_beta: float = 2.0
    aco_rho: float = 0.1
    
    # PSO parameters
    pso_swarm_size: int = 30
    pso_max_iterations: int = 100
    pso_inertia: float = 0.729
    
    # Pheromone parameters
    pheromone_decay_rate: float = 0.1
    pheromone_min_intensity: float = 0.01
    pheromone_max_intensity: float = 10.0
    
    # Decision parameters
    default_decision_type: DecisionType = DecisionType.WEIGHTED_MAJORITY
    consensus_threshold: float = 0.6
    max_voting_rounds: int = 3


@dataclass
class CoordinationMetrics:
    """Metrics for swarm coordination performance."""
    total_assignments: int = 0
    successful_assignments: int = 0
    consensus_decisions: int = 0
    unanimous_decisions: int = 0
    average_confidence: float = 0.0
    average_assignment_time: float = 0.0
    pheromone_trail_count: int = 0
    active_agents: int = 0
    task_completion_rate: float = 0.0
    coordination_efficiency: float = 0.0


class SwarmCoordinator:
    """
    Main coordination engine for swarm intelligence.
    
    This class integrates all swarm intelligence components to provide
    unified coordination for the MCP Swarm Intelligence Server:
    - Agent-task assignment using ACO
    - Consensus building using PSO
    - Pheromone trail management for learning
    - Collective decision-making for complex choices
    - Performance monitoring and optimization
    """
    
    def __init__(
        self,
        config: Optional[SwarmConfiguration] = None,
        database_path: str = "data/memory.db"
    ):
        """
        Initialize swarm coordinator.
        
        Args:
            config: Swarm configuration parameters
            database_path: Path to SQLite database for persistence
        """
        self.config = config or SwarmConfiguration()
        self.database_path = database_path
        
        # Initialize components
        self.aco_optimizer: Optional[AntColonyOptimizer] = None
        self.pso_consensus: Optional[ParticleSwarmConsensus] = None
        self.pheromone_trail: Optional[PheromoneTrail] = None
        self.decision_maker: Optional[CollectiveDecisionMaker] = None
        
        # State management
        self.active_agents: Dict[str, Agent] = {}
        self.pending_tasks: Dict[str, Task] = {}
        self.active_assignments: Dict[str, str] = {}  # agent_id -> task_id
        self.coordination_metrics = CoordinationMetrics()
        
        # Performance tracking
        self.assignment_history: List[Dict[str, Any]] = []
        self.decision_history: List[DecisionResult] = []
        self.optimization_statistics: Dict[str, Any] = {}
        
        # Background tasks
        self._running = False
        self._coordination_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Initialize components based on configuration
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize swarm intelligence components."""
        if self.config.aco_enabled:
            self.aco_optimizer = AntColonyOptimizer(
                num_ants=self.config.aco_num_ants,
                alpha=self.config.aco_alpha,
                beta=self.config.aco_beta,
                rho=self.config.aco_rho
            )
        
        if self.config.pso_enabled:
            self.pso_consensus = ParticleSwarmConsensus(
                swarm_size=self.config.pso_swarm_size,
                max_iterations=self.config.pso_max_iterations,
                w=self.config.pso_inertia
            )
        
        if self.config.pheromone_enabled:
            self.pheromone_trail = PheromoneTrail(
                database_path=self.database_path,
                default_decay_rate=self.config.pheromone_decay_rate,
                min_intensity=self.config.pheromone_min_intensity,
                max_intensity=self.config.pheromone_max_intensity
            )
        
        if self.config.collective_decisions_enabled:
            self.decision_maker = CollectiveDecisionMaker(
                default_decision_type=self.config.default_decision_type,
                minimum_consensus_threshold=self.config.consensus_threshold
            )
    
    async def start(self) -> None:
        """Start the swarm coordinator."""
        if self._running:
            return
        
        self._running = True
        
        # Start components
        if self.pheromone_trail:
            await self.pheromone_trail.start()
        
        # Start background tasks
        self._coordination_task = asyncio.create_task(self._coordination_loop())
        self._metrics_task = asyncio.create_task(self._metrics_loop())
        
        logger.info("Swarm coordinator started with mode: %s", self.config.coordination_mode.value)
    
    async def stop(self) -> None:
        """Stop the swarm coordinator."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop background tasks
        if self._coordination_task:
            self._coordination_task.cancel()
            try:
                await self._coordination_task
            except asyncio.CancelledError:
                pass
        
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        
        # Stop components
        if self.pheromone_trail:
            await self.pheromone_trail.stop()
        
        logger.info("Swarm coordinator stopped")
    
    async def assign_tasks(
        self, 
        agents: List[Agent], 
        tasks: List[Task], 
        max_iterations: Optional[int] = None
    ) -> Dict[str, Optional[str]]:
        """
        Assign tasks to agents using swarm intelligence optimization.
        
        Args:
            agents: List of available agents
            tasks: List of tasks to assign
            max_iterations: Maximum ACO iterations (uses config default if None)
            
        Returns:
            Dictionary mapping task IDs to assigned agent IDs (None if unassigned)
        """
        if not agents or not tasks:
            return {task.id: None for task in tasks}
        
        try:
            # Use ACO for optimization if available
            if self.aco_optimizer:
                original_max_iter = None
                if max_iterations:
                    original_max_iter = self.aco_optimizer.max_iterations
                    self.aco_optimizer.max_iterations = max_iterations
                
                assignments, cost, metadata = await self.aco_optimizer.find_optimal_assignment(agents, tasks)
                
                if max_iterations and original_max_iter is not None:
                    self.aco_optimizer.max_iterations = original_max_iter
                
                # Convert assignments to task_id -> agent_id mapping with proper nulls
                task_assignments = {}
                
                # Ensure all tasks have entries (even if unassigned)
                for task in tasks:
                    task_assignments[task.id] = assignments.get(task.id, None)
                
                # Update internal state
                for task_id, agent_id in assignments.items():
                    if agent_id:
                        self.active_assignments[agent_id] = task_id
                
                # Update metrics
                self.coordination_metrics.total_assignments += len([a for a in assignments.values() if a])
                
                return task_assignments
            
            else:
                # Fallback to simple round-robin assignment
                assignments = {}
                for i, task in enumerate(tasks):
                    if i < len(agents):
                        assignments[task.id] = agents[i % len(agents)].id
                    else:
                        assignments[task.id] = None
                
                return assignments
                
        except Exception as e:
            logger.error("Error in task assignment: %s", str(e))
            return {task.id: None for task in tasks}
    
    async def _coordination_loop(self) -> None:
        """Main coordination loop for ongoing swarm management."""
        while self._running:
            try:
                # Periodic coordination tasks
                await self._update_pheromone_trails()
                await self._process_pending_assignments()
                await self._monitor_agent_performance()
                
                # Sleep between coordination cycles
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except (RuntimeError, ValueError) as e:
                logger.error("Error in coordination loop: %s", e)
                await asyncio.sleep(5.0)  # Wait longer on error
    
    async def _metrics_loop(self) -> None:
        """Background loop for metrics collection and analysis."""
        while self._running:
            try:
                # Update metrics periodically
                await self._update_coordination_metrics()
                await self._analyze_performance_trends()
                
                # Sleep between metrics updates
                await asyncio.sleep(10.0)
                
            except asyncio.CancelledError:
                break
            except (RuntimeError, ValueError, TypeError) as e:
                logger.error("Error in metrics loop: %s", e)
                await asyncio.sleep(30.0)  # Wait longer on error
    
    async def _update_pheromone_trails(self) -> None:
        """Update pheromone trails based on current assignments."""
        if not self.pheromone_trail:
            return
        
        # Update trails based on successful assignments
        for agent_id, task_id in self.active_assignments.items():
            if agent_id in self.active_agents and task_id in self.pending_tasks:
                agent = self.active_agents[agent_id]
                task = self.pending_tasks[task_id]
                
                # Deposit pheromone for successful coordination
                await self.pheromone_trail.deposit_pheromone(
                    source_id=agent_id,
                    target_id=task_id,
                    trail_type="assignment",
                    intensity=1.0,
                    success=True,
                    metadata={"agent_capabilities": agent.capabilities, "task_requirements": task.requirements}
                )
    
    async def _process_pending_assignments(self) -> None:
        """Process pending task assignments using swarm algorithms."""
        if not self.pending_tasks or not self.active_agents:
            return
        
        # Use ACO for optimal assignment if enabled
        if self.aco_optimizer and self.config.aco_enabled:
            try:
                agents = list(self.active_agents.values())
                tasks = list(self.pending_tasks.values())
                
                if agents and tasks:
                    assignments, cost, metadata = await self.aco_optimizer.find_optimal_assignment(agents, tasks)
                    
                    # Log optimization results
                    logger.debug("ACO optimization completed with cost: %s, metadata: %s", cost, metadata)
                    
                    # Process assignments
                    for agent_id, task_id in assignments.items():
                        if agent_id not in self.active_assignments:
                            self.active_assignments[agent_id] = task_id
                            
                            # Remove assigned task from pending
                            if task_id in self.pending_tasks:
                                del self.pending_tasks[task_id]
                            
                            self.coordination_metrics.total_assignments += 1
                            
            except (RuntimeError, ValueError) as e:
                logger.error("Error in ACO assignment: %s", e)
    
    async def _monitor_agent_performance(self) -> None:
        """Monitor and analyze agent performance."""
        self.coordination_metrics.active_agents = len(self.active_agents)
        
        # Calculate completion rate
        if self.coordination_metrics.total_assignments > 0:
            self.coordination_metrics.task_completion_rate = (
                self.coordination_metrics.successful_assignments / 
                self.coordination_metrics.total_assignments
            )
    
    async def _update_coordination_metrics(self) -> None:
        """Update coordination performance metrics."""
        if self.pheromone_trail:
            stats = await self.pheromone_trail.get_statistics()
            self.coordination_metrics.pheromone_trail_count = stats.total_trails
        
        # Update coordination efficiency based on recent performance
        recent_decisions = self.decision_history[-10:] if self.decision_history else []
        if recent_decisions:
            avg_confidence = sum(d.confidence_score for d in recent_decisions) / len(recent_decisions)
            self.coordination_metrics.average_confidence = avg_confidence
            
            # Calculate efficiency as combination of confidence and completion rate
            self.coordination_metrics.coordination_efficiency = (
                (self.coordination_metrics.average_confidence * 0.6) +
                (self.coordination_metrics.task_completion_rate * 0.4)
            )
    
    async def _analyze_performance_trends(self) -> None:
        """Analyze performance trends and adjust parameters."""
        # This could include adaptive parameter tuning based on performance
        # For now, just log current performance
        if len(self.assignment_history) > 0:
            recent_assignments = self.assignment_history[-10:]
            avg_time = sum(a.get("duration", 0) for a in recent_assignments) / len(recent_assignments)
            self.coordination_metrics.average_assignment_time = avg_time
    
    async def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get comprehensive coordination statistics."""
        stats = {
            "configuration": {
                "mode": self.config.coordination_mode.value,
                "aco_enabled": self.config.aco_enabled,
                "pso_enabled": self.config.pso_enabled,
                "pheromone_enabled": self.config.pheromone_enabled,
                "decisions_enabled": self.config.collective_decisions_enabled
            },
            "metrics": {
                "active_agents": self.coordination_metrics.active_agents,
                "pending_tasks": len(self.pending_tasks),
                "active_assignments": len(self.active_assignments),
                "total_assignments": self.coordination_metrics.total_assignments,
                "successful_assignments": self.coordination_metrics.successful_assignments,
                "task_completion_rate": self.coordination_metrics.task_completion_rate,
                "coordination_efficiency": self.coordination_metrics.coordination_efficiency,
                "consensus_decisions": self.coordination_metrics.consensus_decisions,
                "unanimous_decisions": self.coordination_metrics.unanimous_decisions,
                "average_confidence": self.coordination_metrics.average_confidence,
                "pheromone_trail_count": self.coordination_metrics.pheromone_trail_count
            }
        }
        
        return stats


# Export main classes
__all__ = [
    "SwarmCoordinator",
    "SwarmConfiguration", 
    "CoordinationMode",
    "CoordinationMetrics",
    "AntColonyOptimizer",
    "ParticleSwarmConsensus",
    "PheromoneTrail",
    "CollectiveDecisionMaker",
    "Agent",
    "Task",
    "ConsensusOption",
    "DecisionOption",
    "Vote",
    "RankedVote", 
    "ApprovalVote",
    "DecisionResult",
    "DecisionContext",
    "DecisionType",
    "ConflictResolutionStrategy"
]

__all__ = ["SwarmCoordinator"]