"""
Load Balancing Engine

This module implements real-time load balancing for agent assignment with capacity prediction,
load monitoring, and performance optimization to maintain agent availability and performance.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import asyncio
import logging
from datetime import datetime, timedelta
import statistics
from enum import Enum

logger = logging.getLogger(__name__)


class LoadStatus(Enum):
    """Agent load status levels."""
    IDLE = "idle"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    OVERLOADED = "overloaded"


@dataclass
class AgentLoad:
    """Represents current load information for an agent."""
    agent_id: str
    current_tasks: int
    max_capacity: int
    cpu_usage: float  # 0.0 to 1.0
    memory_usage: float  # 0.0 to 1.0
    response_time: float  # Average response time in seconds
    last_updated: datetime
    active_since: datetime = field(default_factory=datetime.now)
    completed_tasks_today: int = 0
    failed_tasks_today: int = 0
    average_task_duration: float = 0.0
    
    @property
    def task_load_ratio(self) -> float:
        """Calculate task load as ratio of current to max capacity."""
        return self.current_tasks / self.max_capacity if self.max_capacity > 0 else 1.0
    
    @property
    def system_load_ratio(self) -> float:
        """Calculate system load as average of CPU and memory usage."""
        return (self.cpu_usage + self.memory_usage) / 2.0
    
    @property
    def overall_load(self) -> float:
        """Calculate overall load combining task and system loads."""
        return max(self.task_load_ratio, self.system_load_ratio)
    
    @property
    def load_status(self) -> LoadStatus:
        """Determine current load status."""
        load = self.overall_load
        if load < 0.1:
            return LoadStatus.IDLE
        elif load < 0.4:
            return LoadStatus.LOW
        elif load < 0.7:
            return LoadStatus.MEDIUM
        elif load < 0.9:
            return LoadStatus.HIGH
        else:
            return LoadStatus.OVERLOADED
    
    @property
    def success_rate_today(self) -> float:
        """Calculate success rate for tasks completed today."""
        total_today = self.completed_tasks_today + self.failed_tasks_today
        return self.completed_tasks_today / total_today if total_today > 0 else 1.0


class LoadBalancer:
    """
    Real-time load balancing for agent assignment.
    
    Provides load monitoring, capacity prediction, and optimization algorithms
    to maintain optimal agent performance and availability.
    """
    
    def __init__(
        self, 
        update_interval: float = 10.0,
        load_history_size: int = 100,
        prediction_window: int = 300  # seconds
    ):
        """
        Initialize load balancer.
        
        Args:
            update_interval: Seconds between load updates
            load_history_size: Number of historical load samples to keep
            prediction_window: Time window for load prediction in seconds
        """
        self.update_interval = update_interval
        self.load_history_size = load_history_size
        self.prediction_window = prediction_window
        
        # Current agent loads
        self.agent_loads: Dict[str, AgentLoad] = {}
        
        # Historical load data for predictions
        self.load_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Monitoring control
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Performance metrics
        self.balancing_metrics = {
            "assignments_made": 0,
            "load_predictions": 0,
            "capacity_warnings": 0,
            "rebalancing_events": 0
        }
    
    async def start_monitoring(self) -> None:
        """Start real-time load monitoring."""
        if self._running:
            logger.warning("Load monitoring already running")
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_loads())
        logger.info("Load monitoring started with %.1f second intervals", self.update_interval)
    
    async def stop_monitoring(self) -> None:
        """Stop load monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Load monitoring stopped")
    
    async def update_agent_loads(self, agents: List[Any]) -> None:
        """Update load information for a list of agents."""
        for agent in agents:
            if hasattr(agent, 'id'):
                await self._update_single_agent_load(agent)
    
    async def _update_single_agent_load(self, agent: Any) -> None:
        """Update load information for a single agent."""
        try:
            # Get or create agent load entry
            if agent.id not in self.agent_loads:
                self.agent_loads[agent.id] = AgentLoad(
                    agent_id=agent.id,
                    current_tasks=getattr(agent, 'current_load', 0) * 10,  # Convert load ratio to task count estimate
                    max_capacity=10,  # Default capacity
                    cpu_usage=getattr(agent, 'current_load', 0.0),
                    memory_usage=getattr(agent, 'current_load', 0.0),
                    response_time=getattr(agent, 'average_completion_time', 1.0),
                    last_updated=datetime.now()
                )
            else:
                # Update existing load info
                load_info = self.agent_loads[agent.id]
                load_info.current_tasks = int(getattr(agent, 'current_load', 0) * load_info.max_capacity)
                load_info.cpu_usage = getattr(agent, 'current_load', 0.0)
                load_info.memory_usage = getattr(agent, 'current_load', 0.0)
                load_info.response_time = getattr(agent, 'average_completion_time', 1.0)
                load_info.last_updated = datetime.now()
            
            # Update load history
            await self._update_load_history(agent.id, self.agent_loads[agent.id].overall_load)
            
        except Exception as e:
            logger.error("Error updating agent load for %s: %s", agent.id, str(e))
    
    async def _update_load_history(self, agent_id: str, load_value: float) -> None:
        """Update load history for an agent."""
        if agent_id not in self.load_history:
            self.load_history[agent_id] = []
        
        self.load_history[agent_id].append((datetime.now(), load_value))
        
        # Trim history to maximum size
        if len(self.load_history[agent_id]) > self.load_history_size:
            self.load_history[agent_id] = self.load_history[agent_id][-self.load_history_size:]
    
    async def get_least_loaded_agents(self, count: int = 5, exclude_overloaded: bool = True) -> List[str]:
        """
        Get least loaded agents for assignment.
        
        Args:
            count: Number of agents to return
            exclude_overloaded: Whether to exclude overloaded agents
            
        Returns:
            List of agent IDs sorted by load (ascending)
        """
        if not self.agent_loads:
            return []
        
        # Filter agents based on criteria
        eligible_agents = []
        for agent_id, load_info in self.agent_loads.items():
            if exclude_overloaded and load_info.load_status == LoadStatus.OVERLOADED:
                continue
            eligible_agents.append((agent_id, load_info.overall_load))
        
        # Sort by load and return top N
        eligible_agents.sort(key=lambda x: x[1])
        return [agent_id for agent_id, _ in eligible_agents[:count]]
    
    async def can_accept_task(
        self, 
        agent_id: str, 
        max_load_threshold: float = 0.8,
        task_complexity: float = 0.1
    ) -> bool:
        """
        Check if agent can handle additional task.
        
        Args:
            agent_id: Agent to check
            max_load_threshold: Maximum load threshold
            task_complexity: Estimated load increase from task
            
        Returns:
            True if agent can accept task, False otherwise
        """
        if agent_id not in self.agent_loads:
            return True  # Assume new agents can accept tasks
        
        load_info = self.agent_loads[agent_id]
        
        # Check current load
        if load_info.overall_load >= max_load_threshold:
            return False
        
        # Predict load after task assignment
        predicted_load = load_info.overall_load + task_complexity
        return predicted_load <= max_load_threshold
    
    async def predict_completion_time(
        self, 
        agent_id: str, 
        task_complexity: float = 0.5
    ) -> Optional[float]:
        """
        Predict task completion time for agent.
        
        Args:
            agent_id: Agent to predict for
            task_complexity: Task complexity factor (0.0 to 1.0)
            
        Returns:
            Predicted completion time in hours, or None if cannot predict
        """
        if agent_id not in self.agent_loads:
            return None
        
        load_info = self.agent_loads[agent_id]
        
        # Base completion time from agent's average
        base_time = load_info.average_task_duration or 1.0
        
        # Adjust for complexity
        complexity_factor = 1.0 + task_complexity
        
        # Adjust for current load
        load_factor = 1.0 + load_info.overall_load
        
        # Predict completion time
        predicted_time = base_time * complexity_factor * load_factor
        
        return predicted_time
    
    async def get_load_distribution_metrics(self) -> Dict[str, float]:
        """Calculate load distribution metrics across all agents."""
        if not self.agent_loads:
            return {"mean_load": 0.0, "std_deviation": 0.0, "load_balance_score": 1.0}
        
        loads = [load_info.overall_load for load_info in self.agent_loads.values()]
        
        mean_load = statistics.mean(loads)
        std_deviation = statistics.stdev(loads) if len(loads) > 1 else 0.0
        
        # Load balance score: 1.0 = perfectly balanced, 0.0 = completely unbalanced
        max_possible_std = 0.5  # Maximum std dev if half agents idle, half at 100%
        load_balance_score = max(0.0, 1.0 - (std_deviation / max_possible_std))
        
        return {
            "mean_load": mean_load,
            "std_deviation": std_deviation,
            "load_balance_score": load_balance_score,
            "agent_count": len(self.agent_loads),
            "overloaded_agents": len([l for l in self.agent_loads.values() if l.load_status == LoadStatus.OVERLOADED])
        }
    
    async def suggest_rebalancing(self) -> List[Dict[str, Any]]:
        """
        Suggest load rebalancing actions.
        
        Returns:
            List of suggested rebalancing actions
        """
        suggestions = []
        
        if len(self.agent_loads) < 2:
            return suggestions
        
        # Find overloaded and underloaded agents
        overloaded = []
        underloaded = []
        
        for agent_id, load_info in self.agent_loads.items():
            if load_info.load_status == LoadStatus.OVERLOADED:
                overloaded.append((agent_id, load_info.overall_load))
            elif load_info.load_status in [LoadStatus.IDLE, LoadStatus.LOW]:
                underloaded.append((agent_id, load_info.overall_load))
        
        # Sort by load
        overloaded.sort(key=lambda x: x[1], reverse=True)
        underloaded.sort(key=lambda x: x[1])
        
        # Generate rebalancing suggestions
        for over_agent, over_load in overloaded:
            for under_agent, under_load in underloaded:
                if over_load - under_load > 0.3:  # Significant load difference
                    suggestions.append({
                        "action": "redistribute_tasks",
                        "from_agent": over_agent,
                        "to_agent": under_agent,
                        "load_difference": over_load - under_load,
                        "priority": "high" if over_load > 0.9 else "medium"
                    })
        
        return suggestions
    
    async def _monitor_loads(self) -> None:
        """Continuously monitor agent loads."""
        while self._running:
            try:
                await self._update_all_agent_loads()
                await self._check_capacity_warnings()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in load monitoring: %s", str(e))
                await asyncio.sleep(self.update_interval)
    
    async def _update_all_agent_loads(self) -> None:
        """Update load information for all tracked agents."""
        current_time = datetime.now()
        
        # Update load history for all agents
        for agent_id, load_info in self.agent_loads.items():
            await self._update_load_history(agent_id, load_info.overall_load)
            
            # Check if load data is stale
            if current_time - load_info.last_updated > timedelta(minutes=5):
                logger.warning("Stale load data for agent %s", agent_id)
    
    async def _check_capacity_warnings(self) -> None:
        """Check for capacity warnings and log them."""
        for agent_id, load_info in self.agent_loads.items():
            if load_info.load_status == LoadStatus.OVERLOADED:
                logger.warning(
                    "Agent %s is overloaded: %.1f%% load (tasks: %d/%d, CPU: %.1f%%, memory: %.1f%%)",
                    agent_id, 
                    load_info.overall_load * 100,
                    load_info.current_tasks,
                    load_info.max_capacity,
                    load_info.cpu_usage * 100,
                    load_info.memory_usage * 100
                )
                self.balancing_metrics["capacity_warnings"] += 1
    
    def get_agent_load_info(self, agent_id: str) -> Optional[AgentLoad]:
        """Get current load information for specific agent."""
        return self.agent_loads.get(agent_id)
    
    def get_all_agent_loads(self) -> Dict[str, AgentLoad]:
        """Get current load information for all agents."""
        return self.agent_loads.copy()
    
    def get_balancing_metrics(self) -> Dict[str, int]:
        """Get load balancing performance metrics."""
        return self.balancing_metrics.copy()