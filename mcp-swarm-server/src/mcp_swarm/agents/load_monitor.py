"""
Agent Load Monitor for MCP Swarm Intelligence Server

This module provides real-time monitoring of agent load, task queue status,
and resource utilization to prevent overutilization and ensure optimal
performance across the swarm ecosystem.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentLoadMetrics:
    """Metrics for a single agent's load"""
    agent_id: str
    cpu_percent: float
    memory_percent: float
    active_tasks: int
    queued_tasks: int
    avg_response_time: float
    last_updated: datetime
    status: str  # 'normal', 'warning', 'critical', 'offline'

@dataclass
class SystemLoadMetrics:
    """System-wide load metrics"""
    total_cpu_percent: float
    total_memory_percent: float
    total_active_tasks: int
    total_queued_tasks: int
    agent_count: int
    bottleneck_agents: List[str]
    last_updated: datetime

class AgentLoadMonitor:
    """Real-time agent load monitoring system"""
    
    def __init__(self, 
                 warning_cpu_threshold: float = 70.0,
                 critical_cpu_threshold: float = 90.0,
                 warning_memory_threshold: float = 80.0,
                 critical_memory_threshold: float = 95.0,
                 max_queue_size: int = 100,
                 monitoring_interval: float = 5.0):
        """
        Initialize the agent load monitor
        
        Args:
            warning_cpu_threshold: CPU usage percentage to trigger warnings
            critical_cpu_threshold: CPU usage percentage to trigger critical alerts
            warning_memory_threshold: Memory usage percentage to trigger warnings
            critical_memory_threshold: Memory usage percentage to trigger critical alerts
            max_queue_size: Maximum queue size before triggering alerts
            monitoring_interval: Interval between monitoring cycles in seconds
        """
        self.warning_cpu_threshold = warning_cpu_threshold
        self.critical_cpu_threshold = critical_cpu_threshold
        self.warning_memory_threshold = warning_memory_threshold
        self.critical_memory_threshold = critical_memory_threshold
        self.max_queue_size = max_queue_size
        self.monitoring_interval = monitoring_interval
        
        # Agent tracking
        self.agent_metrics: Dict[str, AgentLoadMetrics] = {}
        self.agent_task_queues: Dict[str, List[Any]] = defaultdict(list)
        self.agent_response_times: Dict[str, List[float]] = defaultdict(list)
        self.agent_last_heartbeat: Dict[str, datetime] = {}
        
        # System metrics
        self.system_metrics: Optional[SystemLoadMetrics] = None
        
        # Monitoring control
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Load balancing thresholds
        self.load_balancing_enabled = True
        self.rebalance_threshold = 0.3  # 30% load difference triggers rebalancing

    async def start_monitoring(self):
        """Start the continuous monitoring process"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
            
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Agent load monitoring started")

    async def stop_monitoring(self):
        """Stop the monitoring process"""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Agent load monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await self._collect_metrics()
                await self._update_system_metrics()
                await self._check_thresholds()
                await asyncio.sleep(self.monitoring_interval)
            except (asyncio.CancelledError, OSError) as e:
                logger.error("Error in monitoring loop: %s", e)
                await asyncio.sleep(self.monitoring_interval)

    async def _collect_metrics(self):
        """Collect metrics for all registered agents"""
        current_time = datetime.now()
        
        for agent_id in list(self.agent_metrics.keys()):
            try:
                # Check if agent is still responsive
                last_heartbeat = self.agent_last_heartbeat.get(agent_id)
                if last_heartbeat and (current_time - last_heartbeat).total_seconds() > 60:
                    # Agent hasn't responded in 60 seconds, mark as offline
                    self.agent_metrics[agent_id].status = 'offline'
                    continue
                
                # Get current metrics
                cpu_percent = await self._get_agent_cpu_usage(agent_id)
                memory_percent = await self._get_agent_memory_usage(agent_id)
                active_tasks = await self._get_active_task_count(agent_id)
                queued_tasks = len(self.agent_task_queues[agent_id])
                avg_response_time = self._calculate_avg_response_time(agent_id)
                
                # Determine status
                status = self._determine_agent_status(cpu_percent, memory_percent, queued_tasks)
                
                # Update metrics
                self.agent_metrics[agent_id] = AgentLoadMetrics(
                    agent_id=agent_id,
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    active_tasks=active_tasks,
                    queued_tasks=queued_tasks,
                    avg_response_time=avg_response_time,
                    last_updated=current_time,
                    status=status
                )
                
            except (ValueError, KeyError) as e:
                logger.error("Error collecting metrics for agent %s: %s", agent_id, e)

    async def _get_agent_cpu_usage(self, agent_id: str) -> float:
        """Get CPU usage for a specific agent"""
        # In a real implementation, this would track per-agent CPU usage
        # For now, we'll simulate based on task load
        active_tasks = await self._get_active_task_count(agent_id)
        queued_tasks = len(self.agent_task_queues[agent_id])
        
        # Simulate CPU usage based on task load
        base_cpu = min(active_tasks * 10, 80)  # Each task adds ~10% CPU
        queue_penalty = min(queued_tasks * 2, 15)  # Queued tasks add pressure
        
        return min(base_cpu + queue_penalty, 100.0)

    async def _get_agent_memory_usage(self, agent_id: str) -> float:
        """Get memory usage for a specific agent"""
        # Simulate memory usage based on agent activity
        active_tasks = await self._get_active_task_count(agent_id)
        queue_size = len(self.agent_task_queues[agent_id])
        
        # Each active task uses ~5% memory, queued tasks use ~1%
        memory_usage = (active_tasks * 5) + (queue_size * 1)
        return min(memory_usage, 95.0)

    async def _get_active_task_count(self, agent_id: str) -> int:
        """Get number of active tasks for an agent"""
        # This would integrate with the actual task management system
        # For now, simulate based on queue size and processing capacity
        queue_size = len(self.agent_task_queues[agent_id])
        
        # Assume agents can handle up to 5 concurrent tasks
        return min(queue_size, 5)

    def _calculate_avg_response_time(self, agent_id: str) -> float:
        """Calculate average response time for an agent"""
        response_times = self.agent_response_times[agent_id]
        if not response_times:
            return 0.0
        
        # Keep only recent response times (last 100)
        recent_times = response_times[-100:]
        return sum(recent_times) / len(recent_times)

    def _determine_agent_status(self, cpu_percent: float, memory_percent: float, queue_size: int) -> str:
        """Determine agent status based on metrics"""
        if cpu_percent >= self.critical_cpu_threshold or memory_percent >= self.critical_memory_threshold:
            return 'critical'
        elif cpu_percent >= self.warning_cpu_threshold or memory_percent >= self.warning_memory_threshold:
            return 'warning'
        elif queue_size > self.max_queue_size:
            return 'warning'
        else:
            return 'normal'

    async def _update_system_metrics(self):
        """Update system-wide metrics"""
        if not self.agent_metrics:
            return
        
        total_cpu = sum(metrics.cpu_percent for metrics in self.agent_metrics.values())
        total_memory = sum(metrics.memory_percent for metrics in self.agent_metrics.values())
        total_active = sum(metrics.active_tasks for metrics in self.agent_metrics.values())
        total_queued = sum(metrics.queued_tasks for metrics in self.agent_metrics.values())
        
        # Find bottleneck agents (high load relative to others)
        avg_cpu = total_cpu / len(self.agent_metrics)
        bottlenecks = [
            agent_id for agent_id, metrics in self.agent_metrics.items()
            if metrics.cpu_percent > avg_cpu * 1.5 and metrics.status in ['warning', 'critical']
        ]
        
        self.system_metrics = SystemLoadMetrics(
            total_cpu_percent=total_cpu / len(self.agent_metrics),
            total_memory_percent=total_memory / len(self.agent_metrics),
            total_active_tasks=total_active,
            total_queued_tasks=total_queued,
            agent_count=len(self.agent_metrics),
            bottleneck_agents=bottlenecks,
            last_updated=datetime.now()
        )

    async def _check_thresholds(self):
        """Check thresholds and trigger alerts if needed"""
        for agent_id, metrics in self.agent_metrics.items():
            if metrics.status == 'critical':
                await self._trigger_critical_alert(agent_id, metrics)
            elif metrics.status == 'warning':
                await self._trigger_warning_alert(agent_id, metrics)

    async def _trigger_critical_alert(self, agent_id: str, metrics: AgentLoadMetrics):
        """Trigger critical alert for an agent"""
        logger.critical("CRITICAL: Agent %s - CPU: %.1f%%, Memory: %.1f%%, Queue: %d", 
                        agent_id, metrics.cpu_percent, metrics.memory_percent, metrics.queued_tasks)
        
        # In a real implementation, this would integrate with alerting systems
        # For now, we'll implement load shedding
        if self.load_balancing_enabled:
            await self._initiate_load_shedding(agent_id)

    async def _trigger_warning_alert(self, agent_id: str, metrics: AgentLoadMetrics):
        """Trigger warning alert for an agent"""
        logger.warning("WARNING: Agent %s - CPU: %.1f%%, Memory: %.1f%%, Queue: %d",
                      agent_id, metrics.cpu_percent, metrics.memory_percent, metrics.queued_tasks)

    async def _initiate_load_shedding(self, agent_id: str):
        """Initiate load shedding for an overloaded agent"""
        logger.info("Initiating load shedding for agent %s", agent_id)
        
        # Move some queued tasks to other agents
        if agent_id in self.agent_task_queues:
            tasks_to_move = self.agent_task_queues[agent_id][::2]  # Move every other task
            self.agent_task_queues[agent_id] = self.agent_task_queues[agent_id][1::2]
            
            # Redistribute to least loaded agents
            target_agents = self._get_least_loaded_agents(exclude=[agent_id])
            if target_agents:
                tasks_per_agent = len(tasks_to_move) // len(target_agents)
                for i, target_agent in enumerate(target_agents):
                    start_idx = i * tasks_per_agent
                    end_idx = start_idx + tasks_per_agent if i < len(target_agents) - 1 else None
                    self.agent_task_queues[target_agent].extend(tasks_to_move[start_idx:end_idx])

    def _get_least_loaded_agents(self, exclude: Optional[List[str]] = None, limit: int = 3) -> List[str]:
        """Get the least loaded agents for load balancing"""
        exclude = exclude or []
        
        available_agents = [
            (agent_id, metrics) for agent_id, metrics in self.agent_metrics.items()
            if agent_id not in exclude and metrics.status in ['normal', 'warning']
        ]
        
        # Sort by combined load (CPU + memory + queue pressure)
        available_agents.sort(key=lambda x: (
            x[1].cpu_percent + x[1].memory_percent + (x[1].queued_tasks * 2)
        ))
        
        return [agent_id for agent_id, _ in available_agents[:limit]]

    # Public interface methods
    
    def register_agent(self, agent_id: str):
        """Register a new agent for monitoring"""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentLoadMetrics(
                agent_id=agent_id,
                cpu_percent=0.0,
                memory_percent=0.0,
                active_tasks=0,
                queued_tasks=0,
                avg_response_time=0.0,
                last_updated=datetime.now(),
                status='normal'
            )
            self.agent_last_heartbeat[agent_id] = datetime.now()
            logger.info("Registered agent %s for monitoring", agent_id)

    def update_agent_heartbeat(self, agent_id: str):
        """Update agent heartbeat timestamp"""
        self.agent_last_heartbeat[agent_id] = datetime.now()
        if agent_id in self.agent_metrics:
            self.agent_metrics[agent_id].status = 'normal'

    def add_task_to_queue(self, agent_id: str, task: Any):
        """Add a task to an agent's queue"""
        if agent_id in self.agent_metrics:
            self.agent_task_queues[agent_id].append(task)

    def record_response_time(self, agent_id: str, response_time: float):
        """Record response time for an agent"""
        if agent_id in self.agent_metrics:
            self.agent_response_times[agent_id].append(response_time)
            # Keep only recent response times
            if len(self.agent_response_times[agent_id]) > 1000:
                self.agent_response_times[agent_id] = self.agent_response_times[agent_id][-100:]

    def get_agent_metrics(self, agent_id: str) -> Optional[AgentLoadMetrics]:
        """Get metrics for a specific agent"""
        return self.agent_metrics.get(agent_id)

    def get_system_metrics(self) -> Optional[SystemLoadMetrics]:
        """Get system-wide metrics"""
        return self.system_metrics

    def get_all_agent_metrics(self) -> Dict[str, AgentLoadMetrics]:
        """Get metrics for all agents"""
        return self.agent_metrics.copy()

    def get_load_distribution_recommendation(self) -> Dict[str, Any]:
        """Get load distribution recommendations"""
        if not self.agent_metrics:
            return {"recommendation": "no_agents"}
        
        # Calculate load imbalance
        cpu_loads = [metrics.cpu_percent for metrics in self.agent_metrics.values()]
        memory_loads = [metrics.memory_percent for metrics in self.agent_metrics.values()]
        
        cpu_std = (sum((x - (sum(cpu_loads) / len(cpu_loads))) ** 2 for x in cpu_loads) / len(cpu_loads)) ** 0.5
        memory_std = (sum((x - (sum(memory_loads) / len(memory_loads))) ** 2 for x in memory_loads) / len(memory_loads)) ** 0.5
        
        # Determine if rebalancing is needed
        if cpu_std > 20 or memory_std > 20:  # High variance in load
            overloaded = [aid for aid, m in self.agent_metrics.items() if m.status in ['warning', 'critical']]
            underloaded = self._get_least_loaded_agents(limit=5)
            
            return {
                "recommendation": "rebalance",
                "overloaded_agents": overloaded,
                "underloaded_agents": underloaded,
                "cpu_variance": cpu_std,
                "memory_variance": memory_std
            }
        
        return {
            "recommendation": "balanced",
            "cpu_variance": cpu_std,
            "memory_variance": memory_std
        }