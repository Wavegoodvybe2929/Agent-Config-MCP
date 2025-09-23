"""
Agent Availability Tracker for MCP Swarm Intelligence Server

This module provides real-time tracking of agent availability, heartbeat monitoring,
and failure detection to ensure reliable agent coordination and rapid response
to availability changes.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent availability status"""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    RECONNECTING = "reconnecting"
    MAINTENANCE = "maintenance"

@dataclass
class HeartbeatInfo:
    """Heartbeat information for an agent"""
    agent_id: str
    last_heartbeat: datetime
    consecutive_misses: int
    average_interval: float
    status: AgentStatus
    response_time: float
    metadata: Dict[str, Any]

@dataclass
class AvailabilityMetrics:
    """Availability metrics for an agent"""
    agent_id: str
    uptime_percentage: float
    total_downtime: timedelta
    availability_status: AgentStatus
    last_status_change: datetime
    failure_count: int
    recovery_time: Optional[float]
    heartbeat_reliability: float
    failure_start_time: Optional[datetime] = None

class AgentAvailabilityTracker:
    """Real-time agent availability tracking system"""
    
    def __init__(self, 
                 heartbeat_interval: float = 10.0,
                 failure_threshold: int = 3,
                 degraded_threshold: int = 2,
                 timeout_seconds: float = 30.0,
                 cleanup_interval: float = 300.0):
        """
        Initialize the availability tracker
        
        Args:
            heartbeat_interval: Expected interval between heartbeats in seconds
            failure_threshold: Number of missed heartbeats before marking as offline
            degraded_threshold: Number of missed heartbeats before marking as degraded
            timeout_seconds: Timeout for considering an agent unresponsive
            cleanup_interval: Interval for cleaning up old tracking data
        """
        self.heartbeat_interval = heartbeat_interval
        self.failure_threshold = failure_threshold
        self.degraded_threshold = degraded_threshold
        self.timeout_seconds = timeout_seconds
        self.cleanup_interval = cleanup_interval
        
        # Tracking data
        self.heartbeats: Dict[str, HeartbeatInfo] = {}
        self.availability_metrics: Dict[str, AvailabilityMetrics] = {}
        self.status_history: Dict[str, List[tuple]] = {}  # (timestamp, status, reason)
        
        # Event callbacks
        self.status_change_callbacks: List[Callable] = []
        self.failure_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        # Monitoring control
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.heartbeat_intervals: Dict[str, List[float]] = {}
        self.response_times: Dict[str, List[float]] = {}

    async def start_monitoring(self):
        """Start the availability monitoring process"""
        if self.is_monitoring:
            logger.warning("Availability monitoring already started")
            return
            
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Agent availability monitoring started")

    async def stop_monitoring(self):
        """Stop the monitoring process"""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        
        # Cancel monitoring tasks
        for task in [self.monitor_task, self.cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        logger.info("Agent availability monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await self._check_heartbeats()
                await self._update_availability_metrics()
                await asyncio.sleep(self.heartbeat_interval / 2)  # Check twice per interval
            except (asyncio.CancelledError, OSError) as e:
                logger.error("Error in availability monitoring loop: %s", e)
                await asyncio.sleep(self.heartbeat_interval)

    async def _cleanup_loop(self):
        """Periodic cleanup of old tracking data"""
        while self.is_monitoring:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(self.cleanup_interval)
            except (asyncio.CancelledError, OSError) as e:
                logger.error("Error in cleanup loop: %s", e)
                await asyncio.sleep(self.cleanup_interval)

    async def _check_heartbeats(self):
        """Check heartbeat status for all tracked agents"""
        current_time = datetime.now()
        
        for agent_id, heartbeat_info in list(self.heartbeats.items()):
            time_since_last = (current_time - heartbeat_info.last_heartbeat).total_seconds()
            
            # Determine new status based on time elapsed
            new_status = self._determine_status(time_since_last, heartbeat_info.consecutive_misses)
            
            # Update consecutive misses if needed
            if time_since_last > self.heartbeat_interval:
                heartbeat_info.consecutive_misses += 1
            
            # Check for status change
            if new_status != heartbeat_info.status:
                await self._handle_status_change(agent_id, heartbeat_info.status, new_status, current_time)
                heartbeat_info.status = new_status
                
                # Record status change
                self._record_status_change(agent_id, new_status, f"Heartbeat check: {time_since_last:.1f}s since last")

    def _determine_status(self, time_since_last: float, consecutive_misses: int) -> AgentStatus:
        """Determine agent status based on heartbeat timing"""
        if time_since_last > self.timeout_seconds or consecutive_misses >= self.failure_threshold:
            return AgentStatus.OFFLINE
        elif consecutive_misses >= self.degraded_threshold:
            return AgentStatus.DEGRADED
        else:
            return AgentStatus.ONLINE

    async def _handle_status_change(self, agent_id: str, old_status: AgentStatus, new_status: AgentStatus, timestamp: datetime):
        """Handle agent status changes and trigger callbacks"""
        logger.info("Agent %s status changed from %s to %s", agent_id, old_status.value, new_status.value)
        
        # Update availability metrics
        if agent_id in self.availability_metrics:
            metrics = self.availability_metrics[agent_id]
            metrics.last_status_change = timestamp
            
            # Update failure count
            if new_status == AgentStatus.OFFLINE:
                metrics.failure_count += 1
            
            # Calculate recovery time
            if old_status == AgentStatus.OFFLINE and new_status in [AgentStatus.ONLINE, AgentStatus.DEGRADED]:
                if metrics.failure_start_time:
                    recovery_time = (timestamp - metrics.failure_start_time).total_seconds()
                    metrics.recovery_time = recovery_time
                    metrics.failure_start_time = None
            elif new_status == AgentStatus.OFFLINE:
                metrics.failure_start_time = timestamp
        
        # Trigger appropriate callbacks
        if new_status == AgentStatus.OFFLINE and old_status != AgentStatus.OFFLINE:
            await self._trigger_failure_callbacks(agent_id, old_status, new_status)
        elif old_status == AgentStatus.OFFLINE and new_status != AgentStatus.OFFLINE:
            await self._trigger_recovery_callbacks(agent_id, old_status, new_status)
        
        # Always trigger status change callbacks
        await self._trigger_status_change_callbacks(agent_id, old_status, new_status)

    async def _trigger_status_change_callbacks(self, agent_id: str, old_status: AgentStatus, new_status: AgentStatus):
        """Trigger status change callbacks"""
        for callback in self.status_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_id, old_status, new_status)
                else:
                    callback(agent_id, old_status, new_status)
            except (ValueError, TypeError) as e:
                logger.error("Error in status change callback: %s", e)

    async def _trigger_failure_callbacks(self, agent_id: str, old_status: AgentStatus, new_status: AgentStatus):
        """Trigger failure callbacks"""
        for callback in self.failure_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_id, old_status, new_status)
                else:
                    callback(agent_id, old_status, new_status)
            except (ValueError, TypeError) as e:
                logger.error("Error in failure callback: %s", e)

    async def _trigger_recovery_callbacks(self, agent_id: str, old_status: AgentStatus, new_status: AgentStatus):
        """Trigger recovery callbacks"""
        for callback in self.recovery_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_id, old_status, new_status)
                else:
                    callback(agent_id, old_status, new_status)
            except (ValueError, TypeError) as e:
                logger.error("Error in recovery callback: %s", e)

    def _record_status_change(self, agent_id: str, status: AgentStatus, reason: str):
        """Record status change in history"""
        if agent_id not in self.status_history:
            self.status_history[agent_id] = []
        
        self.status_history[agent_id].append((datetime.now(), status, reason))
        
        # Keep only recent history (last 100 changes)
        if len(self.status_history[agent_id]) > 100:
            self.status_history[agent_id] = self.status_history[agent_id][-50:]

    async def _update_availability_metrics(self):
        """Update availability metrics for all agents"""
        current_time = datetime.now()
        
        for agent_id, heartbeat_info in self.heartbeats.items():
            if agent_id not in self.availability_metrics:
                self.availability_metrics[agent_id] = AvailabilityMetrics(
                    agent_id=agent_id,
                    uptime_percentage=100.0,
                    total_downtime=timedelta(),
                    availability_status=heartbeat_info.status,
                    last_status_change=current_time,
                    failure_count=0,
                    recovery_time=None,
                    heartbeat_reliability=100.0
                )
            
            metrics = self.availability_metrics[agent_id]
            
            # Calculate uptime percentage
            total_tracking_time = (current_time - heartbeat_info.last_heartbeat).total_seconds()
            if total_tracking_time > 0:
                # Get downtime from status history
                downtime = self._calculate_downtime(agent_id, current_time)
                uptime_percentage = max(0.0, (total_tracking_time - downtime.total_seconds()) / total_tracking_time * 100)
                metrics.uptime_percentage = uptime_percentage
                metrics.total_downtime = downtime
            
            # Calculate heartbeat reliability
            reliability = self._calculate_heartbeat_reliability(agent_id)
            metrics.heartbeat_reliability = reliability
            
            # Update current status
            metrics.availability_status = heartbeat_info.status

    def _calculate_downtime(self, agent_id: str, current_time: datetime) -> timedelta:
        """Calculate total downtime for an agent"""
        if agent_id not in self.status_history:
            return timedelta()
        
        total_downtime = timedelta()
        downtime_start = None
        
        for timestamp, status, _ in self.status_history[agent_id]:
            if status == AgentStatus.OFFLINE and downtime_start is None:
                downtime_start = timestamp
            elif status != AgentStatus.OFFLINE and downtime_start is not None:
                total_downtime += timestamp - downtime_start
                downtime_start = None
        
        # If currently down, add current downtime
        if downtime_start is not None:
            total_downtime += current_time - downtime_start
        
        return total_downtime

    def _calculate_heartbeat_reliability(self, agent_id: str) -> float:
        """Calculate heartbeat reliability percentage"""
        if agent_id not in self.heartbeat_intervals:
            return 100.0
        
        intervals = self.heartbeat_intervals[agent_id]
        if len(intervals) < 10:  # Need enough data points
            return 100.0
        
        # Calculate what percentage of heartbeats arrived within expected interval
        expected = self.heartbeat_interval
        tolerance = expected * 0.2  # 20% tolerance
        
        on_time = sum(1 for interval in intervals if interval <= expected + tolerance)
        reliability = (on_time / len(intervals)) * 100
        
        return reliability

    async def _cleanup_old_data(self):
        """Clean up old tracking data"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=24)  # Keep 24 hours of data
        
        # Clean up old heartbeat intervals
        for agent_id in list(self.heartbeat_intervals.keys()):
            # Keep only recent intervals (last 1000)
            if len(self.heartbeat_intervals[agent_id]) > 1000:
                self.heartbeat_intervals[agent_id] = self.heartbeat_intervals[agent_id][-500:]
        
        # Clean up old response times
        for agent_id in list(self.response_times.keys()):
            if len(self.response_times[agent_id]) > 1000:
                self.response_times[agent_id] = self.response_times[agent_id][-500:]
        
        # Clean up very old status history entries
        for agent_id in list(self.status_history.keys()):
            if agent_id in self.status_history:
                old_history = self.status_history[agent_id]
                recent_history = [
                    entry for entry in old_history 
                    if entry[0] > cutoff_time
                ]
                if len(recent_history) != len(old_history):
                    self.status_history[agent_id] = recent_history

    # Public interface methods
    
    def register_agent(self, agent_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Register a new agent for availability tracking"""
        current_time = datetime.now()
        
        if agent_id not in self.heartbeats:
            self.heartbeats[agent_id] = HeartbeatInfo(
                agent_id=agent_id,
                last_heartbeat=current_time,
                consecutive_misses=0,
                average_interval=self.heartbeat_interval,
                status=AgentStatus.ONLINE,
                response_time=0.0,
                metadata=metadata or {}
            )
            
            self.heartbeat_intervals[agent_id] = []
            self.response_times[agent_id] = []
            self.status_history[agent_id] = []
            
            logger.info("Registered agent %s for availability tracking", agent_id)

    def record_heartbeat(self, agent_id: str, response_time: float = 0.0, metadata: Optional[Dict[str, Any]] = None):
        """Record a heartbeat from an agent"""
        current_time = datetime.now()
        
        if agent_id not in self.heartbeats:
            self.register_agent(agent_id, metadata)
        
        heartbeat_info = self.heartbeats[agent_id]
        
        # Calculate interval since last heartbeat
        if heartbeat_info.last_heartbeat:
            interval = (current_time - heartbeat_info.last_heartbeat).total_seconds()
            self.heartbeat_intervals[agent_id].append(interval)
            
            # Update average interval
            recent_intervals = self.heartbeat_intervals[agent_id][-20:]  # Last 20 intervals
            heartbeat_info.average_interval = sum(recent_intervals) / len(recent_intervals)
        
        # Update heartbeat info
        heartbeat_info.last_heartbeat = current_time
        heartbeat_info.consecutive_misses = 0
        heartbeat_info.response_time = response_time
        
        # Update metadata if provided
        if metadata:
            heartbeat_info.metadata.update(metadata)
        
        # Record response time
        self.response_times[agent_id].append(response_time)
        
        # Ensure status is online if heartbeat received
        if heartbeat_info.status != AgentStatus.ONLINE:
            old_status = heartbeat_info.status
            heartbeat_info.status = AgentStatus.ONLINE
            self._record_status_change(agent_id, AgentStatus.ONLINE, "Heartbeat received")
            
            # Trigger callbacks asynchronously
            asyncio.create_task(self._handle_status_change(agent_id, old_status, AgentStatus.ONLINE, current_time))

    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Get current status of an agent"""
        heartbeat_info = self.heartbeats.get(agent_id)
        return heartbeat_info.status if heartbeat_info else None

    def get_agent_heartbeat_info(self, agent_id: str) -> Optional[HeartbeatInfo]:
        """Get heartbeat information for an agent"""
        return self.heartbeats.get(agent_id)

    def get_agent_availability_metrics(self, agent_id: str) -> Optional[AvailabilityMetrics]:
        """Get availability metrics for an agent"""
        return self.availability_metrics.get(agent_id)

    def get_all_agent_statuses(self) -> Dict[str, AgentStatus]:
        """Get status of all tracked agents"""
        return {agent_id: info.status for agent_id, info in self.heartbeats.items()}

    def get_online_agents(self) -> List[str]:
        """Get list of currently online agents"""
        return [
            agent_id for agent_id, info in self.heartbeats.items()
            if info.status == AgentStatus.ONLINE
        ]

    def get_offline_agents(self) -> List[str]:
        """Get list of currently offline agents"""
        return [
            agent_id for agent_id, info in self.heartbeats.items()
            if info.status == AgentStatus.OFFLINE
        ]

    def get_degraded_agents(self) -> List[str]:
        """Get list of currently degraded agents"""
        return [
            agent_id for agent_id, info in self.heartbeats.items()
            if info.status == AgentStatus.DEGRADED
        ]

    def get_status_history(self, agent_id: str, limit: int = 50) -> List[tuple]:
        """Get status change history for an agent"""
        history = self.status_history.get(agent_id, [])
        return history[-limit:] if limit else history

    def get_availability_summary(self) -> Dict[str, Any]:
        """Get overall availability summary"""
        if not self.heartbeats:
            return {"total_agents": 0, "online": 0, "offline": 0, "degraded": 0, "overall_availability": 0.0}
        
        statuses = [info.status for info in self.heartbeats.values()]
        
        summary = {
            "total_agents": len(statuses),
            "online": sum(1 for s in statuses if s == AgentStatus.ONLINE),
            "offline": sum(1 for s in statuses if s == AgentStatus.OFFLINE),
            "degraded": sum(1 for s in statuses if s == AgentStatus.DEGRADED),
            "maintenance": sum(1 for s in statuses if s == AgentStatus.MAINTENANCE),
            "overall_availability": 0.0,
        }
        
        # Calculate overall availability
        if self.availability_metrics:
            avg_uptime = sum(metrics.uptime_percentage for metrics in self.availability_metrics.values())
            summary["overall_availability"] = avg_uptime / len(self.availability_metrics)
        
        return summary

    # Callback management
    
    def add_status_change_callback(self, callback: Callable):
        """Add callback for status changes"""
        self.status_change_callbacks.append(callback)

    def add_failure_callback(self, callback: Callable):
        """Add callback for agent failures"""
        self.failure_callbacks.append(callback)

    def add_recovery_callback(self, callback: Callable):
        """Add callback for agent recoveries"""
        self.recovery_callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """Remove a callback from all callback lists"""
        for callback_list in [self.status_change_callbacks, self.failure_callbacks, self.recovery_callbacks]:
            if callback in callback_list:
                callback_list.remove(callback)

    def set_agent_maintenance_mode(self, agent_id: str, maintenance: bool = True):
        """Set an agent to maintenance mode"""
        if agent_id in self.heartbeats:
            old_status = self.heartbeats[agent_id].status
            new_status = AgentStatus.MAINTENANCE if maintenance else AgentStatus.ONLINE
            self.heartbeats[agent_id].status = new_status
            
            reason = "Entered maintenance mode" if maintenance else "Exited maintenance mode"
            self._record_status_change(agent_id, new_status, reason)
            
            # Trigger callbacks
            asyncio.create_task(self._handle_status_change(agent_id, old_status, new_status, datetime.now()))