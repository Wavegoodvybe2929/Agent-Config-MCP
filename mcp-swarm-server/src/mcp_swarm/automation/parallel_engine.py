"""
Parallel Execution Engine for MCP Swarm Intelligence Server

This module implements optimal parallel task execution with resource utilization,
providing efficient concurrent execution, resource management, and dependency
resolution for multi-agent workflows.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Awaitable
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import psutil
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Parallel execution strategies"""
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    ASYNCIO_CONCURRENT = "asyncio_concurrent"
    HYBRID = "hybrid"


class ResourceType(Enum):
    """System resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"


class TaskPriority(Enum):
    """Task execution priorities"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class ResourceRequirement:
    """Resource requirements for task execution"""
    cpu_cores: float = 1.0
    memory_mb: int = 512
    io_intensive: bool = False
    network_intensive: bool = False
    gpu_required: bool = False


@dataclass
class ExecutionTask:
    """Task for parallel execution"""
    id: str
    name: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    resource_requirements: ResourceRequirement = field(default_factory=ResourceRequirement)
    estimated_duration: float = 300.0
    retry_attempts: int = 1
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ResourceAllocation:
    """Resource allocation for parallel execution"""
    allocated_cpu_cores: float
    allocated_memory_mb: int
    max_concurrent_tasks: int
    execution_strategy: ExecutionStrategy
    pool_size: int
    resource_limits: Dict[ResourceType, float]


@dataclass
class ExecutionResult:
    """Result of task execution"""
    task_id: str
    status: str  # "completed", "failed", "timeout", "cancelled"
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class ParallelExecutionResult:
    """Result of parallel execution batch"""
    execution_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    cancelled_tasks: int
    total_execution_time: float
    task_results: Dict[str, ExecutionResult]
    resource_utilization: Dict[str, float]
    performance_metrics: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None


class ResourceManager:
    """System resource management for optimal allocation"""

    def __init__(self):
        """Initialize resource manager"""
        self.system_resources = self._detect_system_resources()
        self.allocated_resources = {
            ResourceType.CPU: 0.0,
            ResourceType.MEMORY: 0,
            ResourceType.IO: 0.0,
            ResourceType.NETWORK: 0.0
        }

    def _detect_system_resources(self) -> Dict[str, Any]:
        """Detect available system resources"""
        try:
            cpu_count = psutil.cpu_count(logical=True)
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            return {
                "cpu_cores": cpu_count,
                "logical_cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
                "memory_total_mb": memory_info.total // (1024 * 1024),
                "memory_available_mb": memory_info.available // (1024 * 1024),
                "disk_total_gb": disk_info.total // (1024 * 1024 * 1024),
                "disk_free_gb": disk_info.free // (1024 * 1024 * 1024)
            }
        except Exception as e:
            logger.warning("Failed to detect system resources: %s", e)
            return {
                "cpu_cores": 4,
                "logical_cores": 4,
                "physical_cores": 2,
                "memory_total_mb": 8192,
                "memory_available_mb": 4096,
                "disk_total_gb": 100,
                "disk_free_gb": 50
            }

    def get_available_resources(self) -> Dict[str, float]:
        """Get currently available resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            
            available_cpu = (100 - cpu_percent) / 100.0 * self.system_resources["cpu_cores"]
            available_memory = memory_info.available // (1024 * 1024)
            
            return {
                "cpu_cores": max(0, available_cpu - self.allocated_resources[ResourceType.CPU]),
                "memory_mb": max(0, available_memory - self.allocated_resources[ResourceType.MEMORY]),
                "cpu_percent": 100 - cpu_percent,
                "memory_percent": memory_info.percent
            }
        except Exception as e:
            logger.warning("Failed to get available resources: %s", e)
            return {
                "cpu_cores": 2.0,
                "memory_mb": 2048,
                "cpu_percent": 50.0,
                "memory_percent": 50.0
            }

    def allocate_resources(
        self, 
        requirement: ResourceRequirement
    ) -> bool:
        """
        Allocate resources for a task.
        
        Args:
            requirement: Resource requirement
            
        Returns:
            True if allocation successful, False otherwise
        """
        available = self.get_available_resources()
        
        # Check if resources are available
        if (requirement.cpu_cores <= available["cpu_cores"] and
            requirement.memory_mb <= available["memory_mb"]):
            
            # Allocate resources
            self.allocated_resources[ResourceType.CPU] += requirement.cpu_cores
            self.allocated_resources[ResourceType.MEMORY] += requirement.memory_mb
            
            return True
        
        return False

    def deallocate_resources(self, requirement: ResourceRequirement):
        """
        Deallocate resources after task completion.
        
        Args:
            requirement: Resource requirement to deallocate
        """
        self.allocated_resources[ResourceType.CPU] = max(
            0, self.allocated_resources[ResourceType.CPU] - requirement.cpu_cores
        )
        self.allocated_resources[ResourceType.MEMORY] = max(
            0, self.allocated_resources[ResourceType.MEMORY] - requirement.memory_mb
        )


class DependencyResolver:
    """Dependency resolution for task execution order"""

    def __init__(self):
        """Initialize dependency resolver"""
        pass

    def resolve_dependencies(
        self, 
        tasks: List[ExecutionTask]
    ) -> List[List[ExecutionTask]]:
        """
        Resolve task dependencies and group tasks for parallel execution.
        
        Args:
            tasks: List of tasks with dependencies
            
        Returns:
            List of task groups that can be executed in parallel
        """
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(tasks)
        
        # Perform topological sort with parallelization
        execution_groups = self._topological_sort_parallel(dependency_graph)
        
        return execution_groups

    def _build_dependency_graph(
        self, 
        tasks: List[ExecutionTask]
    ) -> Dict[str, Set[str]]:
        """Build dependency graph from tasks"""
        graph = {}
        task_ids = {task.id for task in tasks}
        
        for task in tasks:
            # Filter valid dependencies
            valid_deps = [dep for dep in task.dependencies if dep in task_ids]
            graph[task.id] = set(valid_deps)
        
        return graph

    def _topological_sort_parallel(
        self, 
        dependency_graph: Dict[str, Set[str]]
    ) -> List[List[str]]:
        """Perform topological sort with parallel grouping"""
        execution_groups = []
        remaining_nodes = set(dependency_graph.keys())
        
        while remaining_nodes:
            # Find nodes with no unresolved dependencies
            ready_nodes = []
            for node in remaining_nodes:
                if not dependency_graph[node].intersection(remaining_nodes):
                    ready_nodes.append(node)
            
            if not ready_nodes:
                # Circular dependency - break arbitrarily
                ready_nodes = [remaining_nodes.pop()]
                logger.warning("Circular dependency detected, breaking arbitrarily")
            
            # Add ready nodes as a parallel group
            execution_groups.append(ready_nodes)
            remaining_nodes -= set(ready_nodes)
        
        return execution_groups


class ParallelExecutionEngine:
    """
    Optimize parallel task execution with resource utilization.
    
    This engine provides efficient concurrent execution, intelligent resource
    management, and optimal task scheduling for multi-agent workflows.
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel execution engine.
        
        Args:
            max_workers: Maximum number of worker threads/processes
        """
        self.resource_manager = ResourceManager()
        self.dependency_resolver = DependencyResolver()
        
        # Configure execution pools
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 4) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(self.max_workers, psutil.cpu_count() or 4))
        
        # Execution tracking
        self.active_executions: Dict[str, ParallelExecutionResult] = {}
        self.execution_history: List[ParallelExecutionResult] = []
        
        # Performance metrics
        self.metrics = {
            "total_executions": 0,
            "total_tasks_executed": 0,
            "average_execution_time": 0.0,
            "resource_utilization_efficiency": 0.0,
            "success_rate": 0.0
        }

    async def execute_parallel_tasks(
        self, 
        tasks: List[ExecutionTask]
    ) -> ParallelExecutionResult:
        """
        Execute tasks in parallel with optimal resource allocation.
        
        Args:
            tasks: List of tasks to execute
            
        Returns:
            Parallel execution result with performance metrics
        """
        execution_id = f"parallel_exec_{int(datetime.now().timestamp())}"
        logger.info("Starting parallel execution %s with %d tasks", execution_id, len(tasks))
        
        start_time = datetime.now()
        
        # Phase 1: Optimize resource allocation
        resource_allocation = await self._optimize_resource_allocation(tasks)
        
        # Phase 2: Resolve dependencies and group tasks
        execution_groups = self.dependency_resolver.resolve_dependencies(tasks)
        
        # Phase 3: Execute task groups in parallel
        task_results = await self._execute_task_groups(
            execution_groups, tasks, resource_allocation
        )
        
        # Phase 4: Compile execution result
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        result = self._compile_execution_result(
            execution_id, tasks, task_results, execution_time, start_time, end_time
        )
        
        # Update metrics and tracking
        self.active_executions[execution_id] = result
        self.execution_history.append(result)
        self._update_metrics(result)
        
        logger.info("Completed parallel execution %s in %.2f seconds", execution_id, execution_time)
        
        return result

    async def _optimize_resource_allocation(
        self, 
        tasks: List[ExecutionTask]
    ) -> ResourceAllocation:
        """
        Optimize resource allocation for parallel execution.
        
        Args:
            tasks: List of tasks to allocate resources for
            
        Returns:
            Optimized resource allocation configuration
        """
        # Analyze task resource requirements
        total_cpu_needed = sum(task.resource_requirements.cpu_cores for task in tasks)
        total_memory_needed = sum(task.resource_requirements.memory_mb for task in tasks)
        
        # Get available system resources
        available_resources = self.resource_manager.get_available_resources()
        
        # Determine optimal allocation strategy
        if total_cpu_needed <= available_resources["cpu_cores"] * 1.2:
            # Can run most tasks concurrently
            max_concurrent = min(len(tasks), int(available_resources["cpu_cores"]))
            execution_strategy = ExecutionStrategy.ASYNCIO_CONCURRENT
        elif total_memory_needed > available_resources["memory_mb"] * 0.8:
            # Memory constrained - use process pool
            max_concurrent = max(1, int(available_resources["memory_mb"] // 
                                     (total_memory_needed / len(tasks))))
            execution_strategy = ExecutionStrategy.PROCESS_POOL
        else:
            # Balanced load - use thread pool
            max_concurrent = min(self.max_workers, len(tasks))
            execution_strategy = ExecutionStrategy.THREAD_POOL
        
        # Calculate resource limits
        resource_limits = {
            ResourceType.CPU: min(available_resources["cpu_cores"], total_cpu_needed),
            ResourceType.MEMORY: min(available_resources["memory_mb"], total_memory_needed),
            ResourceType.IO: 0.8,  # 80% IO utilization limit
            ResourceType.NETWORK: 0.8  # 80% network utilization limit
        }
        
        return ResourceAllocation(
            allocated_cpu_cores=resource_limits[ResourceType.CPU],
            allocated_memory_mb=int(resource_limits[ResourceType.MEMORY]),
            max_concurrent_tasks=max_concurrent,
            execution_strategy=execution_strategy,
            pool_size=min(max_concurrent, self.max_workers),
            resource_limits=resource_limits
        )

    async def _execute_task_groups(
        self,
        execution_groups: List[List[str]],
        tasks: List[ExecutionTask],
        resource_allocation: ResourceAllocation
    ) -> Dict[str, ExecutionResult]:
        """Execute task groups with optimal parallel execution"""
        task_dict = {task.id: task for task in tasks}
        all_results = {}
        
        # Execute each group sequentially, but tasks within groups in parallel
        for group_index, task_group in enumerate(execution_groups):
            logger.info("Executing task group %d with %d tasks", group_index + 1, len(task_group))
            
            # Get tasks for this group
            group_tasks = [task_dict[task_id] for task_id in task_group]
            
            # Execute group tasks in parallel
            group_results = await self._execute_task_group_parallel(
                group_tasks, resource_allocation
            )
            
            all_results.update(group_results)
        
        return all_results

    async def _execute_task_group_parallel(
        self,
        group_tasks: List[ExecutionTask],
        resource_allocation: ResourceAllocation
    ) -> Dict[str, ExecutionResult]:
        """Execute a group of tasks in parallel"""
        semaphore = asyncio.Semaphore(resource_allocation.max_concurrent_tasks)
        
        async def execute_with_semaphore(task: ExecutionTask) -> Tuple[str, ExecutionResult]:
            async with semaphore:
                result = await self._execute_single_task(task, resource_allocation)
                return task.id, result
        
        # Execute all tasks in the group concurrently
        tasks_futures = [execute_with_semaphore(task) for task in group_tasks]
        results = await asyncio.gather(*tasks_futures, return_exceptions=True)
        
        # Process results
        group_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error("Task execution failed with exception: %s", result)
                # Create failure result for exception
                group_results["unknown"] = ExecutionResult(
                    task_id="unknown",
                    status="failed",
                    error=str(result)
                )
            else:
                task_id, execution_result = result
                group_results[task_id] = execution_result
        
        return group_results

    async def _execute_single_task(
        self,
        task: ExecutionTask,
        resource_allocation: ResourceAllocation
    ) -> ExecutionResult:
        """Execute a single task with resource management"""
        start_time = datetime.now()
        
        # Allocate resources
        if not self.resource_manager.allocate_resources(task.resource_requirements):
            return ExecutionResult(
                task_id=task.id,
                status="failed",
                error="Failed to allocate required resources",
                started_at=start_time
            )
        
        try:
            # Execute task based on strategy
            if resource_allocation.execution_strategy == ExecutionStrategy.ASYNCIO_CONCURRENT:
                result = await self._execute_async_task(task)
            elif resource_allocation.execution_strategy == ExecutionStrategy.THREAD_POOL:
                result = await self._execute_thread_task(task)
            elif resource_allocation.execution_strategy == ExecutionStrategy.PROCESS_POOL:
                result = await self._execute_process_task(task)
            else:
                result = await self._execute_hybrid_task(task)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return ExecutionResult(
                task_id=task.id,
                status="completed",
                result=result,
                execution_time=execution_time,
                started_at=start_time,
                completed_at=end_time,
                resource_usage={
                    "cpu_cores": task.resource_requirements.cpu_cores,
                    "memory_mb": task.resource_requirements.memory_mb
                }
            )
            
        except asyncio.TimeoutError:
            return ExecutionResult(
                task_id=task.id,
                status="timeout",
                error=f"Task timed out after {task.timeout} seconds",
                execution_time=task.timeout or 0,
                started_at=start_time,
                completed_at=datetime.now()
            )
            
        except Exception as e:
            return ExecutionResult(
                task_id=task.id,
                status="failed",
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
                started_at=start_time,
                completed_at=datetime.now()
            )
            
        finally:
            # Deallocate resources
            self.resource_manager.deallocate_resources(task.resource_requirements)

    async def _execute_async_task(self, task: ExecutionTask) -> Any:
        """Execute task using asyncio"""
        if task.timeout:
            return await asyncio.wait_for(
                self._run_task_function(task), 
                timeout=task.timeout
            )
        else:
            return await self._run_task_function(task)

    async def _execute_thread_task(self, task: ExecutionTask) -> Any:
        """Execute task using thread pool"""
        loop = asyncio.get_event_loop()
        future = self.thread_pool.submit(
            self._run_sync_task_function, task
        )
        
        if task.timeout:
            return await asyncio.wait_for(
                loop.run_in_executor(None, future.result),
                timeout=task.timeout
            )
        else:
            return await loop.run_in_executor(None, future.result)

    async def _execute_process_task(self, task: ExecutionTask) -> Any:
        """Execute task using process pool"""
        loop = asyncio.get_event_loop()
        
        try:
            future = self.process_pool.submit(
                self._run_sync_task_function, task
            )
            
            if task.timeout:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, future.result),
                    timeout=task.timeout
                )
            else:
                return await loop.run_in_executor(None, future.result)
        except Exception as e:
            # Process pool might not support all task types
            logger.warning("Process pool execution failed, falling back to thread pool: %s", e)
            return await self._execute_thread_task(task)

    async def _execute_hybrid_task(self, task: ExecutionTask) -> Any:
        """Execute task using hybrid strategy"""
        # Determine best execution method based on task characteristics
        if task.resource_requirements.io_intensive:
            return await self._execute_thread_task(task)
        elif task.resource_requirements.cpu_cores > 1:
            return await self._execute_process_task(task)
        else:
            return await self._execute_async_task(task)

    async def _run_task_function(self, task: ExecutionTask) -> Any:
        """Run task function asynchronously"""
        if asyncio.iscoroutinefunction(task.function):
            return await task.function(*task.args, **task.kwargs)
        else:
            # Run sync function in thread
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: task.function(*task.args, **task.kwargs)
            )

    def _run_sync_task_function(self, task: ExecutionTask) -> Any:
        """Run task function synchronously"""
        return task.function(*task.args, **task.kwargs)

    def _compile_execution_result(
        self,
        execution_id: str,
        tasks: List[ExecutionTask],
        task_results: Dict[str, ExecutionResult],
        execution_time: float,
        start_time: datetime,
        end_time: datetime
    ) -> ParallelExecutionResult:
        """Compile final execution result"""
        total_tasks = len(tasks)
        completed_tasks = sum(1 for result in task_results.values() 
                            if result.status == "completed")
        failed_tasks = sum(1 for result in task_results.values() 
                         if result.status == "failed")
        cancelled_tasks = sum(1 for result in task_results.values() 
                            if result.status == "cancelled")
        
        # Calculate resource utilization
        total_cpu_used = sum(result.resource_usage.get("cpu_cores", 0) 
                           for result in task_results.values())
        total_memory_used = sum(result.resource_usage.get("memory_mb", 0) 
                              for result in task_results.values())
        
        available_resources = self.resource_manager.get_available_resources()
        cpu_utilization = min(100.0, (total_cpu_used / available_resources["cpu_cores"]) * 100)
        memory_utilization = min(100.0, (total_memory_used / available_resources["memory_mb"]) * 100)
        
        # Calculate performance metrics
        successful_task_times = [result.execution_time for result in task_results.values() 
                               if result.status == "completed"]
        avg_task_time = sum(successful_task_times) / len(successful_task_times) if successful_task_times else 0
        
        performance_metrics = {
            "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "average_task_time": avg_task_time,
            "throughput": completed_tasks / execution_time if execution_time > 0 else 0,
            "cpu_utilization_percent": cpu_utilization,
            "memory_utilization_percent": memory_utilization,
            "parallel_efficiency": self._calculate_parallel_efficiency(tasks, task_results, execution_time)
        }
        
        return ParallelExecutionResult(
            execution_id=execution_id,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            cancelled_tasks=cancelled_tasks,
            total_execution_time=execution_time,
            task_results=task_results,
            resource_utilization={
                "cpu_utilization": cpu_utilization,
                "memory_utilization": memory_utilization,
                "peak_concurrent_tasks": min(len(tasks), self.max_workers)
            },
            performance_metrics=performance_metrics,
            started_at=start_time,
            completed_at=end_time
        )

    def _calculate_parallel_efficiency(
        self,
        tasks: List[ExecutionTask],
        task_results: Dict[str, ExecutionResult],
        total_execution_time: float
    ) -> float:
        """Calculate parallel execution efficiency"""
        # Sequential execution time estimate
        sequential_time = sum(task.estimated_duration for task in tasks)
        
        # Actual parallel execution time
        if total_execution_time > 0 and sequential_time > 0:
            return min(1.0, sequential_time / total_execution_time / len(tasks))
        
        return 0.0

    def _update_metrics(self, result: ParallelExecutionResult):
        """Update engine performance metrics"""
        self.metrics["total_executions"] += 1
        self.metrics["total_tasks_executed"] += result.total_tasks
        
        # Update averages
        total_executions = self.metrics["total_executions"]
        current_avg_time = self.metrics["average_execution_time"]
        
        self.metrics["average_execution_time"] = (
            (current_avg_time * (total_executions - 1) + result.total_execution_time) 
            / total_executions
        )
        
        current_success_rate = self.metrics["success_rate"]
        task_success_rate = result.performance_metrics["success_rate"]
        
        self.metrics["success_rate"] = (
            (current_success_rate * (total_executions - 1) + task_success_rate)
            / total_executions
        )
        
        # Update resource utilization efficiency
        cpu_util = result.resource_utilization["cpu_utilization"]
        memory_util = result.resource_utilization["memory_utilization"]
        current_efficiency = (cpu_util + memory_util) / 2
        
        current_res_efficiency = self.metrics["resource_utilization_efficiency"]
        self.metrics["resource_utilization_efficiency"] = (
            (current_res_efficiency * (total_executions - 1) + current_efficiency)
            / total_executions
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics"""
        return self.metrics.copy()

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and resource availability"""
        available_resources = self.resource_manager.get_available_resources()
        
        return {
            "system_resources": self.resource_manager.system_resources,
            "available_resources": available_resources,
            "allocated_resources": dict(self.resource_manager.allocated_resources),
            "active_executions": len(self.active_executions),
            "max_workers": self.max_workers,
            "pool_status": {
                "thread_pool_active": True,
                "process_pool_active": True
            }
        }

    def shutdown(self):
        """Shutdown execution pools"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except:
            pass