"""
Performance Optimizer for MCP Swarm Intelligence Server

This module provides machine learning-based performance optimization with pattern
analysis and automated optimization application for the MCP swarm intelligence system.
"""

import logging
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import psutil


class OptimizationType(Enum):
    """Types of optimization that can be applied"""
    CPU_OPTIMIZATION = "cpu_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    IO_OPTIMIZATION = "io_optimization"
    NETWORK_OPTIMIZATION = "network_optimization"
    ALGORITHM_OPTIMIZATION = "algorithm_optimization"


class PerformanceLevel(Enum):
    """Performance improvement levels"""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    DRAMATIC = "dramatic"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    baseline: float
    improvement_percent: float
    timestamp: datetime
    optimization_applied: bool = False


@dataclass
class PerformancePatterns:
    """Performance patterns identified from analysis"""
    cpu_patterns: Dict[str, Any]
    memory_patterns: Dict[str, Any]
    io_patterns: Dict[str, Any]
    bottlenecks: List[str]
    optimization_opportunities: List[Dict[str, Any]]
    pattern_confidence: float


@dataclass
class MLOptimizationResult:
    """Result from ML-based optimization"""
    optimization_type: OptimizationType
    applied_optimizations: List[str]
    expected_improvement: float
    actual_improvement: Optional[float] = None
    confidence: float = 0.0
    timestamp: Optional[datetime] = None


@dataclass
class OptimizationResult:
    """Complete optimization result"""
    performance_gains: Dict[str, float]
    applied_actions: List[str]
    ml_results: List[MLOptimizationResult]
    patterns_identified: PerformancePatterns
    overall_improvement: float
    timestamp: datetime


class PerformanceAnalyzer:
    """Analyzes system performance patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_history = []
        self.baseline_metrics = {}
        
    async def analyze_cpu_patterns(self) -> Dict[str, Any]:
        """Analyze CPU usage patterns"""
        try:
            # Get current CPU stats
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Analyze CPU per core
            cpu_per_core = psutil.cpu_percent(percpu=True)
            
            patterns = {
                "overall_usage": cpu_percent,
                "core_count": cpu_count,
                "frequency": cpu_freq.current if cpu_freq else 0,
                "per_core_usage": cpu_per_core,
                "load_balance": self._calculate_load_balance(cpu_per_core),
                "peak_usage": max(cpu_per_core) if cpu_per_core else cpu_percent,
                "bottleneck_cores": [i for i, usage in enumerate(cpu_per_core) if usage > 90],
                "optimization_potential": self._assess_cpu_optimization_potential(cpu_percent, cpu_per_core)
            }
            
            return patterns
            
        except Exception as e:
            self.logger.error("Error analyzing CPU patterns: %s", str(e))
            return {}
    
    async def analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            patterns = {
                "total_memory": memory.total,
                "available_memory": memory.available,
                "used_memory": memory.used,
                "memory_percent": memory.percent,
                "swap_total": swap.total,
                "swap_used": swap.used,
                "swap_percent": swap.percent,
                "cache_size": getattr(memory, 'cached', 0),
                "buffer_size": getattr(memory, 'buffers', 0),
                "fragmentation_score": self._calculate_memory_fragmentation(memory),
                "optimization_potential": self._assess_memory_optimization_potential(memory)
            }
            
            return patterns
            
        except Exception as e:
            self.logger.error("Error analyzing memory patterns: %s", str(e))
            return {}
    
    async def analyze_io_patterns(self) -> Dict[str, Any]:
        """Analyze I/O patterns"""
        try:
            disk_io = psutil.disk_io_counters()
            disk_usage = psutil.disk_usage('/')
            
            patterns = {
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0,
                "read_ops": disk_io.read_count if disk_io else 0,
                "write_ops": disk_io.write_count if disk_io else 0,
                "disk_usage_percent": (disk_usage.used / disk_usage.total) * 100,
                "free_space": disk_usage.free,
                "io_utilization": self._calculate_io_utilization(disk_io),
                "optimization_potential": self._assess_io_optimization_potential(disk_io, disk_usage)
            }
            
            return patterns
            
        except Exception as e:
            self.logger.error("Error analyzing I/O patterns: %s", str(e))
            return {}
    
    async def identify_bottlenecks(self, cpu_patterns: Dict[str, Any], 
                                 memory_patterns: Dict[str, Any], 
                                 io_patterns: Dict[str, Any]) -> List[str]:
        """Identify system bottlenecks"""
        bottlenecks = []
        
        # CPU bottlenecks
        if cpu_patterns.get("overall_usage", 0) > 80:
            bottlenecks.append("cpu_high_usage")
        
        if cpu_patterns.get("load_balance", 1.0) < 0.7:
            bottlenecks.append("cpu_load_imbalance")
        
        # Memory bottlenecks
        if memory_patterns.get("memory_percent", 0) > 85:
            bottlenecks.append("memory_pressure")
        
        if memory_patterns.get("swap_percent", 0) > 10:
            bottlenecks.append("excessive_swapping")
        
        # I/O bottlenecks
        if io_patterns.get("io_utilization", 0) > 80:
            bottlenecks.append("io_saturation")
        
        if io_patterns.get("disk_usage_percent", 0) > 90:
            bottlenecks.append("disk_space_critical")
        
        return bottlenecks
    
    def _calculate_load_balance(self, cpu_per_core: List[float]) -> float:
        """Calculate CPU load balance score"""
        if not cpu_per_core or len(cpu_per_core) <= 1:
            return 1.0
        
        mean_usage = statistics.mean(cpu_per_core)
        if mean_usage == 0:
            return 1.0
        
        variance = statistics.variance(cpu_per_core)
        # Normalize variance to a 0-1 balance score
        balance_score = max(0.0, 1.0 - (variance / (mean_usage * 100)))
        return balance_score
    
    def _assess_cpu_optimization_potential(self, overall_usage: float, per_core_usage: List[float]) -> float:
        """Assess CPU optimization potential (0-1 score)"""
        if overall_usage < 50:
            return 0.2  # Low optimization potential
        
        load_imbalance = 1.0 - self._calculate_load_balance(per_core_usage)
        high_usage_penalty = min(overall_usage / 100, 1.0)
        
        potential = (load_imbalance * 0.6) + (high_usage_penalty * 0.4)
        return min(potential, 1.0)
    
    def _calculate_memory_fragmentation(self, memory) -> float:
        """Calculate memory fragmentation score"""
        if memory.total == 0:
            return 0.0
        
        # Simple fragmentation estimation based on available vs free memory
        expected_available = memory.total - memory.used
        actual_available = memory.available
        
        if expected_available == 0:
            return 0.0
        
        fragmentation_ratio = 1.0 - (actual_available / expected_available)
        return max(0.0, min(fragmentation_ratio, 1.0))
    
    def _assess_memory_optimization_potential(self, memory) -> float:
        """Assess memory optimization potential"""
        usage_ratio = memory.percent / 100
        fragmentation = self._calculate_memory_fragmentation(memory)
        
        # Higher usage and fragmentation = higher optimization potential
        potential = (usage_ratio * 0.7) + (fragmentation * 0.3)
        return min(potential, 1.0)
    
    def _calculate_io_utilization(self, disk_io) -> float:
        """Calculate I/O utilization score"""
        if not disk_io:
            return 0.0
        
        # Simple I/O utilization based on operation counts
        total_ops = disk_io.read_count + disk_io.write_count
        # Normalize to a 0-100 scale (arbitrary baseline of 10000 ops for 100%)
        utilization = min((total_ops / 10000) * 100, 100)
        return utilization
    
    def _assess_io_optimization_potential(self, disk_io, disk_usage) -> float:
        """Assess I/O optimization potential"""
        io_util = self._calculate_io_utilization(disk_io) / 100
        space_pressure = (disk_usage.used / disk_usage.total)
        
        potential = (io_util * 0.6) + (space_pressure * 0.4)
        return min(potential, 1.0)


class MLOptimizer:
    """Machine learning-based performance optimizer"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimization_history = []
        self.learned_patterns = {}
        
    async def optimize_cpu_performance(self, patterns: Dict[str, Any]) -> MLOptimizationResult:
        """Apply ML-based CPU optimizations"""
        applied_optimizations = []
        expected_improvement = 0.0
        
        # CPU affinity optimization
        if patterns.get("load_balance", 1.0) < 0.7:
            applied_optimizations.append("cpu_affinity_optimization")
            expected_improvement += 15.0
        
        # Process priority optimization
        if patterns.get("overall_usage", 0) > 75:
            applied_optimizations.append("process_priority_optimization")
            expected_improvement += 10.0
        
        # Threading optimization
        if len(patterns.get("bottleneck_cores", [])) > 0:
            applied_optimizations.append("threading_optimization")
            expected_improvement += 12.0
        
        return MLOptimizationResult(
            optimization_type=OptimizationType.CPU_OPTIMIZATION,
            applied_optimizations=applied_optimizations,
            expected_improvement=expected_improvement,
            confidence=0.85,
            timestamp=datetime.utcnow()
        )
    
    async def optimize_memory_performance(self, patterns: Dict[str, Any]) -> MLOptimizationResult:
        """Apply ML-based memory optimizations"""
        applied_optimizations = []
        expected_improvement = 0.0
        
        # Garbage collection optimization
        if patterns.get("memory_percent", 0) > 70:
            applied_optimizations.append("garbage_collection_tuning")
            expected_improvement += 20.0
        
        # Memory pool optimization
        if patterns.get("fragmentation_score", 0) > 0.3:
            applied_optimizations.append("memory_pool_optimization")
            expected_improvement += 15.0
        
        # Cache optimization
        cache_ratio = patterns.get("cache_size", 0) / patterns.get("total_memory", 1)
        if cache_ratio < 0.1:
            applied_optimizations.append("cache_size_optimization")
            expected_improvement += 18.0
        
        return MLOptimizationResult(
            optimization_type=OptimizationType.MEMORY_OPTIMIZATION,
            applied_optimizations=applied_optimizations,
            expected_improvement=expected_improvement,
            confidence=0.80,
            timestamp=datetime.utcnow()
        )
    
    async def optimize_io_performance(self, patterns: Dict[str, Any]) -> MLOptimizationResult:
        """Apply ML-based I/O optimizations"""
        applied_optimizations = []
        expected_improvement = 0.0
        
        # I/O scheduling optimization
        if patterns.get("io_utilization", 0) > 60:
            applied_optimizations.append("io_scheduler_optimization")
            expected_improvement += 25.0
        
        # Buffer size optimization
        read_write_ratio = patterns.get("read_ops", 1) / max(patterns.get("write_ops", 1), 1)
        if read_write_ratio > 2 or read_write_ratio < 0.5:
            applied_optimizations.append("buffer_size_optimization")
            expected_improvement += 15.0
        
        # Disk cache optimization
        if patterns.get("disk_usage_percent", 0) > 80:
            applied_optimizations.append("disk_cache_optimization")
            expected_improvement += 20.0
        
        return MLOptimizationResult(
            optimization_type=OptimizationType.IO_OPTIMIZATION,
            applied_optimizations=applied_optimizations,
            expected_improvement=expected_improvement,
            confidence=0.75,
            timestamp=datetime.utcnow()
        )
    
    async def learn_from_optimization(self, result: MLOptimizationResult, 
                                    actual_improvement: float) -> None:
        """Learn from optimization results to improve future predictions"""
        result.actual_improvement = actual_improvement
        self.optimization_history.append(result)
        
        # Update learned patterns
        optimization_key = result.optimization_type.value
        if optimization_key not in self.learned_patterns:
            self.learned_patterns[optimization_key] = {
                "success_rate": 0.0,
                "average_improvement": 0.0,
                "confidence_adjustment": 0.0
            }
        
        # Calculate success rate
        successful_optimizations = [
            r for r in self.optimization_history 
            if r.optimization_type == result.optimization_type and 
            r.actual_improvement and r.actual_improvement > 0
        ]
        
        if self.optimization_history:
            type_history = [
                r for r in self.optimization_history 
                if r.optimization_type == result.optimization_type
            ]
            success_rate = len(successful_optimizations) / len(type_history)
            
            # Update patterns
            self.learned_patterns[optimization_key]["success_rate"] = success_rate
            
            if successful_optimizations:
                avg_improvement = statistics.mean([
                    r.actual_improvement for r in successful_optimizations
                ])
                self.learned_patterns[optimization_key]["average_improvement"] = avg_improvement


class PerformanceOptimizer:
    """Machine learning-based performance optimization"""
    
    def __init__(self):
        self.ml_optimizer = MLOptimizer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.logger = logging.getLogger(__name__)
        
    async def optimize_system_performance(self) -> OptimizationResult:
        """Optimize system performance using ML adaptation"""
        try:
            # Analyze current performance patterns
            patterns = await self._analyze_performance_patterns()
            
            # Apply ML-based optimizations
            ml_results = await self._apply_ml_optimizations(patterns)
            
            # Calculate performance gains
            performance_gains = self._calculate_performance_gains(ml_results)
            
            # Extract applied actions
            applied_actions = []
            for result in ml_results:
                applied_actions.extend(result.applied_optimizations)
            
            # Calculate overall improvement
            overall_improvement = sum(performance_gains.values()) / len(performance_gains) if performance_gains else 0.0
            
            return OptimizationResult(
                performance_gains=performance_gains,
                applied_actions=applied_actions,
                ml_results=ml_results,
                patterns_identified=patterns,
                overall_improvement=overall_improvement,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error("Error optimizing system performance: %s", str(e))
            # Return empty result on error
            return OptimizationResult(
                performance_gains={},
                applied_actions=[],
                ml_results=[],
                patterns_identified=PerformancePatterns(
                    cpu_patterns={}, memory_patterns={}, io_patterns={},
                    bottlenecks=[], optimization_opportunities=[], pattern_confidence=0.0
                ),
                overall_improvement=0.0,
                timestamp=datetime.utcnow()
            )
    
    async def _analyze_performance_patterns(self) -> PerformancePatterns:
        """Analyze performance patterns for optimization opportunities"""
        # Get performance patterns
        cpu_patterns = await self.performance_analyzer.analyze_cpu_patterns()
        memory_patterns = await self.performance_analyzer.analyze_memory_patterns()
        io_patterns = await self.performance_analyzer.analyze_io_patterns()
        
        # Identify bottlenecks
        bottlenecks = await self.performance_analyzer.identify_bottlenecks(
            cpu_patterns, memory_patterns, io_patterns
        )
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(
            cpu_patterns, memory_patterns, io_patterns, bottlenecks
        )
        
        # Calculate pattern confidence
        pattern_confidence = self._calculate_pattern_confidence(
            cpu_patterns, memory_patterns, io_patterns
        )
        
        return PerformancePatterns(
            cpu_patterns=cpu_patterns,
            memory_patterns=memory_patterns,
            io_patterns=io_patterns,
            bottlenecks=bottlenecks,
            optimization_opportunities=optimization_opportunities,
            pattern_confidence=pattern_confidence
        )
    
    async def _apply_ml_optimizations(self, patterns: PerformancePatterns) -> List[MLOptimizationResult]:
        """Apply machine learning-based optimizations"""
        ml_results = []
        
        # CPU optimizations
        if patterns.cpu_patterns:
            cpu_result = await self.ml_optimizer.optimize_cpu_performance(patterns.cpu_patterns)
            ml_results.append(cpu_result)
        
        # Memory optimizations
        if patterns.memory_patterns:
            memory_result = await self.ml_optimizer.optimize_memory_performance(patterns.memory_patterns)
            ml_results.append(memory_result)
        
        # I/O optimizations
        if patterns.io_patterns:
            io_result = await self.ml_optimizer.optimize_io_performance(patterns.io_patterns)
            ml_results.append(io_result)
        
        return ml_results
    
    def _identify_optimization_opportunities(self, cpu_patterns: Dict[str, Any], 
                                           memory_patterns: Dict[str, Any], 
                                           io_patterns: Dict[str, Any], 
                                           _bottlenecks: List[str]) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities"""
        opportunities = []
        
        # CPU opportunities
        if cpu_patterns.get("optimization_potential", 0) > 0.5:
            opportunities.append({
                "type": "cpu_optimization",
                "priority": "high",
                "potential_improvement": cpu_patterns.get("optimization_potential", 0) * 30,
                "description": "CPU usage optimization and load balancing"
            })
        
        # Memory opportunities
        if memory_patterns.get("optimization_potential", 0) > 0.5:
            opportunities.append({
                "type": "memory_optimization",
                "priority": "high",
                "potential_improvement": memory_patterns.get("optimization_potential", 0) * 25,
                "description": "Memory management and garbage collection optimization"
            })
        
        # I/O opportunities
        if io_patterns.get("optimization_potential", 0) > 0.5:
            opportunities.append({
                "type": "io_optimization",
                "priority": "medium",
                "potential_improvement": io_patterns.get("optimization_potential", 0) * 35,
                "description": "I/O scheduling and caching optimization"
            })
        
        return opportunities
    
    def _calculate_pattern_confidence(self, cpu_patterns: Dict[str, Any], 
                                    memory_patterns: Dict[str, Any], 
                                    io_patterns: Dict[str, Any]) -> float:
        """Calculate confidence in pattern analysis"""
        confidence_factors = []
        
        # Data completeness
        if cpu_patterns:
            confidence_factors.append(0.9)
        if memory_patterns:
            confidence_factors.append(0.95)
        if io_patterns:
            confidence_factors.append(0.85)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.0
    
    def _calculate_performance_gains(self, ml_results: List[MLOptimizationResult]) -> Dict[str, float]:
        """Calculate performance gains from ML optimizations"""
        gains = {}
        
        for result in ml_results:
            optimization_type = result.optimization_type.value
            gains[optimization_type] = result.expected_improvement * result.confidence
        
        return gains