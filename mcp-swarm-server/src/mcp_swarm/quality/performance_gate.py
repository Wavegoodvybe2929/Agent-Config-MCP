"""
Performance Benchmark Quality Gate for MCP Swarm Intelligence Server

This module implements the performance benchmarking quality gate that monitors
performance metrics and detects regressions in the codebase.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio
import time
import json
import logging
from dataclasses import dataclass
import statistics

from mcp_swarm.quality.gate_engine import (
    BaseQualityGate, QualityGateResult, QualityGateType, QualityGateStatus
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetric:
    """Individual benchmark metric"""
    name: str
    value: float
    unit: str
    baseline: Optional[float] = None
    threshold: Optional[float] = None
    
    @property
    def regression_percent(self) -> Optional[float]:
        """Calculate regression percentage compared to baseline"""
        if self.baseline is None or self.baseline == 0:
            return None
        return ((self.value - self.baseline) / self.baseline) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "baseline": self.baseline,
            "threshold": self.threshold,
            "regression_percent": self.regression_percent
        }


@dataclass
class BenchmarkResults:
    """Results from performance benchmarking"""
    overall_score: float
    execution_time: float
    memory_usage: float
    throughput: float
    latency: float
    custom_metrics: List[BenchmarkMetric]
    baseline_comparison: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "throughput": self.throughput,
            "latency": self.latency,
            "custom_metrics": [metric.to_dict() for metric in self.custom_metrics],
            "baseline_comparison": self.baseline_comparison
        }


@dataclass
class PerformanceRegression:
    """Detected performance regression"""
    metric_name: str
    current_value: float
    baseline_value: float
    regression_percent: float
    severity: str
    impact_description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "regression_percent": self.regression_percent,
            "severity": self.severity,
            "impact_description": self.impact_description
        }


class PerformanceBenchmarkSuite:
    """Suite of performance benchmarks for the MCP Swarm system"""
    
    def __init__(self):
        self.benchmark_data_file = "performance_baseline.json"
        
    async def run_benchmarks(self, code_path: Path) -> BenchmarkResults:
        """Run comprehensive performance benchmarks"""
        
        # Run different types of benchmarks
        execution_time = await self._benchmark_execution_time(code_path)
        memory_usage = await self._benchmark_memory_usage(code_path)
        throughput = await self._benchmark_throughput(code_path)
        latency = await self._benchmark_latency(code_path)
        custom_metrics = await self._run_custom_benchmarks(code_path)
        
        # Load baseline for comparison
        baseline_data = await self._load_baseline_data(code_path)
        baseline_comparison = self._compare_with_baseline(
            {
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "throughput": throughput,
                "latency": latency
            },
            baseline_data
        )
        
        # Calculate overall performance score
        overall_score = self._calculate_overall_score(
            execution_time, memory_usage, throughput, latency, baseline_comparison
        )
        
        return BenchmarkResults(
            overall_score=overall_score,
            execution_time=execution_time,
            memory_usage=memory_usage,
            throughput=throughput,
            latency=latency,
            custom_metrics=custom_metrics,
            baseline_comparison=baseline_comparison
        )
    
    async def _benchmark_execution_time(self, code_path: Path) -> float:
        """Benchmark overall execution time for key operations"""
        
        # Test import time
        import_start = time.time()
        try:
            # Simulate importing the main modules
            cmd = ["python", "-c", "import sys; sys.path.insert(0, 'src'); import mcp_swarm"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=code_path
            )
            await process.communicate()
            import_time = time.time() - import_start
        except (OSError, ValueError, RuntimeError):
            import_time = 0.0
        
        # Test basic operation execution time
        operation_start = time.time()
        try:
            # Run a simple test to measure basic operations
            cmd = [
                "python", "-c", 
                "import sys; sys.path.insert(0, 'src'); "
                "from mcp_swarm.quality.gate_engine import QualityGateEngine; "
                "engine = QualityGateEngine(); "
                "print('Basic operations completed')"
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=code_path
            )
            await process.communicate()
            operation_time = time.time() - operation_start
        except (OSError, ValueError, RuntimeError):
            operation_time = 0.0
        
        return max(import_time + operation_time, 0.001)  # Avoid zero division
    
    async def _benchmark_memory_usage(self, code_path: Path) -> float:
        """Benchmark memory usage of key operations"""
        
        try:
            # Use memory_profiler if available
            cmd = [
                "python", "-c",
                "import tracemalloc; "
                "tracemalloc.start(); "
                "import sys; sys.path.insert(0, 'src'); "
                "import mcp_swarm; "
                "current, peak = tracemalloc.get_traced_memory(); "
                "tracemalloc.stop(); "
                "print(f'{peak / 1024 / 1024:.2f}')"  # MB
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=code_path
            )
            
            stdout, _ = await process.communicate()
            
            if process.returncode == 0:
                memory_mb = float(stdout.decode().strip())
                return memory_mb
                
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Memory benchmark failed: %s", e)
        
        return 50.0  # Default estimate in MB
    
    async def _benchmark_throughput(self, code_path: Path) -> float:
        """Benchmark throughput for typical operations"""
        
        operations_count = 100
        start_time = time.time()
        
        try:
            # Test simple operation throughput
            cmd = [
                "python", "-c",
                f"import sys; sys.path.insert(0, 'src'); "
                f"from mcp_swarm.quality.gate_engine import QualityGateEngine; "
                f"engine = QualityGateEngine(); "
                f"for i in range({operations_count}): "
                f"    summary = engine.get_execution_summary(1); "
                f"print('Throughput test completed')"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=code_path
            )
            
            await process.communicate()
            
            elapsed_time = time.time() - start_time
            throughput = operations_count / elapsed_time if elapsed_time > 0 else 0
            
            return throughput
            
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Throughput benchmark failed: %s", e)
            return 50.0  # Default operations per second
    
    async def _benchmark_latency(self, code_path: Path) -> float:
        """Benchmark latency for single operations"""
        
        latencies = []
        
        try:
            for _ in range(10):  # Sample 10 operations
                start_time = time.time()
                
                cmd = [
                    "python", "-c",
                    "import sys; sys.path.insert(0, 'src'); "
                    "from mcp_swarm.quality.gate_engine import QualityGateEngine; "
                    "engine = QualityGateEngine(); "
                    "summary = engine.get_execution_summary(1)"
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=code_path
                )
                
                await process.communicate()
                
                latency = (time.time() - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)
            
            return statistics.mean(latencies) if latencies else 100.0
            
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Latency benchmark failed: %s", e)
            return 100.0  # Default 100ms
    
    async def _run_custom_benchmarks(self, code_path: Path) -> List[BenchmarkMetric]:
        """Run custom benchmarks specific to MCP Swarm functionality"""
        
        custom_metrics = []
        
        # Benchmark quality gate engine initialization
        try:
            start_time = time.time()
            cmd = [
                "python", "-c",
                "import sys; sys.path.insert(0, 'src'); "
                "from mcp_swarm.quality.gate_engine import QualityGateEngine; "
                "engine = QualityGateEngine()"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=code_path
            )
            
            await process.communicate()
            init_time = (time.time() - start_time) * 1000
            
            custom_metrics.append(BenchmarkMetric(
                name="quality_gate_init_time",
                value=init_time,
                unit="ms",
                threshold=500.0  # 500ms threshold
            ))
            
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Quality gate benchmark failed: %s", e)
        
        # Add more custom benchmarks as needed
        
        return custom_metrics
    
    async def _load_baseline_data(self, code_path: Path) -> Dict[str, float]:
        """Load baseline performance data for comparison"""
        
        baseline_file = code_path / self.benchmark_data_file
        
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load baseline data: %s", e)
        
        # Return default baseline values
        return {
            "execution_time": 1.0,
            "memory_usage": 50.0,
            "throughput": 100.0,
            "latency": 50.0
        }
    
    def _compare_with_baseline(
        self, 
        current_metrics: Dict[str, float], 
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Compare current metrics with baseline"""
        
        comparison = {}
        
        for metric_name, current_value in current_metrics.items():
            baseline_value = baseline_metrics.get(metric_name)
            
            if baseline_value is not None and baseline_value > 0:
                regression_percent = ((current_value - baseline_value) / baseline_value) * 100
                comparison[metric_name] = regression_percent
            else:
                comparison[metric_name] = 0.0
        
        return comparison
    
    def _calculate_overall_score(
        self, 
        execution_time: float, 
        memory_usage: float, 
        throughput: float, 
        latency: float,
        baseline_comparison: Dict[str, float]
    ) -> float:
        """Calculate overall performance score (0.0 to 1.0)"""
        
        # Base score starts at 1.0
        base_score = 1.0
        
        # Apply penalties for poor performance
        score_adjustments = 0.0
        
        # Execution time penalty (higher is worse)
        if execution_time > 5.0:  # More than 5 seconds
            score_adjustments -= 0.2
        elif execution_time > 2.0:  # More than 2 seconds
            score_adjustments -= 0.1
        
        # Memory usage penalty (higher is worse)
        if memory_usage > 200.0:  # More than 200 MB
            score_adjustments -= 0.2
        elif memory_usage > 100.0:  # More than 100 MB
            score_adjustments -= 0.1
        
        # Throughput bonus (higher is better)
        if throughput > 200.0:
            score_adjustments += 0.1
        elif throughput < 50.0:
            score_adjustments -= 0.1
        
        # Latency penalty (higher is worse)
        if latency > 200.0:  # More than 200ms
            score_adjustments -= 0.2
        elif latency > 100.0:  # More than 100ms
            score_adjustments -= 0.1
        
        # Regression penalties
        for regression_percent in baseline_comparison.values():
            if regression_percent > 20:  # More than 20% regression
                score_adjustments -= 0.15
            elif regression_percent > 10:  # More than 10% regression
                score_adjustments -= 0.1
            elif regression_percent > 5:  # More than 5% regression
                score_adjustments -= 0.05
        
        final_score = max(0.0, min(1.0, base_score + score_adjustments))
        
        return final_score


class PerformanceBenchmarkGate(BaseQualityGate):
    """Quality gate for performance benchmarking and regression detection"""
    
    def __init__(self, max_regression_percent: float = 5.0, timeout: float = 600.0):
        super().__init__("performance_benchmark", timeout)
        self.max_regression_percent = max_regression_percent
        self.benchmark_suite = PerformanceBenchmarkSuite()
        
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute performance benchmarking quality gate"""
        code_path = Path(context["code_path"])
        
        try:
            # Run performance benchmarks
            benchmark_results = await self.benchmark_suite.run_benchmarks(code_path)
            
            # Detect performance regressions
            regressions = self._detect_performance_regressions(benchmark_results)
            
            # Generate recommendations
            recommendations = self._generate_performance_recommendations(
                benchmark_results, regressions
            )
            
            # Determine status and score
            status = self._determine_status(benchmark_results, regressions)
            score = benchmark_results.overall_score
            
            # Prepare detailed results
            details = {
                "benchmark_results": benchmark_results.to_dict(),
                "regressions": [reg.to_dict() for reg in regressions],
                "thresholds": {
                    "max_regression_percent": self.max_regression_percent,
                    "execution_time_threshold": 5.0,
                    "memory_threshold_mb": 200.0,
                    "latency_threshold_ms": 200.0
                }
            }
            
            return QualityGateResult(
                gate_type=QualityGateType.PERFORMANCE_BENCHMARK,
                status=status,
                score=score,
                details=details,
                recommendations=recommendations
            )
            
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("Error executing performance benchmark gate: %s", e)
            return QualityGateResult(
                gate_type=QualityGateType.PERFORMANCE_BENCHMARK,
                status=QualityGateStatus.FAILED,
                score=0.0,
                error_message=str(e),
                recommendations=["Check performance benchmark configuration",
                               "Ensure code can be imported and executed",
                               "Verify system resources are available"]
            )
    
    def _detect_performance_regressions(
        self, 
        results: BenchmarkResults
    ) -> List[PerformanceRegression]:
        """Detect performance regressions against baseline"""
        
        regressions = []
        
        # Check baseline comparisons
        for metric_name, regression_percent in results.baseline_comparison.items():
            if regression_percent > self.max_regression_percent:
                # Determine severity
                if regression_percent > 20:
                    severity = "critical"
                elif regression_percent > 10:
                    severity = "high"
                else:
                    severity = "medium"
                
                # Get current and baseline values
                current_value = getattr(results, metric_name, 0.0)
                baseline_value = current_value / (1 + regression_percent / 100) if regression_percent != -100 else 0.0
                
                regression = PerformanceRegression(
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    regression_percent=regression_percent,
                    severity=severity,
                    impact_description=self._get_impact_description(metric_name, regression_percent)
                )
                
                regressions.append(regression)
        
        return regressions
    
    def _get_impact_description(self, metric_name: str, regression_percent: float) -> str:
        """Get human-readable impact description for regression"""
        
        impact_descriptions = {
            "execution_time": f"Code execution is {regression_percent:.1f}% slower than baseline",
            "memory_usage": f"Memory usage increased by {regression_percent:.1f}%",
            "throughput": f"System throughput decreased by {abs(regression_percent):.1f}%",
            "latency": f"Response latency increased by {regression_percent:.1f}%"
        }
        
        return impact_descriptions.get(
            metric_name, 
            f"{metric_name} performance degraded by {regression_percent:.1f}%"
        )
    
    def _determine_status(
        self, 
        results: BenchmarkResults, 
        regressions: List[PerformanceRegression]
    ) -> QualityGateStatus:
        """Determine overall status based on benchmark results and regressions"""
        
        # Check for critical regressions
        critical_regressions = [r for r in regressions if r.severity == "critical"]
        if critical_regressions:
            return QualityGateStatus.FAILED
        
        # Check for high severity regressions
        high_regressions = [r for r in regressions if r.severity == "high"]
        if high_regressions:
            return QualityGateStatus.FAILED
        
        # Check overall performance score
        if results.overall_score < 0.5:
            return QualityGateStatus.FAILED
        elif results.overall_score < 0.7:
            return QualityGateStatus.WARNING
        
        # Check for any regressions
        if regressions:
            return QualityGateStatus.WARNING
        
        return QualityGateStatus.PASSED
    
    def _generate_performance_recommendations(
        self, 
        results: BenchmarkResults,
        regressions: List[PerformanceRegression]
    ) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        # Regression-specific recommendations
        if regressions:
            recommendations.append(f"Address {len(regressions)} performance regressions")
            
            for regression in regressions[:3]:  # Show top 3
                recommendations.append(f"  • {regression.impact_description}")
        
        # Execution time recommendations
        if results.execution_time > 5.0:
            recommendations.append("Optimize code execution time - currently taking too long")
        
        # Memory usage recommendations
        if results.memory_usage > 200.0:
            recommendations.append(f"Reduce memory usage from {results.memory_usage:.1f}MB")
        elif results.memory_usage > 100.0:
            recommendations.append("Consider memory usage optimization")
        
        # Throughput recommendations
        if results.throughput < 50.0:
            recommendations.append(f"Improve system throughput (currently {results.throughput:.1f} ops/sec)")
        
        # Latency recommendations
        if results.latency > 200.0:
            recommendations.append(f"Reduce latency from {results.latency:.1f}ms")
        elif results.latency > 100.0:
            recommendations.append("Consider latency optimization")
        
        # Overall score recommendations
        if results.overall_score < 0.7:
            recommendations.append("Overall performance needs improvement")
            recommendations.append("Profile code to identify bottlenecks")
            recommendations.append("Consider performance optimization sprint")
        
        # General recommendations
        if not recommendations:
            recommendations.extend([
                "Performance is within acceptable limits",
                "Continue monitoring for regressions",
                "Consider setting up continuous performance monitoring"
            ])
        
        return recommendations