"""
Quality Gate Engine for MCP Swarm Intelligence Server

This module implements the core quality gate execution engine that manages
and coordinates various quality checks including test coverage, security
scanning, performance benchmarking, and documentation validation.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import asyncio
import time
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class QualityGateType(Enum):
    """Types of quality gates available in the system"""
    CODE_QUALITY = "code_quality"
    TEST_COVERAGE = "test_coverage"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    DOCUMENTATION_CHECK = "documentation_check"
    DEPLOYMENT_READINESS = "deployment_readiness"


class QualityGateStatus(Enum):
    """Status values for quality gate execution results"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class QualityGateResult:
    """Result of executing a single quality gate"""
    gate_type: QualityGateType
    status: QualityGateStatus
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "gate_type": self.gate_type.value,
            "status": self.status.value,
            "score": self.score,
            "details": self.details,
            "recommendations": self.recommendations,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class QualityGateResults:
    """Results from executing multiple quality gates"""
    gate_results: List[QualityGateResult] = field(default_factory=list)
    overall_status: QualityGateStatus = QualityGateStatus.PASSED
    total_execution_time: float = 0.0
    execution_timestamp: datetime = field(default_factory=datetime.utcnow)

    def calculate_overall_score(self) -> float:
        """Calculate weighted overall score from all gates"""
        if not self.gate_results:
            return 0.0
        
        # Weight different gate types based on importance
        weights = {
            QualityGateType.TEST_COVERAGE: 0.25,
            QualityGateType.SECURITY_SCAN: 0.25,
            QualityGateType.CODE_QUALITY: 0.20,
            QualityGateType.PERFORMANCE_BENCHMARK: 0.15,
            QualityGateType.DOCUMENTATION_CHECK: 0.10,
            QualityGateType.DEPLOYMENT_READINESS: 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in self.gate_results:
            weight = weights.get(result.gate_type, 0.1)
            weighted_score += result.score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0

    def generate_recommendations(self) -> List[str]:
        """Generate consolidated recommendations from all gates"""
        all_recommendations = []
        for result in self.gate_results:
            if result.status in [QualityGateStatus.FAILED, QualityGateStatus.WARNING]:
                all_recommendations.extend(result.recommendations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization"""
        return {
            "gate_results": [result.to_dict() for result in self.gate_results],
            "overall_status": self.overall_status.value,
            "overall_score": self.calculate_overall_score(),
            "total_execution_time": self.total_execution_time,
            "execution_timestamp": self.execution_timestamp.isoformat(),
            "summary": {
                "total_gates": len(self.gate_results),
                "passed": len([r for r in self.gate_results if r.status == QualityGateStatus.PASSED]),
                "failed": len([r for r in self.gate_results if r.status == QualityGateStatus.FAILED]),
                "warnings": len([r for r in self.gate_results if r.status == QualityGateStatus.WARNING]),
                "skipped": len([r for r in self.gate_results if r.status == QualityGateStatus.SKIPPED])
            }
        }


class BaseQualityGate:
    """Abstract base class for all quality gates"""
    
    def __init__(self, name: str, timeout: float = 300.0):
        self.name = name
        self.timeout = timeout
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute the quality gate and return results"""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def validate_context(self, context: Dict[str, Any]) -> bool:
        """Validate that context contains required parameters"""
        return "code_path" in context


class QualityGateEngine:
    """Main engine for executing quality gates"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.registered_gates: Dict[QualityGateType, BaseQualityGate] = {}
        self.gate_thresholds = self._load_thresholds(config_path)
        self.execution_history: List[QualityGateResults] = []
        self.concurrent_execution = True
        
    def _load_thresholds(self, config_path: Optional[Path]) -> Dict[QualityGateType, Dict[str, Any]]:
        """Load quality gate thresholds from configuration"""
        # Default thresholds - can be overridden by config file
        default_thresholds = {
            QualityGateType.TEST_COVERAGE: {
                "minimum_coverage": 0.95,
                "fail_on_decrease": True,
                "coverage_threshold": 0.90
            },
            QualityGateType.SECURITY_SCAN: {
                "max_high_vulnerabilities": 0,
                "max_medium_vulnerabilities": 5,
                "fail_on_critical": True
            },
            QualityGateType.PERFORMANCE_BENCHMARK: {
                "max_regression_percent": 0.05,
                "timeout_threshold": 30.0,
                "memory_threshold_mb": 1000
            },
            QualityGateType.DOCUMENTATION_CHECK: {
                "minimum_api_coverage": 0.95,
                "require_examples": True,
                "check_spelling": True
            },
            QualityGateType.CODE_QUALITY: {
                "max_complexity": 10,
                "min_maintainability": 7.0,
                "enforce_style": True
            }
        }
        
        if config_path and config_path.exists():
            logger.info("Using default thresholds (config loading not implemented yet)")
        
        return default_thresholds

    async def register_quality_gate(
        self, 
        gate_type: QualityGateType, 
        gate_impl: BaseQualityGate
    ) -> None:
        """Register a quality gate for execution"""
        if not isinstance(gate_impl, BaseQualityGate):
            raise ValueError("Gate implementation must inherit from BaseQualityGate")
        
        self.registered_gates[gate_type] = gate_impl
        logger.info("Registered quality gate: %s", gate_type.value)

    async def execute_quality_gates(
        self, 
        gate_types: List[QualityGateType],
        context: Dict[str, Any],
        fail_fast: bool = False
    ) -> QualityGateResults:
        """Execute specified quality gates"""
        start_time = time.time()
        results = QualityGateResults()
        
        # Validate that all requested gates are registered
        missing_gates = [gt for gt in gate_types if gt not in self.registered_gates]
        if missing_gates:
            logger.error("Missing gate implementations: %s", [gt.value for gt in missing_gates])
            for gt in missing_gates:
                results.gate_results.append(QualityGateResult(
                    gate_type=gt,
                    status=QualityGateStatus.SKIPPED,
                    score=0.0,
                    error_message="Gate implementation not registered"
                ))
        
        # Execute registered gates
        available_gates = [gt for gt in gate_types if gt in self.registered_gates]
        
        if self.concurrent_execution and not fail_fast:
            # Execute gates concurrently
            gate_tasks = [
                self._execute_single_gate(gate_type, context)
                for gate_type in available_gates
            ]
            gate_results = await asyncio.gather(*gate_tasks, return_exceptions=True)
            
            for result in gate_results:
                if isinstance(result, Exception):
                    logger.error("Gate execution error: %s", result)
                    results.gate_results.append(QualityGateResult(
                        gate_type=QualityGateType.CODE_QUALITY,  # Default fallback
                        status=QualityGateStatus.FAILED,
                        score=0.0,
                        error_message=str(result)
                    ))
                elif isinstance(result, QualityGateResult):
                    results.gate_results.append(result)
        else:
            # Execute gates sequentially
            for gate_type in available_gates:
                try:
                    result = await self._execute_single_gate(gate_type, context)
                    results.gate_results.append(result)
                    
                    # Fail fast if requested and gate failed
                    if fail_fast and result.status == QualityGateStatus.FAILED:
                        logger.warning("Failing fast due to %s failure", gate_type.value)
                        break
                        
                except (asyncio.TimeoutError, ValueError) as e:
                    logger.error("Error executing %s: %s", gate_type.value, e)
                    results.gate_results.append(QualityGateResult(
                        gate_type=gate_type,
                        status=QualityGateStatus.FAILED,
                        score=0.0,
                        error_message=str(e)
                    ))
                    
                    if fail_fast:
                        break

        # Calculate overall status and execution time
        results.total_execution_time = time.time() - start_time
        results.overall_status = self._determine_overall_status(results.gate_results)
        
        # Store in execution history
        self.execution_history.append(results)
        
        logger.info("Quality gates execution completed: %s (%d gates in %.2fs)", 
                   results.overall_status.value, len(results.gate_results), 
                   results.total_execution_time)
        
        return results

    async def execute_all_gates(
        self, 
        context: Dict[str, Any],
        fail_fast: bool = False
    ) -> QualityGateResults:
        """Execute all registered quality gates"""
        return await self.execute_quality_gates(
            list(self.registered_gates.keys()), 
            context, 
            fail_fast
        )

    async def _execute_single_gate(
        self, 
        gate_type: QualityGateType, 
        context: Dict[str, Any]
    ) -> QualityGateResult:
        """Execute a single quality gate with error handling"""
        gate = self.registered_gates[gate_type]
        start_time = time.time()
        
        try:
            # Validate context
            if not gate.validate_context(context):
                return QualityGateResult(
                    gate_type=gate_type,
                    status=QualityGateStatus.FAILED,
                    score=0.0,
                    error_message="Invalid context for gate execution"
                )
            
            # Execute gate with timeout
            result = await asyncio.wait_for(
                gate.execute(context), 
                timeout=gate.timeout
            )
            
            result.execution_time = time.time() - start_time
            logger.info("Gate %s completed: %s (score: %.2f)", 
                       gate_type.value, result.status.value, result.score)
            
            return result
            
        except asyncio.TimeoutError:
            return QualityGateResult(
                gate_type=gate_type,
                status=QualityGateStatus.FAILED,
                score=0.0,
                execution_time=time.time() - start_time,
                error_message=f"Gate execution timed out after {gate.timeout}s"
            )
        except (ValueError, RuntimeError) as e:
            logger.error("Error executing %s: %s", gate_type.value, e)
            return QualityGateResult(
                gate_type=gate_type,
                status=QualityGateStatus.FAILED,
                score=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    def _determine_overall_status(
        self, 
        results: List[QualityGateResult]
    ) -> QualityGateStatus:
        """Determine overall quality gate status from individual results"""
        if not results:
            return QualityGateStatus.SKIPPED
        
        status_counts = {
            QualityGateStatus.FAILED: 0,
            QualityGateStatus.WARNING: 0,
            QualityGateStatus.PASSED: 0,
            QualityGateStatus.SKIPPED: 0
        }
        
        for result in results:
            status_counts[result.status] += 1
        
        # Determine overall status based on priority
        if status_counts[QualityGateStatus.FAILED] > 0:
            return QualityGateStatus.FAILED
        elif status_counts[QualityGateStatus.WARNING] > 0:
            return QualityGateStatus.WARNING
        elif status_counts[QualityGateStatus.PASSED] > 0:
            return QualityGateStatus.PASSED
        else:
            return QualityGateStatus.SKIPPED

    def get_execution_summary(self, limit: int = 10) -> Dict[str, Any]:
        """Get summary of recent quality gate executions"""
        recent_executions = self.execution_history[-limit:]
        
        return {
            "total_executions": len(self.execution_history),
            "recent_executions": limit,
            "success_rate": len([ex for ex in recent_executions 
                               if ex.overall_status == QualityGateStatus.PASSED]) / len(recent_executions)
                               if recent_executions else 0.0,
            "average_execution_time": sum(ex.total_execution_time for ex in recent_executions) / len(recent_executions)
                                    if recent_executions else 0.0,
            "registered_gates": [gt.value for gt in self.registered_gates.keys()],
            "recent_results": [ex.to_dict() for ex in recent_executions]
        }