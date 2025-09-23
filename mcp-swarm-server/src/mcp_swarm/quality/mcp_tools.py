"""
MCP Tools Interface for Quality Gates

This module implements the MCP (Model Context Protocol) tool interface
for the quality gate system, allowing external systems to execute and
monitor quality gates through standardized tool calls.
"""

from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
from datetime import datetime

from mcp_swarm.quality.gate_engine import (
    QualityGateEngine, QualityGateResult, QualityGateType, QualityGateStatus
)
from mcp_swarm.quality.test_coverage_gate import TestCoverageGate
from mcp_swarm.quality.security_gate import SecurityScanningGate
from mcp_swarm.quality.performance_gate import PerformanceBenchmarkGate
from mcp_swarm.quality.documentation_gate import DocumentationValidationGate

logger = logging.getLogger(__name__)


class QualityGatesMCPTools:
    """MCP Tools interface for quality gate operations"""
    
    def __init__(self):
        self.engine = QualityGateEngine()
        # Register gates will be done async
        
    async def _register_quality_gates(self):
        """Register all available quality gates with the engine"""
        
        # Register test coverage gate
        test_coverage_gate = TestCoverageGate()
        await self.engine.register_quality_gate(QualityGateType.TEST_COVERAGE, test_coverage_gate)
        
        # Register security scanning gate
        security_gate = SecurityScanningGate()
        await self.engine.register_quality_gate(QualityGateType.SECURITY_SCAN, security_gate)
        
        # Register performance benchmark gate
        performance_gate = PerformanceBenchmarkGate()
        await self.engine.register_quality_gate(QualityGateType.PERFORMANCE_BENCHMARK, performance_gate)
        
        # Register documentation validation gate
        documentation_gate = DocumentationValidationGate()
        await self.engine.register_quality_gate(QualityGateType.DOCUMENTATION_CHECK, documentation_gate)
    
    async def execute_quality_gates(
        self,
        code_path: str,
        gate_types: Optional[List[str]] = None,
        parallel: bool = True,
        fail_fast: bool = False
    ) -> Dict[str, Any]:
        """
        Execute quality gates for the specified code path
        
        Args:
            code_path: Path to the code directory to analyze
            gate_types: Optional list of specific gate types to run (if None, runs all)
            parallel: Whether to run gates in parallel (default: True)
            fail_fast: Whether to stop execution on first failure (default: False)
            
        Returns:
            Dictionary containing execution results and summary
        """
        
        try:
            # Ensure gates are registered
            await self._register_quality_gates()
            
            # Validate code path
            if not Path(code_path).exists():
                raise ValueError(f"Code path does not exist: {code_path}")
            
            # Parse gate types if provided
            selected_gates = None
            if gate_types:
                try:
                    selected_gates = [QualityGateType(gate_type) for gate_type in gate_types]
                except ValueError as e:
                    raise ValueError(f"Invalid gate type: {e}") from e
            
            # Prepare execution context
            context = {
                "code_path": code_path,
                "timestamp": datetime.utcnow().isoformat(),
                "execution_mode": "parallel" if parallel else "sequential"
            }
            
            # Execute quality gates
            if selected_gates:
                results = await self.engine.execute_quality_gates(
                    gate_types=selected_gates,
                    context=context,
                    fail_fast=fail_fast
                )
            else:
                results = await self.engine.execute_all_gates(
                    context=context,
                    fail_fast=fail_fast
                )
            
            # Generate execution summary
            summary = self.engine.get_execution_summary(1)
            
            # Format response
            response = {
                "success": True,
                "execution_id": summary.get("execution_id"),
                "total_gates": len(results.gate_results),
                "passed_gates": len([r for r in results.gate_results if r.status == QualityGateStatus.PASSED]),
                "failed_gates": len([r for r in results.gate_results if r.status == QualityGateStatus.FAILED]),
                "warning_gates": len([r for r in results.gate_results if r.status == QualityGateStatus.WARNING]),
                "overall_score": results.calculate_overall_score(),
                "execution_time": results.total_execution_time,
                "results": [self._format_gate_result(result) for result in results.gate_results],
                "summary": summary,
                "recommendations": self._generate_overall_recommendations(results.gate_results)
            }
            
            return response
            
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("Error executing quality gates: %s", e)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_gate_status(self, gate_type: str) -> Dict[str, Any]:
        """
        Get the status and configuration of a specific quality gate
        
        Args:
            gate_type: Type of quality gate to check
            
        Returns:
            Dictionary containing gate status and configuration
        """
        
        try:
            # Validate gate type
            try:
                gate_enum = QualityGateType(gate_type)
            except ValueError as exc:
                raise ValueError(f"Invalid gate type: {gate_type}") from exc
            
            # Ensure gates are registered
            await self._register_quality_gates()
            
            # Get gate information
            gate_info = {
                "gate_type": gate_type,
                "available": gate_enum in self.engine.registered_gates,
                "configuration": self._get_gate_configuration(gate_enum),
                "last_execution": None,  # Could be enhanced to track last execution
                "supported_parameters": self._get_gate_parameters(gate_enum)
            }
            
            return {
                "success": True,
                "gate_info": gate_info
            }
            
        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "ValueError"
            }
    
    async def list_available_gates(self) -> Dict[str, Any]:
        """
        List all available quality gates and their configurations
        
        Returns:
            Dictionary containing all available gates and their details
        """
        
        # Ensure gates are registered
        await self._register_quality_gates()
        
        gates_info = []
        
        for gate_type, gate in self.engine.registered_gates.items():
            gate_info = {
                "gate_type": gate_type.value,
                "name": gate.name,
                "timeout": gate.timeout,
                "description": self._get_gate_description(gate_type),
                "configuration": self._get_gate_configuration(gate_type),
                "supported_parameters": self._get_gate_parameters(gate_type)
            }
            gates_info.append(gate_info)
        
        return {
            "success": True,
            "total_gates": len(gates_info),
            "available_gates": gates_info,
            "supported_gate_types": [gate.value for gate in QualityGateType]
        }
    
    async def validate_code_path(self, code_path: str) -> Dict[str, Any]:
        """
        Validate that a code path is suitable for quality gate execution
        
        Args:
            code_path: Path to validate
            
        Returns:
            Dictionary containing validation results
        """
        
        path_obj = Path(code_path)
        
        validation_results = {
            "path": code_path,
            "exists": path_obj.exists(),
            "is_directory": path_obj.is_dir() if path_obj.exists() else False,
            "is_readable": False,
            "has_python_files": False,
            "has_tests": False,
            "has_requirements": False,
            "estimated_analysis_time": 0,
            "issues": []
        }
        
        if not validation_results["exists"]:
            validation_results["issues"].append("Path does not exist")
            return {
                "success": False,
                "validation": validation_results
            }
        
        if not validation_results["is_directory"]:
            validation_results["issues"].append("Path is not a directory")
            return {
                "success": False,
                "validation": validation_results
            }
        
        try:
            # Check readability
            list(path_obj.iterdir())
            validation_results["is_readable"] = True
            
            # Check for Python files
            python_files = list(path_obj.rglob("*.py"))
            validation_results["has_python_files"] = len(python_files) > 0
            validation_results["python_file_count"] = len(python_files)
            
            # Check for test files
            test_files = [f for f in python_files if "test" in f.name.lower()]
            validation_results["has_tests"] = len(test_files) > 0
            validation_results["test_file_count"] = len(test_files)
            
            # Check for requirements files
            req_files = ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"]
            found_req_files = [req for req in req_files if (path_obj / req).exists()]
            validation_results["has_requirements"] = len(found_req_files) > 0
            validation_results["requirements_files"] = found_req_files
            
            # Estimate analysis time (rough calculation)
            validation_results["estimated_analysis_time"] = max(30, len(python_files) * 2)  # 2 seconds per file minimum
            
            # Check for common issues
            if not validation_results["has_python_files"]:
                validation_results["issues"].append("No Python files found")
            
            if not validation_results["has_tests"]:
                validation_results["issues"].append("No test files found - test coverage analysis may be limited")
            
            if not validation_results["has_requirements"]:
                validation_results["issues"].append("No requirements file found - dependency analysis may be limited")
            
        except (PermissionError, OSError) as e:
            validation_results["issues"].append(f"Cannot access directory: {e}")
            return {
                "success": False,
                "validation": validation_results
            }
        
        is_valid = (validation_results["is_readable"] and 
                   validation_results["has_python_files"] and
                   len(validation_results["issues"]) == 0)
        
        return {
            "success": True,
            "is_valid": is_valid,
            "validation": validation_results
        }
    
    def _format_gate_result(self, result: QualityGateResult) -> Dict[str, Any]:
        """Format a quality gate result for MCP response"""
        
        return {
            "gate_type": result.gate_type.value,
            "status": result.status.value,
            "score": result.score,
            "execution_time": result.execution_time,
            "timestamp": result.timestamp.isoformat(),
            "details": result.details,
            "recommendations": result.recommendations,
            "error_message": result.error_message
        }
    
    def _generate_overall_recommendations(self, results: List[QualityGateResult]) -> List[str]:
        """Generate overall recommendations based on all gate results"""
        
        recommendations = []
        
        # Check for failed gates
        failed_gates = [r for r in results if r.status == QualityGateStatus.FAILED]
        if failed_gates:
            recommendations.append(f"Address {len(failed_gates)} failed quality gates before proceeding")
            for result in failed_gates[:3]:  # Show first 3 failed gates
                recommendations.append(f"  • {result.gate_type.value}: {result.error_message or 'Check gate details'}")
        
        # Check for warning gates
        warning_gates = [r for r in results if r.status == QualityGateStatus.WARNING]
        if warning_gates:
            recommendations.append(f"Review {len(warning_gates)} quality gates with warnings")
        
        # Calculate overall score
        if results:
            avg_score = sum(r.score for r in results) / len(results)
            if avg_score < 0.7:
                recommendations.append(f"Overall quality score is low ({avg_score:.2f}) - focus on improvement")
            elif avg_score < 0.9:
                recommendations.append("Good quality score - consider minor improvements")
            else:
                recommendations.append("Excellent quality score - maintain current standards")
        
        # Add specific recommendations from individual gates
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        # Find most common recommendations
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        # Add top common recommendations
        common_recs = sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for rec, count in common_recs:
            if count > 1:
                recommendations.append(f"Common issue: {rec} (mentioned {count} times)")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _get_gate_configuration(self, gate_type: QualityGateType) -> Dict[str, Any]:
        """Get configuration for a specific gate type"""
        
        configurations = {
            QualityGateType.TEST_COVERAGE: {
                "min_coverage": 80.0,
                "min_branch_coverage": 75.0,
                "coverage_formats": ["xml", "json"],
                "test_frameworks": ["pytest", "unittest"]
            },
            QualityGateType.SECURITY_SCAN: {
                "tools": ["bandit", "safety"],
                "fail_on_high_severity": True,
                "max_medium_issues": 5,
                "scan_types": ["vulnerability", "dependency"]
            },
            QualityGateType.PERFORMANCE_BENCHMARK: {
                "max_regression_percent": 5.0,
                "benchmark_types": ["execution_time", "memory_usage", "throughput", "latency"],
                "baseline_comparison": True
            },
            QualityGateType.DOCUMENTATION_CHECK: {
                "min_coverage": 80.0,
                "min_quality": 0.7,
                "check_types": ["docstrings", "api_docs", "examples"]
            }
        }
        
        return configurations.get(gate_type, {})
    
    def _get_gate_parameters(self, gate_type: QualityGateType) -> List[str]:
        """Get supported parameters for a specific gate type"""
        
        parameters = {
            QualityGateType.TEST_COVERAGE: [
                "min_coverage", "min_branch_coverage", "test_command", "coverage_format"
            ],
            QualityGateType.SECURITY_SCAN: [
                "fail_on_high_severity", "max_medium_issues", "scan_dependencies", "exclude_patterns"
            ],
            QualityGateType.PERFORMANCE_BENCHMARK: [
                "max_regression_percent", "benchmark_types", "baseline_file", "timeout"
            ],
            QualityGateType.DOCUMENTATION_CHECK: [
                "min_coverage", "min_quality", "check_examples", "api_docs_required"
            ]
        }
        
        return parameters.get(gate_type, [])
    
    def _get_gate_description(self, gate_type: QualityGateType) -> str:
        """Get description for a specific gate type"""
        
        descriptions = {
            QualityGateType.TEST_COVERAGE: "Analyzes test coverage and validates minimum coverage thresholds",
            QualityGateType.SECURITY_SCAN: "Scans code for security vulnerabilities and dependency issues",
            QualityGateType.PERFORMANCE_BENCHMARK: "Benchmarks performance and detects regressions",
            QualityGateType.DOCUMENTATION_CHECK: "Validates documentation coverage and quality",
            QualityGateType.CODE_QUALITY: "Performs static code analysis and quality checks",
            QualityGateType.DEPLOYMENT_READINESS: "Validates readiness for deployment"
        }
        
        return descriptions.get(gate_type, "Quality gate for code validation")


# MCP Tool Registration Functions
# These functions can be registered with the MCP server

async def mcp_execute_quality_gates(
    code_path: str,
    gate_types: Optional[List[str]] = None,
    parallel: bool = True,
    fail_fast: bool = False
) -> Dict[str, Any]:
    """MCP Tool: Execute quality gates on code"""
    tools = QualityGatesMCPTools()
    return await tools.execute_quality_gates(code_path, gate_types, parallel, fail_fast)


async def mcp_get_gate_status(gate_type: str) -> Dict[str, Any]:
    """MCP Tool: Get status of a specific quality gate"""
    tools = QualityGatesMCPTools()
    return await tools.get_gate_status(gate_type)


async def mcp_list_available_gates() -> Dict[str, Any]:
    """MCP Tool: List all available quality gates"""
    tools = QualityGatesMCPTools()
    return await tools.list_available_gates()


async def mcp_validate_code_path(code_path: str) -> Dict[str, Any]:
    """MCP Tool: Validate code path for quality gate execution"""
    tools = QualityGatesMCPTools()
    return await tools.validate_code_path(code_path)