"""
Test Coverage Quality Gate for MCP Swarm Intelligence Server

This module implements the test coverage quality gate that ensures 
comprehensive test coverage for all Python code in the project.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio
import json
import logging
from dataclasses import dataclass

from mcp_swarm.quality.gate_engine import (
    BaseQualityGate, QualityGateResult, QualityGateType, QualityGateStatus
)

logger = logging.getLogger(__name__)


@dataclass
class CoverageMetrics:
    """Test coverage metrics for analysis"""
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    total_lines: int
    covered_lines: int
    missing_lines: List[int]
    excluded_lines: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_coverage": self.line_coverage,
            "branch_coverage": self.branch_coverage,
            "function_coverage": self.function_coverage,
            "total_lines": self.total_lines,
            "covered_lines": self.covered_lines,
            "missing_lines": self.missing_lines,
            "excluded_lines": self.excluded_lines
        }


@dataclass
class TestResults:
    """Test execution results"""
    tests_run: int
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    test_duration: float
    test_files: List[str]
    failure_details: List[Dict[str, Any]]
    
    @property
    def success_rate(self) -> float:
        """Calculate test success rate"""
        return self.tests_passed / self.tests_run if self.tests_run > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "test_duration": self.test_duration,
            "success_rate": self.success_rate,
            "test_files": self.test_files,
            "failure_details": self.failure_details
        }


@dataclass
class CoverageAnalysis:
    """Comprehensive coverage analysis results"""
    overall_coverage: float
    file_coverage: Dict[str, CoverageMetrics]
    package_coverage: Dict[str, float]
    uncovered_files: List[str]
    critical_gaps: List[str]
    coverage_trend: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_coverage": self.overall_coverage,
            "file_coverage": {file: metrics.to_dict() for file, metrics in self.file_coverage.items()},
            "package_coverage": self.package_coverage,
            "uncovered_files": self.uncovered_files,
            "critical_gaps": self.critical_gaps,
            "coverage_trend": self.coverage_trend
        }


class CoverageAnalyzer:
    """Analyzer for test coverage data"""
    
    def __init__(self):
        self.coverage_db_path = None
        
    async def analyze_coverage_data(
        self, 
        coverage_file: Path
    ) -> CoverageAnalysis:
        """Analyze coverage data from coverage file"""
        
        if not coverage_file.exists():
            raise ValueError(f"Coverage file not found: {coverage_file}")
        
        # Parse coverage.xml file
        coverage_data = await self._parse_coverage_xml(coverage_file)
        
        # Analyze file-level coverage
        file_coverage = {}
        package_coverage = {}
        uncovered_files = []
        critical_gaps = []
        
        overall_line_rate = coverage_data.get("line-rate", 0.0)
        
        # Process packages
        for package in coverage_data.get("packages", []):
            package_name = package.get("name", "unknown")
            package_line_rate = package.get("line-rate", 0.0)
            package_coverage[package_name] = package_line_rate
            
            # Process classes/files in package
            for cls in package.get("classes", []):
                filename = cls.get("filename", "")
                if filename:
                    line_rate = cls.get("line-rate", 0.0)
                    branch_rate = cls.get("branch-rate", 0.0)
                    
                    # Extract line coverage details
                    lines = cls.get("lines", [])
                    total_lines = len(lines)
                    covered_lines = len([line for line in lines if line.get("hits", 0) > 0])
                    missing_lines = [line.get("number", 0) for line in lines if line.get("hits", 0) == 0]
                    
                    file_coverage[filename] = CoverageMetrics(
                        line_coverage=line_rate,
                        branch_coverage=branch_rate,
                        function_coverage=0.0,  # Would need additional parsing
                        total_lines=total_lines,
                        covered_lines=covered_lines,
                        missing_lines=missing_lines,
                        excluded_lines=[]
                    )
                    
                    # Identify critical gaps (low coverage files)
                    if line_rate < 0.8 and total_lines > 10:
                        critical_gaps.append(f"{filename}: {line_rate:.1%} coverage")
                    
                    # Identify completely uncovered files
                    if line_rate == 0.0:
                        uncovered_files.append(filename)
        
        return CoverageAnalysis(
            overall_coverage=overall_line_rate,
            file_coverage=file_coverage,
            package_coverage=package_coverage,
            uncovered_files=uncovered_files,
            critical_gaps=critical_gaps
        )
    
    async def _parse_coverage_xml(self, coverage_file: Path) -> Dict[str, Any]:
        """Parse coverage.xml file into structured data"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            
            # Extract overall coverage metrics
            coverage_data = {
                "line-rate": float(root.get("line-rate", 0.0)),
                "branch-rate": float(root.get("branch-rate", 0.0)),
                "lines-covered": int(root.get("lines-covered", 0)),
                "lines-valid": int(root.get("lines-valid", 0)),
                "branches-covered": int(root.get("branches-covered", 0)),
                "branches-valid": int(root.get("branches-valid", 0)),
                "packages": []
            }
            
            # Parse packages
            packages_elem = root.find("packages")
            if packages_elem is not None:
                for package_elem in packages_elem.findall("package"):
                    package_data = {
                        "name": package_elem.get("name", ""),
                        "line-rate": float(package_elem.get("line-rate", 0.0)),
                        "branch-rate": float(package_elem.get("branch-rate", 0.0)),
                        "classes": []
                    }
                    
                    # Parse classes (files)
                    classes_elem = package_elem.find("classes")
                    if classes_elem is not None:
                        for class_elem in classes_elem.findall("class"):
                            class_data = {
                                "name": class_elem.get("name", ""),
                                "filename": class_elem.get("filename", ""),
                                "line-rate": float(class_elem.get("line-rate", 0.0)),
                                "branch-rate": float(class_elem.get("branch-rate", 0.0)),
                                "lines": []
                            }
                            
                            # Parse lines
                            lines_elem = class_elem.find("lines")
                            if lines_elem is not None:
                                for line_elem in lines_elem.findall("line"):
                                    line_data = {
                                        "number": int(line_elem.get("number", 0)),
                                        "hits": int(line_elem.get("hits", 0)),
                                        "branch": line_elem.get("branch", "false") == "true"
                                    }
                                    class_data["lines"].append(line_data)
                            
                            package_data["classes"].append(class_data)
                    
                    coverage_data["packages"].append(package_data)
            
            return coverage_data
            
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("Error parsing coverage XML: %s", e)
            return {}


class TestCoverageGate(BaseQualityGate):
    """Quality gate for test coverage validation"""
    
    def __init__(self, minimum_coverage: float = 0.95, timeout: float = 600.0):
        super().__init__("test_coverage", timeout)
        self.minimum_coverage = minimum_coverage
        self.coverage_analyzer = CoverageAnalyzer()
        
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute test coverage quality gate"""
        code_path = Path(context["code_path"])
        fail_fast = context.get("fail_fast", False)
        
        try:
            # Run test suite with coverage
            test_results = await self._run_test_suite(code_path, fail_fast)
            
            # Analyze coverage results
            coverage_file = code_path / "coverage.xml"
            if not coverage_file.exists():
                return QualityGateResult(
                    gate_type=QualityGateType.TEST_COVERAGE,
                    status=QualityGateStatus.FAILED,
                    score=0.0,
                    error_message="No coverage file generated",
                    recommendations=["Ensure pytest-cov is installed and configured",
                                   "Run tests with coverage: pytest --cov=src --cov-report=xml"]
                )
            
            coverage_analysis = await self.coverage_analyzer.analyze_coverage_data(
                coverage_file
            )
            
            # Generate recommendations
            recommendations = self._generate_coverage_recommendations(
                coverage_analysis, test_results
            )
            
            # Determine status and score
            coverage_score = coverage_analysis.overall_coverage
            status = self._determine_status(coverage_score, test_results)
            
            # Prepare detailed results
            details = {
                "coverage_analysis": coverage_analysis.to_dict(),
                "test_results": test_results.to_dict(),
                "thresholds": {
                    "minimum_coverage": self.minimum_coverage,
                    "target_coverage": 0.95
                }
            }
            
            return QualityGateResult(
                gate_type=QualityGateType.TEST_COVERAGE,
                status=status,
                score=coverage_score,
                details=details,
                recommendations=recommendations
            )
            
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("Error executing test coverage gate: %s", e)
            return QualityGateResult(
                gate_type=QualityGateType.TEST_COVERAGE,
                status=QualityGateStatus.FAILED,
                score=0.0,
                error_message=str(e),
                recommendations=["Check test configuration and dependencies",
                               "Ensure pytest is properly installed",
                               "Verify test discovery patterns"]
            )
    
    async def _run_test_suite(self, test_path: Path, fail_fast: bool = False) -> TestResults:
        """Run comprehensive test suite with coverage"""
        
        # Prepare pytest command
        cmd = [
            "python", "-m", "pytest",
            str(test_path),
            "--cov=src",
            "--cov-report=xml",
            "--cov-report=term-missing",
            "--verbose",
            "--tb=short"
        ]
        
        if fail_fast:
            cmd.append("--maxfail=1")
        
        # Add JSON output for easier parsing
        json_report = test_path / "test_results.json"
        cmd.extend(["--json-report", f"--json-report-file={json_report}"])
        
        logger.info("Running test suite: %s", " ".join(cmd))
        
        # Run tests
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=test_path
        )
        
        stdout, stderr = await process.communicate()
        
        # Parse results
        test_results = await self._parse_test_results(
            json_report if json_report.exists() else None,
            stdout.decode(),
            stderr.decode(),
            process.returncode or 0
        )
        
        return test_results
    
    async def _parse_test_results(
        self,
        json_report_path: Optional[Path],
        stdout: str,
        stderr: str,
        return_code: int
    ) -> TestResults:
        """Parse test results from pytest output"""
        
        # Try to parse JSON report first
        if json_report_path and json_report_path.exists():
            try:
                with open(json_report_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                summary = json_data.get("summary", {})
                tests = json_data.get("tests", [])
                
                test_files = list(set(test.get("nodeid", "").split("::")[0] for test in tests))
                failure_details = [
                    {
                        "test": test.get("nodeid", ""),
                        "outcome": test.get("outcome", ""),
                        "message": test.get("call", {}).get("longrepr", "")
                    }
                    for test in tests if test.get("outcome") in ["failed", "error"]
                ]
                
                return TestResults(
                    tests_run=summary.get("total", 0),
                    tests_passed=summary.get("passed", 0),
                    tests_failed=summary.get("failed", 0),
                    tests_skipped=summary.get("skipped", 0),
                    test_duration=json_data.get("duration", 0.0),
                    test_files=test_files,
                    failure_details=failure_details
                )
                
            except (ValueError, RuntimeError, OSError) as e:
                logger.warning("Failed to parse JSON test report: %s", e)
        
        # Fallback to parsing stdout
        return self._parse_stdout_results(stdout, stderr, return_code)
    
    def _parse_stdout_results(self, stdout: str, _stderr: str, _return_code: int) -> TestResults:
        """Parse test results from stdout when JSON report is not available"""
        
        # Basic parsing of pytest output
        lines = stdout.split('\n')
        
        tests_run = 0
        tests_passed = 0
        tests_failed = 0
        tests_skipped = 0
        test_duration = 0.0
        
        # Look for summary line like: "5 passed, 2 failed, 1 skipped in 10.5s"
        for line in lines:
            if "passed" in line and "in" in line and "s" in line:
                # Extract numbers from summary line
                import re
                passed_match = re.search(r'(\d+) passed', line)
                failed_match = re.search(r'(\d+) failed', line)
                skipped_match = re.search(r'(\d+) skipped', line)
                duration_match = re.search(r'in ([\d.]+)s', line)
                
                if passed_match:
                    tests_passed = int(passed_match.group(1))
                if failed_match:
                    tests_failed = int(failed_match.group(1))
                if skipped_match:
                    tests_skipped = int(skipped_match.group(1))
                if duration_match:
                    test_duration = float(duration_match.group(1))
                
                tests_run = tests_passed + tests_failed + tests_skipped
                break
        
        return TestResults(
            tests_run=tests_run,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_skipped=tests_skipped,
            test_duration=test_duration,
            test_files=[],
            failure_details=[]
        )
    
    def _determine_status(
        self, 
        coverage_score: float, 
        test_results: TestResults
    ) -> QualityGateStatus:
        """Determine overall status based on coverage and test results"""
        
        # Check if tests failed
        if test_results.tests_failed > 0:
            return QualityGateStatus.FAILED
        
        # Check coverage threshold
        if coverage_score < self.minimum_coverage:
            if coverage_score < self.minimum_coverage * 0.9:  # More than 10% below target
                return QualityGateStatus.FAILED
            else:
                return QualityGateStatus.WARNING
        
        # Check if there are no tests at all
        if test_results.tests_run == 0:
            return QualityGateStatus.FAILED
        
        return QualityGateStatus.PASSED
    
    def _generate_coverage_recommendations(
        self, 
        coverage: CoverageAnalysis,
        test_results: TestResults
    ) -> List[str]:
        """Generate recommendations for improving coverage"""
        recommendations = []
        
        # Coverage-specific recommendations
        if coverage.overall_coverage < self.minimum_coverage:
            recommendations.append(
                f"Increase test coverage from {coverage.overall_coverage:.1%} to at least {self.minimum_coverage:.1%}"
            )
        
        # File-specific recommendations
        if coverage.uncovered_files:
            recommendations.append(
                f"Add tests for {len(coverage.uncovered_files)} completely uncovered files"
            )
        
        if coverage.critical_gaps:
            recommendations.append(
                f"Focus on {len(coverage.critical_gaps)} files with critical coverage gaps"
            )
            
            # Add specific files if not too many
            if len(coverage.critical_gaps) <= 5:
                for gap in coverage.critical_gaps:
                    recommendations.append(f"  • {gap}")
        
        # Test failure recommendations
        if test_results.tests_failed > 0:
            recommendations.append(f"Fix {test_results.tests_failed} failing tests")
        
        # No tests recommendations
        if test_results.tests_run == 0:
            recommendations.append("Add test cases - no tests were found or executed")
        
        # Performance recommendations
        if test_results.test_duration > 300:  # 5 minutes
            recommendations.append("Consider optimizing test performance - tests taking too long")
        
        return recommendations