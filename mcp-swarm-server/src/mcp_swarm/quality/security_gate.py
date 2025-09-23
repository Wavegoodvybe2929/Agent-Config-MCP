"""
Security Scanning Quality Gate for MCP Swarm Intelligence Server

This module implements the security scanning quality gate that identifies
security vulnerabilities and weaknesses in the Python codebase.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum

from mcp_swarm.quality.gate_engine import (
    BaseQualityGate, QualityGateResult, QualityGateType, QualityGateStatus
)

logger = logging.getLogger(__name__)


class VulnerabilitySeverity(Enum):
    """Severity levels for security vulnerabilities"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityVulnerability:
    """Individual security vulnerability"""
    severity: VulnerabilitySeverity
    category: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    recommendation: str
    cwe_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "recommendation": self.recommendation,
            "cwe_id": self.cwe_id
        }


@dataclass
class SecurityScanResults:
    """Results from security scanning tools"""
    total_vulnerabilities: int
    vulnerabilities_by_severity: Dict[VulnerabilitySeverity, int]
    vulnerabilities: List[SecurityVulnerability]
    files_scanned: int
    scan_duration: float
    tool_results: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_vulnerabilities": self.total_vulnerabilities,
            "vulnerabilities_by_severity": {
                sev.value: count for sev, count in self.vulnerabilities_by_severity.items()
            },
            "vulnerabilities": [vuln.to_dict() for vuln in self.vulnerabilities],
            "files_scanned": self.files_scanned,
            "scan_duration": self.scan_duration,
            "tool_results": self.tool_results
        }


@dataclass
class SecurityRiskAssessment:
    """Overall security risk assessment"""
    risk_level: str
    risk_score: float
    critical_issues: List[str]
    high_priority_fixes: List[str]
    security_trends: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "critical_issues": self.critical_issues,
            "high_priority_fixes": self.high_priority_fixes,
            "security_trends": self.security_trends
        }


class BanditScanner:
    """Security scanner using Bandit tool"""
    
    async def scan(self, code_path: Path) -> Dict[str, Any]:
        """Run Bandit security scan"""
        cmd = [
            "python", "-m", "bandit",
            "-r", str(code_path),
            "-f", "json",
            "-ll"  # Low level confidence
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 or process.returncode == 1:  # 1 indicates issues found
                result = json.loads(stdout.decode())
                return self._parse_bandit_results(result)
            else:
                logger.error("Bandit scan failed: %s", stderr.decode())
                return {"vulnerabilities": [], "error": stderr.decode()}
                
        except FileNotFoundError:
            logger.warning("Bandit not found, skipping security scan")
            return {"vulnerabilities": [], "error": "Bandit not installed"}
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("Error running Bandit: %s", e)
            return {"vulnerabilities": [], "error": str(e)}
    
    def _parse_bandit_results(self, bandit_output: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Bandit JSON output into standardized format"""
        vulnerabilities = []
        
        for result in bandit_output.get("results", []):
            severity = self._map_bandit_severity(
                result.get("issue_severity", "LOW"),
                result.get("issue_confidence", "LOW")
            )
            
            vulnerability = SecurityVulnerability(
                severity=severity,
                category=result.get("test_name", "Unknown"),
                description=result.get("issue_text", ""),
                file_path=result.get("filename", ""),
                line_number=result.get("line_number", 0),
                code_snippet=result.get("code", ""),
                recommendation=self._get_bandit_recommendation(result.get("test_id", "")),
                cwe_id=result.get("cwe", {}).get("id") if result.get("cwe") else None
            )
            vulnerabilities.append(vulnerability)
        
        return {
            "vulnerabilities": vulnerabilities,
            "metrics": bandit_output.get("metrics", {}),
            "tool": "bandit"
        }
    
    def _map_bandit_severity(self, severity: str, confidence: str) -> VulnerabilitySeverity:
        """Map Bandit severity/confidence to standard severity"""
        severity_upper = severity.upper()
        confidence_upper = confidence.upper()
        
        # High confidence issues are treated more seriously
        if confidence_upper == "HIGH":
            if severity_upper == "HIGH":
                return VulnerabilitySeverity.CRITICAL
            elif severity_upper == "MEDIUM":
                return VulnerabilitySeverity.HIGH
            else:
                return VulnerabilitySeverity.MEDIUM
        elif confidence_upper == "MEDIUM":
            if severity_upper == "HIGH":
                return VulnerabilitySeverity.HIGH
            elif severity_upper == "MEDIUM":
                return VulnerabilitySeverity.MEDIUM
            else:
                return VulnerabilitySeverity.LOW
        else:  # LOW confidence
            if severity_upper == "HIGH":
                return VulnerabilitySeverity.MEDIUM
            else:
                return VulnerabilitySeverity.LOW
    
    def _get_bandit_recommendation(self, test_id: str) -> str:
        """Get recommendation based on Bandit test ID"""
        recommendations = {
            "B101": "Avoid using assert statements for validation in production code",
            "B102": "Use subprocess with shell=False or validate inputs carefully",
            "B103": "Avoid setting file permissions that are too permissive",
            "B104": "Bind to specific interfaces instead of 0.0.0.0 when possible",
            "B105": "Use secure string formatting methods",
            "B106": "Avoid hardcoded passwords in code",
            "B107": "Validate and sanitize all user inputs",
            "B108": "Use secure temporary file creation methods",
            "B110": "Avoid try-except-pass blocks that suppress all exceptions",
            "B201": "Use secure Flask configurations",
            "B301": "Use pickle alternatives like json for data serialization",
            "B302": "Use secure marshalling libraries",
            "B303": "Use cryptographically secure hash algorithms",
            "B304": "Use secure cipher algorithms and modes",
            "B305": "Use secure cipher modes",
            "B306": "Use secure random number generators",
            "B307": "Use parameterized queries to prevent SQL injection",
            "B308": "Use secure XML parsing configuration",
            "B309": "Use secure HTTP methods and validate inputs",
            "B310": "Validate URL schemes and domains",
            "B311": "Use cryptographically secure random generators",
            "B312": "Use secure communication protocols",
            "B313": "Use secure XML parsing libraries",
            "B314": "Use secure XML processing libraries",
            "B315": "Use secure XML libraries and configurations",
            "B316": "Use secure XML parsing configurations",
            "B317": "Use secure XML processing methods",
            "B318": "Use secure XML libraries",
            "B319": "Use secure XML processing",
            "B320": "Use secure XML parsing",
            "B321": "Use secure FTP alternatives like SFTP",
            "B322": "Use secure input validation methods",
            "B323": "Use secure unverified HTTPS context alternatives",
            "B324": "Use secure hash algorithms for security-sensitive operations",
            "B325": "Use secure temporary file creation",
            "B401": "Use secure shell command execution methods",
            "B402": "Use secure import mechanisms",
            "B403": "Consider security implications of pickle usage",
            "B404": "Use secure subprocess execution",
            "B405": "Use secure import statements",
            "B406": "Use secure Linux command execution",
            "B407": "Use secure XML processing libraries",
            "B408": "Use secure XML libraries and configurations",
            "B409": "Use secure XML parsing methods",
            "B410": "Use secure XML processing",
            "B501": "Use secure SSL/TLS configurations",
            "B502": "Use secure SSL certificates",
            "B503": "Use secure SSL/TLS configurations",
            "B504": "Use secure SSL context",
            "B505": "Use secure cryptographic algorithms",
            "B506": "Use secure YAML loading methods",
            "B507": "Use secure SSH configurations",
            "B601": "Use parameterized queries or ORM to prevent shell injection",
            "B602": "Use secure subprocess execution",
            "B603": "Use secure subprocess execution methods",
            "B604": "Use secure shell command execution",
            "B605": "Use secure process start methods",
            "B606": "Use secure process execution",
            "B607": "Use absolute paths and validate command execution",
            "B608": "Use secure SQL query methods",
            "B609": "Use secure wildcard injection prevention",
            "B610": "Use secure Django configuration",
            "B611": "Use secure Django configuration",
            "B701": "Use secure Jinja2 configurations",
            "B702": "Use secure test configurations",
            "B703": "Use secure Django configurations"
        }
        
        return recommendations.get(test_id, "Review and fix the identified security issue")


class SafetyScanner:
    """Scanner for known security vulnerabilities in dependencies"""
    
    async def scan(self, code_path: Path) -> Dict[str, Any]:
        """Run Safety scan for known vulnerabilities"""
        requirements_file = code_path / "requirements.txt"
        
        if not requirements_file.exists():
            return {"vulnerabilities": [], "error": "No requirements.txt found"}
        
        cmd = [
            "python", "-m", "safety", "check",
            "-r", str(requirements_file),
            "--json"
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # No vulnerabilities found
                return {"vulnerabilities": [], "tool": "safety"}
            elif process.returncode == 255:
                # Vulnerabilities found
                result = json.loads(stdout.decode())
                return self._parse_safety_results(result)
            else:
                logger.error("Safety scan failed: %s", stderr.decode())
                return {"vulnerabilities": [], "error": stderr.decode()}
                
        except FileNotFoundError:
            logger.warning("Safety not found, skipping dependency vulnerability scan")
            return {"vulnerabilities": [], "error": "Safety not installed"}
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("Error running Safety: %s", e)
            return {"vulnerabilities": [], "error": str(e)}
    
    def _parse_safety_results(self, safety_output: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse Safety JSON output into standardized format"""
        vulnerabilities = []
        
        for vuln in safety_output:
            vulnerability = SecurityVulnerability(
                severity=VulnerabilitySeverity.HIGH,  # Safety issues are generally high priority
                category="Dependency Vulnerability",
                description=f"Vulnerable dependency: {vuln.get('package_name', 'unknown')} "
                           f"({vuln.get('installed_version', 'unknown')}) - {vuln.get('advisory', '')}",
                file_path="requirements.txt",
                line_number=0,
                code_snippet=f"{vuln.get('package_name', '')}=={vuln.get('installed_version', '')}",
                recommendation=f"Update to version {vuln.get('vulnerable_versions', 'latest')} or higher",
                cwe_id=None
            )
            vulnerabilities.append(vulnerability)
        
        return {
            "vulnerabilities": vulnerabilities,
            "tool": "safety"
        }


class SecurityScanningGate(BaseQualityGate):
    """Quality gate for security vulnerability scanning"""
    
    def __init__(self, max_high_vulnerabilities: int = 0, timeout: float = 600.0):
        super().__init__("security_scan", timeout)
        self.max_high_vulnerabilities = max_high_vulnerabilities
        self.vulnerability_scanners = {
            "bandit": BanditScanner(),
            "safety": SafetyScanner()
        }
        
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute security scanning quality gate"""
        code_path = Path(context["code_path"])
        
        try:
            # Run all security scanners
            scan_results = await self._run_vulnerability_scan(code_path)
            
            # Assess security risk
            risk_assessment = self._assess_security_risk(scan_results)
            
            # Generate recommendations
            recommendations = self._generate_security_recommendations(risk_assessment)
            
            # Determine status and score
            status = self._determine_status(scan_results, risk_assessment)
            score = self._calculate_security_score(scan_results, risk_assessment)
            
            # Prepare detailed results
            details = {
                "scan_results": scan_results.to_dict(),
                "risk_assessment": risk_assessment.to_dict(),
                "thresholds": {
                    "max_high_vulnerabilities": self.max_high_vulnerabilities,
                    "max_critical_vulnerabilities": 0
                }
            }
            
            return QualityGateResult(
                gate_type=QualityGateType.SECURITY_SCAN,
                status=status,
                score=score,
                details=details,
                recommendations=recommendations
            )
            
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("Error executing security scanning gate: %s", e)
            return QualityGateResult(
                gate_type=QualityGateType.SECURITY_SCAN,
                status=QualityGateStatus.FAILED,
                score=0.0,
                error_message=str(e),
                recommendations=["Check security scanner installation and configuration",
                               "Ensure bandit and safety are installed",
                               "Verify file permissions and access"]
            )
    
    async def _run_vulnerability_scan(self, code_path: Path) -> SecurityScanResults:
        """Run comprehensive vulnerability scanning"""
        all_vulnerabilities = []
        tool_results = {}
        files_scanned = 0
        total_scan_time = 0.0
        
        # Run all scanners concurrently
        scanner_tasks = [
            scanner.scan(code_path) 
            for scanner in self.vulnerability_scanners.values()
        ]
        
        import time
        start_time = time.time()
        results = await asyncio.gather(*scanner_tasks, return_exceptions=True)
        total_scan_time = time.time() - start_time
        
        # Process results from each scanner
        for scanner_name, result in zip(self.vulnerability_scanners.keys(), results):
            if isinstance(result, Exception):
                logger.error("Scanner %s failed: %s", scanner_name, result)
                tool_results[scanner_name] = {"error": str(result)}
            elif isinstance(result, dict):
                tool_results[scanner_name] = result
                if "vulnerabilities" in result:
                    all_vulnerabilities.extend(result["vulnerabilities"])
                if "metrics" in result and "loc" in result["metrics"]:
                    files_scanned += result["metrics"].get("loc", 0)
        
        # Aggregate vulnerability counts by severity
        vulnerabilities_by_severity = {sev: 0 for sev in VulnerabilitySeverity}
        for vuln in all_vulnerabilities:
            if isinstance(vuln, SecurityVulnerability):
                vulnerabilities_by_severity[vuln.severity] += 1
            elif isinstance(vuln, dict) and "severity" in vuln:
                # Handle dict format from scanners
                severity = VulnerabilitySeverity(vuln["severity"])
                vulnerabilities_by_severity[severity] += 1
        
        return SecurityScanResults(
            total_vulnerabilities=len(all_vulnerabilities),
            vulnerabilities_by_severity=vulnerabilities_by_severity,
            vulnerabilities=all_vulnerabilities,
            files_scanned=max(files_scanned, 1),  # Avoid division by zero
            scan_duration=total_scan_time,
            tool_results=tool_results
        )
    
    def _assess_security_risk(self, scan_results: SecurityScanResults) -> SecurityRiskAssessment:
        """Assess overall security risk from scan results"""
        critical_count = scan_results.vulnerabilities_by_severity.get(VulnerabilitySeverity.CRITICAL, 0)
        high_count = scan_results.vulnerabilities_by_severity.get(VulnerabilitySeverity.HIGH, 0)
        medium_count = scan_results.vulnerabilities_by_severity.get(VulnerabilitySeverity.MEDIUM, 0)
        
        # Calculate risk score (0-100)
        risk_score = min(100, (
            critical_count * 25 +
            high_count * 10 +
            medium_count * 3
        ))
        
        # Determine risk level
        if critical_count > 0:
            risk_level = "CRITICAL"
        elif high_count > 3:
            risk_level = "HIGH"
        elif high_count > 0 or medium_count > 5:
            risk_level = "MEDIUM"
        elif medium_count > 0:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        # Identify critical issues
        critical_issues = []
        high_priority_fixes = []
        
        for vuln in scan_results.vulnerabilities:
            if isinstance(vuln, SecurityVulnerability):
                if vuln.severity == VulnerabilitySeverity.CRITICAL:
                    critical_issues.append(f"{vuln.category}: {vuln.description}")
                elif vuln.severity == VulnerabilitySeverity.HIGH:
                    high_priority_fixes.append(f"{vuln.file_path}:{vuln.line_number} - {vuln.category}")
            elif isinstance(vuln, dict):
                if vuln.get("severity") == "critical":
                    critical_issues.append(f"{vuln.get('category', 'Unknown')}: {vuln.get('description', '')}")
                elif vuln.get("severity") == "high":
                    high_priority_fixes.append(f"{vuln.get('file_path', '')}:{vuln.get('line_number', 0)} - {vuln.get('category', '')}")
        
        return SecurityRiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            critical_issues=critical_issues[:10],  # Limit to top 10
            high_priority_fixes=high_priority_fixes[:20]  # Limit to top 20
        )
    
    def _determine_status(
        self, 
        scan_results: SecurityScanResults, 
        risk_assessment: SecurityRiskAssessment
    ) -> QualityGateStatus:
        """Determine overall status based on scan results and risk assessment"""
        
        critical_count = scan_results.vulnerabilities_by_severity.get(VulnerabilitySeverity.CRITICAL, 0)
        high_count = scan_results.vulnerabilities_by_severity.get(VulnerabilitySeverity.HIGH, 0)
        
        # Fail for critical vulnerabilities
        if critical_count > 0:
            return QualityGateStatus.FAILED
        
        # Fail if too many high severity vulnerabilities
        if high_count > self.max_high_vulnerabilities:
            return QualityGateStatus.FAILED
        
        # Warning for some high severity vulnerabilities
        if high_count > 0:
            return QualityGateStatus.WARNING
        
        # Warning for high risk scores
        if risk_assessment.risk_score > 20:
            return QualityGateStatus.WARNING
        
        return QualityGateStatus.PASSED
    
    def _calculate_security_score(
        self, 
        scan_results: SecurityScanResults, 
        _risk_assessment: SecurityRiskAssessment
    ) -> float:
        """Calculate security score (0.0 to 1.0)"""
        # Base score starts at 1.0 (perfect security)
        base_score = 1.0
        
        # Deduct points for vulnerabilities
        critical_count = scan_results.vulnerabilities_by_severity.get(VulnerabilitySeverity.CRITICAL, 0)
        high_count = scan_results.vulnerabilities_by_severity.get(VulnerabilitySeverity.HIGH, 0)
        medium_count = scan_results.vulnerabilities_by_severity.get(VulnerabilitySeverity.MEDIUM, 0)
        
        # Heavy penalty for critical and high severity
        score_deduction = (
            critical_count * 0.3 +  # 30% per critical
            high_count * 0.15 +     # 15% per high
            medium_count * 0.05     # 5% per medium
        )
        
        final_score = max(0.0, base_score - score_deduction)
        
        return final_score
    
    def _generate_security_recommendations(
        self, 
        risk_assessment: SecurityRiskAssessment
    ) -> List[str]:
        """Generate security improvement recommendations"""
        recommendations = []
        
        # Critical issues recommendations
        if risk_assessment.critical_issues:
            recommendations.append(f"URGENT: Fix {len(risk_assessment.critical_issues)} critical security vulnerabilities")
            for issue in risk_assessment.critical_issues[:3]:  # Show top 3
                recommendations.append(f"  • {issue}")
        
        # High priority recommendations
        if risk_assessment.high_priority_fixes:
            recommendations.append(f"Address {len(risk_assessment.high_priority_fixes)} high-priority security issues")
        
        # General security recommendations based on risk level
        if risk_assessment.risk_level == "CRITICAL":
            recommendations.extend([
                "Conduct immediate security review and patching",
                "Consider security code review by expert",
                "Implement security testing in CI/CD pipeline"
            ])
        elif risk_assessment.risk_level == "HIGH":
            recommendations.extend([
                "Schedule security remediation sprint",
                "Add security linting to development workflow",
                "Review and update security dependencies"
            ])
        elif risk_assessment.risk_level == "MEDIUM":
            recommendations.extend([
                "Plan security improvements for next release",
                "Regular security dependency updates",
                "Consider automated security scanning"
            ])
        
        # Always include general best practices
        if not recommendations:
            recommendations.extend([
                "Maintain current security practices",
                "Regular security dependency updates",
                "Continue security scanning in CI/CD"
            ])
        
        return recommendations