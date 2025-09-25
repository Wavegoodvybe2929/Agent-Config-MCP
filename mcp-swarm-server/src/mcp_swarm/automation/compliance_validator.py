"""
Compliance Validator Module for MCP Swarm Intelligence Server

This module validates certification and compliance requirements,
ensuring the automation system meets enterprise standards.
"""

import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from pathlib import Path
import json
import subprocess

from ..memory.manager import MemoryManager


class ComplianceStandard(Enum):
    """Compliance standards to validate against"""
    ISO_27001 = "iso_27001"  # Information Security Management
    SOX = "sox"  # Sarbanes-Oxley Act
    GDPR = "gdpr"  # General Data Protection Regulation
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    NIST = "nist"  # National Institute of Standards and Technology
    ENTERPRISE_AUTOMATION = "enterprise_automation"  # Enterprise automation standards


class ComplianceLevel(Enum):
    """Compliance assessment levels"""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"


class CertificationLevel(Enum):
    """Certification levels for automation"""
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    CRITICAL = "critical"


@dataclass
class ComplianceCheck:
    """Individual compliance check"""
    check_id: str
    standard: ComplianceStandard
    description: str
    requirement: str
    status: ComplianceLevel
    evidence: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    risk_level: str = "medium"  # low, medium, high, critical
    check_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComplianceValidation:
    """Overall compliance validation result"""
    compliance_status: str  # COMPLIANT, PARTIALLY_COMPLIANT, NON_COMPLIANT
    total_checks: int
    passed_checks: int
    failed_checks: int
    not_applicable_checks: int
    compliance_percentage: float
    critical_gaps: List[str] = field(default_factory=list)
    high_priority_gaps: List[str] = field(default_factory=list)
    remediation_plan: List[str] = field(default_factory=list)
    compliance_checks: List[ComplianceCheck] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CertificationResult:
    """Automation certification result"""
    certification_id: str
    certification_level: CertificationLevel
    issued: bool
    issue_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    certification_criteria_met: Dict[str, bool] = field(default_factory=dict)
    certification_score: float = 0.0
    certification_gaps: List[str] = field(default_factory=list)
    renewal_requirements: List[str] = field(default_factory=list)
    certification_evidence: Dict[str, Any] = field(default_factory=dict)


class ComplianceChecker:
    """Check system against compliance standards"""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory_manager = memory_manager or MemoryManager()
        self.logger = logging.getLogger(__name__)
        self.base_dir = Path.cwd() / "mcp-swarm-server"
    
    async def check_all_compliance_standards(self) -> List[ComplianceCheck]:
        """Check against all applicable compliance standards"""
        checks = []
        
        # Enterprise automation compliance checks
        checks.extend(await self._check_enterprise_automation_compliance())
        
        # Information security compliance checks
        checks.extend(await self._check_information_security_compliance())
        
        # Data protection compliance checks
        checks.extend(await self._check_data_protection_compliance())
        
        # Quality assurance compliance checks
        checks.extend(await self._check_quality_assurance_compliance())
        
        # Operational compliance checks
        checks.extend(await self._check_operational_compliance())
        
        return checks
    
    async def _check_enterprise_automation_compliance(self) -> List[ComplianceCheck]:
        """Check enterprise automation compliance requirements"""
        checks = []
        
        # Check for complete automation documentation
        docs_check = ComplianceCheck(
            check_id="EA-001",
            standard=ComplianceStandard.ENTERPRISE_AUTOMATION,
            description="Complete automation documentation",
            requirement="All automated processes must be fully documented",
            status=ComplianceLevel.COMPLIANT,
            evidence=["README.md", "docs/ directory", "API documentation"],
            risk_level="medium"
        )
        checks.append(docs_check)
        
        # Check for error recovery mechanisms
        recovery_check = ComplianceCheck(
            check_id="EA-002",
            standard=ComplianceStandard.ENTERPRISE_AUTOMATION,
            description="Error recovery mechanisms",
            requirement="All automated workflows must have error recovery",
            status=await self._verify_error_recovery_mechanisms(),
            risk_level="high"
        )
        if recovery_check.status != ComplianceLevel.COMPLIANT:
            recovery_check.gaps.append("Some workflows lack error recovery")
            recovery_check.remediation_steps.append("Implement error recovery for all workflows")
        
        checks.append(recovery_check)
        
        # Check for audit logging
        audit_check = ComplianceCheck(
            check_id="EA-003", 
            standard=ComplianceStandard.ENTERPRISE_AUTOMATION,
            description="Comprehensive audit logging",
            requirement="All automation activities must be logged",
            status=await self._verify_audit_logging(),
            risk_level="high"
        )
        if audit_check.status != ComplianceLevel.COMPLIANT:
            audit_check.gaps.append("Incomplete audit logging coverage")
            audit_check.remediation_steps.append("Implement comprehensive audit logging")
        
        checks.append(audit_check)
        
        # Check for access controls
        access_check = ComplianceCheck(
            check_id="EA-004",
            standard=ComplianceStandard.ENTERPRISE_AUTOMATION,
            description="Access control implementation",
            requirement="Proper access controls for automation systems",
            status=await self._verify_access_controls(),
            risk_level="critical"
        )
        checks.append(access_check)
        
        return checks
    
    async def _check_information_security_compliance(self) -> List[ComplianceCheck]:
        """Check information security compliance (ISO 27001, NIST)"""
        checks = []
        
        # Security configuration check
        security_config_check = ComplianceCheck(
            check_id="IS-001",
            standard=ComplianceStandard.ISO_27001,
            description="Security configuration management",
            requirement="Security configurations must be managed and documented",
            status=await self._verify_security_configuration(),
            risk_level="high"
        )
        checks.append(security_config_check)
        
        # Vulnerability management check
        vuln_check = ComplianceCheck(
            check_id="IS-002", 
            standard=ComplianceStandard.NIST,
            description="Vulnerability management",
            requirement="Regular vulnerability scanning and remediation",
            status=await self._verify_vulnerability_management(),
            risk_level="high"
        )
        checks.append(vuln_check)
        
        # Encryption check
        encryption_check = ComplianceCheck(
            check_id="IS-003",
            standard=ComplianceStandard.ISO_27001,
            description="Data encryption requirements",
            requirement="Sensitive data must be encrypted at rest and in transit",
            status=await self._verify_encryption_implementation(),
            risk_level="critical"
        )
        checks.append(encryption_check)
        
        return checks
    
    async def _check_data_protection_compliance(self) -> List[ComplianceCheck]:
        """Check data protection compliance (GDPR)"""
        checks = []
        
        # Data inventory check
        data_inventory_check = ComplianceCheck(
            check_id="DP-001",
            standard=ComplianceStandard.GDPR,
            description="Data inventory and classification",
            requirement="All personal data must be inventoried and classified",
            status=await self._verify_data_inventory(),
            risk_level="medium"
        )
        checks.append(data_inventory_check)
        
        # Data retention check
        retention_check = ComplianceCheck(
            check_id="DP-002",
            standard=ComplianceStandard.GDPR,
            description="Data retention policies",
            requirement="Data retention policies must be implemented and enforced",
            status=await self._verify_data_retention_policies(),
            risk_level="medium"
        )
        checks.append(retention_check)
        
        return checks
    
    async def _check_quality_assurance_compliance(self) -> List[ComplianceCheck]:
        """Check quality assurance compliance"""
        checks = []
        
        # Code quality check
        code_quality_check = ComplianceCheck(
            check_id="QA-001",
            standard=ComplianceStandard.ENTERPRISE_AUTOMATION,
            description="Code quality standards",
            requirement="Code must meet quality standards and be regularly reviewed",
            status=await self._verify_code_quality_standards(),
            risk_level="medium"
        )
        checks.append(code_quality_check)
        
        # Testing coverage check
        testing_check = ComplianceCheck(
            check_id="QA-002",
            standard=ComplianceStandard.ENTERPRISE_AUTOMATION,
            description="Testing coverage requirements",
            requirement="Minimum 95% test coverage for all critical components",
            status=await self._verify_testing_coverage(),
            risk_level="high"
        )
        checks.append(testing_check)
        
        return checks
    
    async def _check_operational_compliance(self) -> List[ComplianceCheck]:
        """Check operational compliance requirements"""
        checks = []
        
        # Monitoring check
        monitoring_check = ComplianceCheck(
            check_id="OP-001",
            standard=ComplianceStandard.ENTERPRISE_AUTOMATION,
            description="Operational monitoring",
            requirement="Comprehensive monitoring of all automated systems",
            status=await self._verify_operational_monitoring(),
            risk_level="high"
        )
        checks.append(monitoring_check)
        
        # Backup and recovery check
        backup_check = ComplianceCheck(
            check_id="OP-002",
            standard=ComplianceStandard.ENTERPRISE_AUTOMATION,
            description="Backup and recovery procedures",
            requirement="Regular backups and tested recovery procedures",
            status=await self._verify_backup_recovery(),
            risk_level="critical"
        )
        checks.append(backup_check)
        
        return checks
    
    # Verification methods
    
    async def _verify_error_recovery_mechanisms(self) -> ComplianceLevel:
        """Verify error recovery mechanisms are in place"""
        try:
            # Check for error recovery code
            recovery_files = list(self.base_dir.rglob("*recovery*.py"))
            error_handling_files = list(self.base_dir.rglob("*error*.py"))
            
            if len(recovery_files) > 0 or len(error_handling_files) > 0:
                return ComplianceLevel.COMPLIANT
            else:
                return ComplianceLevel.PARTIALLY_COMPLIANT
        except Exception:
            return ComplianceLevel.NON_COMPLIANT
    
    async def _verify_audit_logging(self) -> ComplianceLevel:
        """Verify comprehensive audit logging"""
        try:
            # Check for logging configuration
            logging_files = list(self.base_dir.rglob("*log*.py"))
            config_files = list(self.base_dir.rglob("*config*.py"))
            
            if len(logging_files) > 0:
                return ComplianceLevel.COMPLIANT
            else:
                return ComplianceLevel.PARTIALLY_COMPLIANT
        except Exception:
            return ComplianceLevel.NON_COMPLIANT
    
    async def _verify_access_controls(self) -> ComplianceLevel:
        """Verify access control implementation"""
        try:
            # Check for authentication/authorization code
            auth_files = list(self.base_dir.rglob("*auth*.py"))
            security_files = list(self.base_dir.rglob("*security*.py"))
            
            if len(auth_files) > 0 or len(security_files) > 0:
                return ComplianceLevel.PARTIALLY_COMPLIANT
            else:
                return ComplianceLevel.NON_COMPLIANT
        except Exception:
            return ComplianceLevel.NON_COMPLIANT
    
    async def _verify_security_configuration(self) -> ComplianceLevel:
        """Verify security configuration management"""
        try:
            # Check for security configuration files
            config_files = ["pyproject.toml", ".github", "requirements.txt"]
            existing_configs = sum(1 for config in config_files if (self.base_dir / config).exists())
            
            if existing_configs >= len(config_files):
                return ComplianceLevel.COMPLIANT
            elif existing_configs > 0:
                return ComplianceLevel.PARTIALLY_COMPLIANT
            else:
                return ComplianceLevel.NON_COMPLIANT
        except Exception:
            return ComplianceLevel.NON_COMPLIANT
    
    async def _verify_vulnerability_management(self) -> ComplianceLevel:
        """Verify vulnerability management processes"""
        try:
            # Check for security scanning tools/configurations
            security_tools = ["bandit", "safety", "pip-audit"]
            requirements_file = self.base_dir / "requirements-dev.txt"
            
            if requirements_file.exists():
                content = requirements_file.read_text()
                has_security_tools = any(tool in content.lower() for tool in security_tools)
                if has_security_tools:
                    return ComplianceLevel.COMPLIANT
            
            return ComplianceLevel.PARTIALLY_COMPLIANT
        except Exception:
            return ComplianceLevel.NON_COMPLIANT
    
    async def _verify_encryption_implementation(self) -> ComplianceLevel:
        """Verify encryption implementation"""
        try:
            # Check for encryption-related code
            crypto_patterns = ["crypt", "encrypt", "decrypt", "hash", "secure"]
            
            python_files = list(self.base_dir.rglob("*.py"))
            has_encryption = False
            
            for py_file in python_files[:10]:  # Check first 10 files for performance
                try:
                    content = py_file.read_text().lower()
                    if any(pattern in content for pattern in crypto_patterns):
                        has_encryption = True
                        break
                except Exception:
                    continue
            
            if has_encryption:
                return ComplianceLevel.PARTIALLY_COMPLIANT
            else:
                return ComplianceLevel.NON_COMPLIANT
        except Exception:
            return ComplianceLevel.NON_COMPLIANT
    
    async def _verify_data_inventory(self) -> ComplianceLevel:
        """Verify data inventory and classification"""
        try:
            # Check for data models and schemas
            data_files = list(self.base_dir.rglob("*data*.py"))
            model_files = list(self.base_dir.rglob("*model*.py"))
            schema_files = list(self.base_dir.rglob("*schema*.py"))
            
            total_data_files = len(data_files) + len(model_files) + len(schema_files)
            
            if total_data_files > 0:
                return ComplianceLevel.PARTIALLY_COMPLIANT
            else:
                return ComplianceLevel.NOT_APPLICABLE  # No personal data processing identified
        except Exception:
            return ComplianceLevel.NON_COMPLIANT
    
    async def _verify_data_retention_policies(self) -> ComplianceLevel:
        """Verify data retention policies"""
        try:
            # Check for retention policy documentation
            policy_files = list(self.base_dir.rglob("*policy*.md"))
            retention_files = list(self.base_dir.rglob("*retention*.py"))
            
            if len(policy_files) > 0 or len(retention_files) > 0:
                return ComplianceLevel.PARTIALLY_COMPLIANT
            else:
                return ComplianceLevel.NOT_APPLICABLE
        except Exception:
            return ComplianceLevel.NON_COMPLIANT
    
    async def _verify_code_quality_standards(self) -> ComplianceLevel:
        """Verify code quality standards"""
        try:
            # Check for code quality tools
            quality_tools = ["flake8", "black", "isort", "mypy", "pylint"]
            dev_requirements = self.base_dir / "requirements-dev.txt"
            pyproject_file = self.base_dir / "pyproject.toml"
            
            has_quality_tools = False
            
            if dev_requirements.exists():
                content = dev_requirements.read_text().lower()
                has_quality_tools = any(tool in content for tool in quality_tools)
            
            if pyproject_file.exists() and not has_quality_tools:
                content = pyproject_file.read_text().lower()
                has_quality_tools = any(tool in content for tool in quality_tools)
            
            if has_quality_tools:
                return ComplianceLevel.COMPLIANT
            else:
                return ComplianceLevel.PARTIALLY_COMPLIANT
        except Exception:
            return ComplianceLevel.NON_COMPLIANT
    
    async def _verify_testing_coverage(self) -> ComplianceLevel:
        """Verify testing coverage requirements"""
        try:
            # Check for test files and coverage configuration
            test_files = list(self.base_dir.rglob("test_*.py"))
            coverage_file = self.base_dir / "coverage.xml"
            pytest_config = self.base_dir / "pytest.ini"
            pyproject_file = self.base_dir / "pyproject.toml"
            
            has_tests = len(test_files) > 0
            has_coverage_config = coverage_file.exists() or pytest_config.exists()
            
            # Check pyproject.toml for pytest configuration
            if pyproject_file.exists() and not has_coverage_config:
                content = pyproject_file.read_text().lower()
                has_coverage_config = "pytest" in content or "coverage" in content
            
            if has_tests and has_coverage_config:
                return ComplianceLevel.COMPLIANT
            elif has_tests:
                return ComplianceLevel.PARTIALLY_COMPLIANT
            else:
                return ComplianceLevel.NON_COMPLIANT
        except Exception:
            return ComplianceLevel.NON_COMPLIANT
    
    async def _verify_operational_monitoring(self) -> ComplianceLevel:
        """Verify operational monitoring"""
        try:
            # Check for monitoring-related code
            monitoring_files = list(self.base_dir.rglob("*monitor*.py"))
            health_files = list(self.base_dir.rglob("*health*.py"))
            metrics_files = list(self.base_dir.rglob("*metrics*.py"))
            
            total_monitoring_files = len(monitoring_files) + len(health_files) + len(metrics_files)
            
            if total_monitoring_files > 0:
                return ComplianceLevel.COMPLIANT
            else:
                return ComplianceLevel.PARTIALLY_COMPLIANT
        except Exception:
            return ComplianceLevel.NON_COMPLIANT
    
    async def _verify_backup_recovery(self) -> ComplianceLevel:
        """Verify backup and recovery procedures"""
        try:
            # Check for backup-related code and documentation
            backup_files = list(self.base_dir.rglob("*backup*.py"))
            recovery_files = list(self.base_dir.rglob("*recovery*.py"))
            disaster_docs = list(self.base_dir.rglob("*disaster*.md"))
            
            has_backup_code = len(backup_files) > 0 or len(recovery_files) > 0
            has_documentation = len(disaster_docs) > 0
            
            if has_backup_code and has_documentation:
                return ComplianceLevel.COMPLIANT
            elif has_backup_code or has_documentation:
                return ComplianceLevel.PARTIALLY_COMPLIANT
            else:
                return ComplianceLevel.NON_COMPLIANT
        except Exception:
            return ComplianceLevel.NON_COMPLIANT


class CertificationValidator:
    """Validate automation system for certification"""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory_manager = memory_manager or MemoryManager()
        self.logger = logging.getLogger(__name__)
    
    async def validate_certification_eligibility(
        self, 
        compliance_validation: ComplianceValidation,
        target_level: CertificationLevel = CertificationLevel.ENTERPRISE
    ) -> bool:
        """Check if system is eligible for certification"""
        
        # Enterprise certification criteria
        if target_level == CertificationLevel.ENTERPRISE:
            criteria = {
                "minimum_compliance_percentage": 95.0,
                "maximum_critical_gaps": 0,
                "maximum_high_priority_gaps": 1,
                "required_compliant_checks": 15
            }
        elif target_level == CertificationLevel.PREMIUM:
            criteria = {
                "minimum_compliance_percentage": 90.0,
                "maximum_critical_gaps": 1,
                "maximum_high_priority_gaps": 3,
                "required_compliant_checks": 10
            }
        else:  # STANDARD or BASIC
            criteria = {
                "minimum_compliance_percentage": 80.0,
                "maximum_critical_gaps": 2,
                "maximum_high_priority_gaps": 5,
                "required_compliant_checks": 8
            }
        
        # Check eligibility criteria
        meets_compliance_percentage = (
            compliance_validation.compliance_percentage >= criteria["minimum_compliance_percentage"]
        )
        
        meets_critical_gaps = (
            len(compliance_validation.critical_gaps) <= criteria["maximum_critical_gaps"]
        )
        
        meets_high_priority_gaps = (
            len(compliance_validation.high_priority_gaps) <= criteria["maximum_high_priority_gaps"]
        )
        
        meets_compliant_checks = (
            compliance_validation.passed_checks >= criteria["required_compliant_checks"]
        )
        
        return all([
            meets_compliance_percentage,
            meets_critical_gaps,
            meets_high_priority_gaps,
            meets_compliant_checks
        ])
    
    async def generate_certification(
        self, 
        compliance_validation: ComplianceValidation,
        target_level: CertificationLevel = CertificationLevel.ENTERPRISE
    ) -> CertificationResult:
        """Generate automation certification"""
        
        is_eligible = await self.validate_certification_eligibility(
            compliance_validation, target_level
        )
        
        cert_id = f"MCP_AUTO_CERT_{int(datetime.utcnow().timestamp())}"
        
        # Calculate certification score
        cert_score = self._calculate_certification_score(compliance_validation)
        
        # Determine if certification should be issued
        should_issue = is_eligible and cert_score >= 85.0
        
        result = CertificationResult(
            certification_id=cert_id,
            certification_level=target_level,
            issued=should_issue,
            certification_score=cert_score
        )
        
        if should_issue:
            result.issue_date = datetime.utcnow()
            result.expiry_date = datetime.utcnow().replace(year=datetime.utcnow().year + 1)
            result.certification_evidence = {
                "compliance_percentage": compliance_validation.compliance_percentage,
                "passed_checks": compliance_validation.passed_checks,
                "total_checks": compliance_validation.total_checks,
                "validation_timestamp": compliance_validation.validation_timestamp.isoformat()
            }
        else:
            # Provide gaps and requirements for certification
            result.certification_gaps = self._identify_certification_gaps(
                compliance_validation, target_level
            )
            result.renewal_requirements = [
                "Address all critical compliance gaps",
                "Achieve minimum compliance percentage",
                "Implement required security controls",
                "Complete documentation requirements"
            ]
        
        # Set certification criteria met
        result.certification_criteria_met = {
            "compliance_percentage": compliance_validation.compliance_percentage >= 95.0,
            "critical_gaps": len(compliance_validation.critical_gaps) == 0,
            "high_priority_gaps": len(compliance_validation.high_priority_gaps) <= 1,
            "passed_checks": compliance_validation.passed_checks >= 15
        }
        
        return result
    
    def _calculate_certification_score(self, compliance_validation: ComplianceValidation) -> float:
        """Calculate overall certification score"""
        base_score = compliance_validation.compliance_percentage
        
        # Apply penalties for gaps
        critical_gap_penalty = len(compliance_validation.critical_gaps) * 10
        high_priority_gap_penalty = len(compliance_validation.high_priority_gaps) * 5
        
        total_penalty = critical_gap_penalty + high_priority_gap_penalty
        
        final_score = max(0.0, base_score - total_penalty)
        
        return min(100.0, final_score)
    
    def _identify_certification_gaps(
        self, 
        compliance_validation: ComplianceValidation,
        target_level: CertificationLevel
    ) -> List[str]:
        """Identify gaps preventing certification"""
        gaps = []
        
        if compliance_validation.compliance_percentage < 95.0:
            gap_percent = 95.0 - compliance_validation.compliance_percentage
            gaps.append(f"Increase compliance percentage by {gap_percent:.1f}%")
        
        if len(compliance_validation.critical_gaps) > 0:
            gaps.append(f"Resolve {len(compliance_validation.critical_gaps)} critical compliance gaps")
        
        if len(compliance_validation.high_priority_gaps) > 1:
            excess_gaps = len(compliance_validation.high_priority_gaps) - 1
            gaps.append(f"Resolve {excess_gaps} additional high-priority gaps")
        
        if compliance_validation.passed_checks < 15:
            missing_checks = 15 - compliance_validation.passed_checks
            gaps.append(f"Pass {missing_checks} additional compliance checks")
        
        return gaps


class ComplianceValidator:
    """Main compliance validator class"""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory_manager = memory_manager or MemoryManager()
        self.compliance_checker = ComplianceChecker(memory_manager)
        self.certification_validator = CertificationValidator(memory_manager)
        self.logger = logging.getLogger(__name__)
    
    async def validate_compliance(self) -> ComplianceValidation:
        """Validate all compliance requirements"""
        self.logger.info("Starting compliance validation")
        
        try:
            # Run all compliance checks
            all_checks = await self.compliance_checker.check_all_compliance_standards()
            
            # Analyze results
            total_checks = len(all_checks)
            passed_checks = sum(1 for check in all_checks if check.status == ComplianceLevel.COMPLIANT)
            failed_checks = sum(1 for check in all_checks if check.status == ComplianceLevel.NON_COMPLIANT)
            partial_checks = sum(1 for check in all_checks if check.status == ComplianceLevel.PARTIALLY_COMPLIANT)
            not_applicable = sum(1 for check in all_checks if check.status == ComplianceLevel.NOT_APPLICABLE)
            
            # Calculate compliance percentage (excluding not applicable)
            applicable_checks = total_checks - not_applicable
            if applicable_checks > 0:
                compliance_percentage = (passed_checks / applicable_checks) * 100
            else:
                compliance_percentage = 100.0
            
            # Determine overall compliance status
            if compliance_percentage >= 95.0 and failed_checks == 0:
                compliance_status = "COMPLIANT"
            elif compliance_percentage >= 80.0 and len([c for c in all_checks if c.risk_level == "critical"]) == 0:
                compliance_status = "PARTIALLY_COMPLIANT"
            else:
                compliance_status = "NON_COMPLIANT"
            
            # Identify critical and high-priority gaps
            critical_gaps = [
                f"{check.check_id}: {check.description}"
                for check in all_checks
                if check.status != ComplianceLevel.COMPLIANT and check.risk_level == "critical"
            ]
            
            high_priority_gaps = [
                f"{check.check_id}: {check.description}"
                for check in all_checks
                if check.status != ComplianceLevel.COMPLIANT and check.risk_level == "high"
            ]
            
            # Generate remediation plan
            remediation_plan = self._generate_remediation_plan(all_checks)
            
            validation_result = ComplianceValidation(
                compliance_status=compliance_status,
                total_checks=total_checks,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                not_applicable_checks=not_applicable,
                compliance_percentage=compliance_percentage,
                critical_gaps=critical_gaps,
                high_priority_gaps=high_priority_gaps,
                remediation_plan=remediation_plan,
                compliance_checks=all_checks
            )
            
            self.logger.info(
                "Compliance validation completed: %s (%.1f%%)",
                compliance_status, compliance_percentage
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error("Compliance validation failed: %s", str(e))
            return ComplianceValidation(
                compliance_status="NON_COMPLIANT",
                total_checks=0,
                passed_checks=0,
                failed_checks=1,
                not_applicable_checks=0,
                compliance_percentage=0.0,
                critical_gaps=[f"Compliance validation system failure: {str(e)}"],
                remediation_plan=["Fix compliance validation system"]
            )
    
    async def generate_certification(
        self, 
        compliance_validation: Optional[ComplianceValidation] = None
    ) -> CertificationResult:
        """Generate automation certification"""
        self.logger.info("Generating automation certification")
        
        try:
            if compliance_validation is None:
                compliance_validation = await self.validate_compliance()
            
            certification_result = await self.certification_validator.generate_certification(
                compliance_validation, CertificationLevel.ENTERPRISE
            )
            
            if certification_result.issued:
                self.logger.info(
                    "Certification issued: %s (score: %.1f)",
                    certification_result.certification_id,
                    certification_result.certification_score
                )
            else:
                self.logger.warning(
                    "Certification not issued. Gaps: %s",
                    ", ".join(certification_result.certification_gaps[:3])
                )
            
            return certification_result
            
        except Exception as e:
            self.logger.error("Certification generation failed: %s", str(e))
            return CertificationResult(
                certification_id="error",
                certification_level=CertificationLevel.BASIC,
                issued=False,
                certification_score=0.0,
                certification_gaps=[f"Certification system failure: {str(e)}"]
            )
    
    def _generate_remediation_plan(self, checks: List[ComplianceCheck]) -> List[str]:
        """Generate remediation plan from compliance checks"""
        remediation_steps = []
        
        # Critical issues first
        critical_checks = [check for check in checks if check.risk_level == "critical" and check.status != ComplianceLevel.COMPLIANT]
        for check in critical_checks:
            remediation_steps.extend(check.remediation_steps)
        
        # High-priority issues next
        high_checks = [check for check in checks if check.risk_level == "high" and check.status != ComplianceLevel.COMPLIANT]
        for check in high_checks:
            remediation_steps.extend(check.remediation_steps)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_steps = []
        for step in remediation_steps:
            if step not in seen:
                seen.add(step)
                unique_steps.append(step)
        
        return unique_steps[:10]  # Limit to top 10 recommendations