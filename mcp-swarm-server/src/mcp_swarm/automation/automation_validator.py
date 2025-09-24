"""
MCP Swarm Intelligence Server - Simplified Automation Validator

Validates complete automation across all development workflows with practical checks
that work with the existing codebase structure.
"""

import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from pathlib import Path
import subprocess
import json

# MCP Swarm imports
from ..memory.manager import MemoryManager
from ..agents.manager import AgentManager
from ..swarm.coordinator import SwarmCoordinator


class WorkflowType(Enum):
    """Types of workflows to validate for automation"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    COORDINATION = "coordination"
    KNOWLEDGE_MANAGEMENT = "knowledge_management"
    ERROR_RECOVERY = "error_recovery"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    QUALITY_ASSURANCE = "quality_assurance"


class AutomationLevel(Enum):
    """Automation levels for validation"""
    MANUAL = "manual"
    SEMI_AUTOMATED = "semi_automated"
    AUTOMATED = "automated"
    FULLY_AUTOMATED = "fully_automated"


@dataclass
class WorkflowValidation:
    """Validation result for a single workflow"""
    workflow_type: WorkflowType
    automation_level: AutomationLevel
    manual_steps_count: int
    automated_steps_count: int
    error_recovery_available: bool
    performance_meets_standards: bool
    validation_timestamp: datetime
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AutomationValidationResult:
    """Complete automation validation result"""
    overall_automation_percentage: float
    zero_manual_intervention: bool
    workflow_validations: Dict[WorkflowType, WorkflowValidation]
    critical_issues: List[str]
    improvement_recommendations: List[str]
    compliance_status: str
    validation_timestamp: datetime
    certification_eligible: bool


class AutomationValidator:
    """Validate complete automation across all development workflows"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory_manager = MemoryManager()
        self.agent_manager = AgentManager()
        self.swarm_coordinator = SwarmCoordinator()
        
        # Base directory for the project
        self.base_dir = Path(__file__).parent.parent.parent.parent.parent
        
    async def validate_complete_automation(
        self,
        validation_scope: str = "comprehensive"
    ) -> AutomationValidationResult:
        """
        Validate complete automation across all development workflows.
        
        Args:
            validation_scope: Scope of validation (comprehensive/core/minimal)
            
        Returns:
            Complete automation validation result
        """
        self.logger.info("Starting complete automation validation")
        
        try:
            # Determine workflows to validate based on scope
            workflows_to_validate = self._get_workflows_for_scope(validation_scope)
            
            # Validate each workflow
            workflow_validations = {}
            for workflow_type in workflows_to_validate:
                validation = await self._validate_workflow(workflow_type)
                workflow_validations[workflow_type] = validation
                
            # Calculate overall automation metrics
            automation_metrics = self._calculate_automation_metrics(workflow_validations)
            
            # Check for zero manual intervention
            zero_manual_intervention = self._check_zero_manual_intervention(workflow_validations)
            
            # Identify critical issues and recommendations
            critical_issues = self._identify_critical_issues(workflow_validations)
            improvement_recommendations = self._generate_improvement_recommendations(
                workflow_validations, automation_metrics
            )
            
            # Determine compliance status
            compliance_status = self._determine_compliance_status(
                automation_metrics, zero_manual_intervention, critical_issues
            )
            
            # Check certification eligibility
            certification_eligible = self._check_certification_eligibility(
                automation_metrics, zero_manual_intervention, critical_issues
            )
            
            result = AutomationValidationResult(
                overall_automation_percentage=automation_metrics["overall_percentage"],
                zero_manual_intervention=zero_manual_intervention,
                workflow_validations=workflow_validations,
                critical_issues=critical_issues,
                improvement_recommendations=improvement_recommendations,
                compliance_status=compliance_status,
                validation_timestamp=datetime.utcnow(),
                certification_eligible=certification_eligible
            )
            
            # Store validation results in memory
            await self._store_validation_results(result)
            
            self.logger.info(
                "Automation validation completed: %.1f%% automated",
                automation_metrics['overall_percentage']
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Automation validation failed: %s", str(e))
            raise
    
    async def _validate_workflow(self, workflow_type: WorkflowType) -> WorkflowValidation:
        """Validate a specific workflow"""
        manual_steps = 0
        automated_steps = 0
        issues = []
        recommendations = []
        
        # Development workflow checks
        if workflow_type == WorkflowType.DEVELOPMENT:
            # Check project structure automation
            if await self._check_project_structure():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Project structure setup not fully automated")
                recommendations.append("Implement automated project scaffolding")
            
            # Check dependency management
            if await self._check_dependency_management():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Dependency management requires manual intervention")
                recommendations.append("Automate dependency resolution")
                
            # Check build automation
            if await self._check_build_automation():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Build process not automated")
                recommendations.append("Implement automated builds")
                
            # Check code generation
            if await self._check_code_generation():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Code generation not automated")
                recommendations.append("Implement template-based code generation")
        
        # Testing workflow checks
        elif workflow_type == WorkflowType.TESTING:
            # Check test automation
            if await self._check_test_automation():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Tests not automated")
                recommendations.append("Implement continuous testing")
                
            # Check coverage automation
            if await self._check_coverage_automation():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Coverage reporting not automated")
                recommendations.append("Automate coverage collection")
                
            # Check quality gates
            if await self._check_quality_gates():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Quality gates not automated")
                recommendations.append("Implement automated quality gates")
        
        # Deployment workflow checks
        elif workflow_type == WorkflowType.DEPLOYMENT:
            # Check deployment automation
            if await self._check_deployment_automation():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Deployment not automated")
                recommendations.append("Implement CI/CD pipeline")
                
            # Check environment management
            if await self._check_environment_management():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Environment management not automated")
                recommendations.append("Automate environment provisioning")
        
        # Coordination workflow checks
        elif workflow_type == WorkflowType.COORDINATION:
            # Check agent coordination
            if await self._check_agent_coordination():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Agent coordination not automated")
                recommendations.append("Implement automated agent assignment")
                
            # Check task assignment
            if await self._check_task_assignment():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Task assignment requires manual intervention")
                recommendations.append("Automate task distribution")
        
        # Knowledge management workflow checks
        elif workflow_type == WorkflowType.KNOWLEDGE_MANAGEMENT:
            # Check knowledge extraction
            if await self._check_knowledge_extraction():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Knowledge extraction not automated")
                recommendations.append("Implement automated knowledge capture")
                
            # Check knowledge synthesis
            if await self._check_knowledge_synthesis():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Knowledge synthesis requires manual curation")
                recommendations.append("Automate knowledge organization")
        
        # Error recovery workflow checks
        elif workflow_type == WorkflowType.ERROR_RECOVERY:
            # Check error detection
            if await self._check_error_detection():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Error detection not automated")
                recommendations.append("Implement automated error monitoring")
                
            # Check recovery automation
            if await self._check_recovery_automation():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Recovery not automated")
                recommendations.append("Implement self-healing mechanisms")
        
        # Performance optimization workflow checks
        elif workflow_type == WorkflowType.PERFORMANCE_OPTIMIZATION:
            # Check performance monitoring
            if await self._check_performance_monitoring():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Performance monitoring not automated")
                recommendations.append("Implement automated performance tracking")
                
            # Check optimization automation
            if await self._check_optimization_automation():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Performance optimization not automated")
                recommendations.append("Automate performance tuning")
        
        # Quality assurance workflow checks
        elif workflow_type == WorkflowType.QUALITY_ASSURANCE:
            # Check code review automation
            if await self._check_code_review_automation():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Code review not automated")
                recommendations.append("Implement automated code analysis")
                
            # Check documentation automation
            if await self._check_documentation_automation():
                automated_steps += 1
            else:
                manual_steps += 1
                issues.append("Documentation generation not automated")
                recommendations.append("Automate documentation generation")
        
        # Determine automation level
        automation_level = self._calculate_workflow_automation_level(
            automated_steps, manual_steps
        )
        
        return WorkflowValidation(
            workflow_type=workflow_type,
            automation_level=automation_level,
            manual_steps_count=manual_steps,
            automated_steps_count=automated_steps,
            error_recovery_available=await self._check_workflow_error_recovery(workflow_type),
            performance_meets_standards=await self._check_workflow_performance(workflow_type),
            validation_timestamp=datetime.utcnow(),
            issues=issues,
            recommendations=recommendations
        )
    
    # Specific automation check methods
    
    async def _check_project_structure(self) -> bool:
        """Check if project structure is automated"""
        required_files = ["pyproject.toml", "requirements.txt", "src"]
        try:
            return all(
                (self.base_dir / file).exists() for file in required_files
            )
        except Exception:
            return False
    
    async def _check_dependency_management(self) -> bool:
        """Check if dependency management is automated"""
        try:
            return (
                (self.base_dir / "requirements.txt").exists() and
                (self.base_dir / "pyproject.toml").exists()
            )
        except Exception:
            return False
    
    async def _check_build_automation(self) -> bool:
        """Check if build is automated"""
        try:
            return (self.base_dir / "pyproject.toml").exists()
        except Exception:
            return False
    
    async def _check_code_generation(self) -> bool:
        """Check if code generation is automated"""
        try:
            templates_dir = self.base_dir / "src" / "mcp_swarm" / "templates"
            return templates_dir.exists() and any(templates_dir.glob("*.py"))
        except Exception:
            return False
    
    async def _check_test_automation(self) -> bool:
        """Check if testing is automated"""
        try:
            # Check for pytest
            result = subprocess.run(
                ["python", "-m", "pytest", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def _check_coverage_automation(self) -> bool:
        """Check if coverage is automated"""
        try:
            # Check for coverage tools
            result = subprocess.run(
                ["python", "-m", "coverage", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def _check_quality_gates(self) -> bool:
        """Check if quality gates are automated"""
        try:
            quality_dir = self.base_dir / "src" / "mcp_swarm" / "quality"
            return quality_dir.exists() and any(quality_dir.glob("*gate*.py"))
        except Exception:
            return False
    
    async def _check_deployment_automation(self) -> bool:
        """Check if deployment is automated"""
        try:
            ci_cd_files = [".github/workflows", "Dockerfile", "docker-compose.yml"]
            return any((self.base_dir / file).exists() for file in ci_cd_files)
        except Exception:
            return False
    
    async def _check_environment_management(self) -> bool:
        """Check if environment management is automated"""
        try:
            env_files = [".env.example", "docker-compose.yml", "Dockerfile"]
            return any((self.base_dir / file).exists() for file in env_files)
        except Exception:
            return False
    
    async def _check_agent_coordination(self) -> bool:
        """Check if agent coordination is automated"""
        try:
            # Check for swarm coordinator functionality
            return hasattr(self.swarm_coordinator, 'coordinate_agents')
        except Exception:
            return False
    
    async def _check_task_assignment(self) -> bool:
        """Check if task assignment is automated"""
        try:
            # Check for agent manager functionality
            return hasattr(self.agent_manager, 'assign_task')
        except Exception:
            return False
    
    async def _check_knowledge_extraction(self) -> bool:
        """Check if knowledge extraction is automated"""
        try:
            # Check for memory manager functionality
            return hasattr(self.memory_manager, 'extract_knowledge')
        except Exception:
            return False
    
    async def _check_knowledge_synthesis(self) -> bool:
        """Check if knowledge synthesis is automated"""
        try:
            # Check for memory manager synthesis functionality
            return hasattr(self.memory_manager, 'synthesize_knowledge')
        except Exception:
            return False
    
    async def _check_error_detection(self) -> bool:
        """Check if error detection is automated"""
        try:
            automation_dir = self.base_dir / "src" / "mcp_swarm" / "automation"
            return (automation_dir / "error_recovery.py").exists()
        except Exception:
            return False
    
    async def _check_recovery_automation(self) -> bool:
        """Check if recovery is automated"""
        try:
            automation_dir = self.base_dir / "src" / "mcp_swarm" / "automation"
            return (automation_dir / "error_recovery.py").exists()
        except Exception:
            return False
    
    async def _check_performance_monitoring(self) -> bool:
        """Check if performance monitoring is automated"""
        try:
            automation_dir = self.base_dir / "src" / "mcp_swarm" / "automation"
            return (automation_dir / "performance_optimizer.py").exists()
        except Exception:
            return False
    
    async def _check_optimization_automation(self) -> bool:
        """Check if optimization is automated"""
        try:
            automation_dir = self.base_dir / "src" / "mcp_swarm" / "automation"
            return (automation_dir / "performance_optimizer.py").exists()
        except Exception:
            return False
    
    async def _check_code_review_automation(self) -> bool:
        """Check if code review is automated"""
        try:
            quality_dir = self.base_dir / "src" / "mcp_swarm" / "quality"
            return quality_dir.exists()
        except Exception:
            return False
    
    async def _check_documentation_automation(self) -> bool:
        """Check if documentation generation is automated"""
        try:
            docs_dir = self.base_dir / "docs"
            return docs_dir.exists()
        except Exception:
            return False
    
    async def _check_workflow_error_recovery(self, workflow_type: WorkflowType) -> bool:
        """Check if workflow has error recovery"""
        try:
            # Basic check - if error recovery system exists
            automation_dir = self.base_dir / "src" / "mcp_swarm" / "automation"
            return (automation_dir / "error_recovery.py").exists()
        except Exception:
            return False
    
    async def _check_workflow_performance(self, workflow_type: WorkflowType) -> bool:
        """Check if workflow meets performance standards"""
        try:
            # Basic check - assume performance is met if optimization exists
            automation_dir = self.base_dir / "src" / "mcp_swarm" / "automation"
            return (automation_dir / "performance_optimizer.py").exists()
        except Exception:
            return False
    
    # Utility methods
    
    def _get_workflows_for_scope(self, scope: str) -> List[WorkflowType]:
        """Get workflows to validate based on scope"""
        if scope == "comprehensive":
            return list(WorkflowType)
        elif scope == "core":
            return [
                WorkflowType.DEVELOPMENT,
                WorkflowType.TESTING,
                WorkflowType.DEPLOYMENT,
                WorkflowType.COORDINATION
            ]
        elif scope == "minimal":
            return [
                WorkflowType.DEVELOPMENT,
                WorkflowType.TESTING
            ]
        else:
            return list(WorkflowType)
    
    def _calculate_workflow_automation_level(
        self, 
        automated_steps: int, 
        manual_steps: int
    ) -> AutomationLevel:
        """Calculate automation level for a workflow"""
        total_steps = automated_steps + manual_steps
        if total_steps == 0:
            return AutomationLevel.MANUAL
        
        automation_percentage = automated_steps / total_steps
        
        if automation_percentage >= 1.0:
            return AutomationLevel.FULLY_AUTOMATED
        elif automation_percentage >= 0.8:
            return AutomationLevel.AUTOMATED
        elif automation_percentage >= 0.5:
            return AutomationLevel.SEMI_AUTOMATED
        else:
            return AutomationLevel.MANUAL
    
    def _calculate_automation_metrics(
        self, 
        workflow_validations: Dict[WorkflowType, WorkflowValidation]
    ) -> Dict[str, Any]:
        """Calculate overall automation metrics"""
        total_automated_steps = sum(
            validation.automated_steps_count 
            for validation in workflow_validations.values()
        )
        total_manual_steps = sum(
            validation.manual_steps_count 
            for validation in workflow_validations.values()
        )
        total_steps = total_automated_steps + total_manual_steps
        
        overall_percentage = (
            (total_automated_steps / total_steps * 100) if total_steps > 0 else 0
        )
        
        return {
            "overall_percentage": overall_percentage,
            "total_automated_steps": total_automated_steps,
            "total_manual_steps": total_manual_steps,
            "total_steps": total_steps,
            "workflow_count": len(workflow_validations)
        }
    
    def _check_zero_manual_intervention(
        self, 
        workflow_validations: Dict[WorkflowType, WorkflowValidation]
    ) -> bool:
        """Check if zero manual intervention requirement is met"""
        return all(
            validation.manual_steps_count == 0 
            for validation in workflow_validations.values()
        )
    
    def _identify_critical_issues(
        self, 
        workflow_validations: Dict[WorkflowType, WorkflowValidation]
    ) -> List[str]:
        """Identify critical issues across workflows"""
        critical_issues = []
        
        for workflow_type, validation in workflow_validations.items():
            # Critical issues are manual steps in critical workflows
            if workflow_type in [WorkflowType.DEVELOPMENT, WorkflowType.TESTING, WorkflowType.DEPLOYMENT]:
                if validation.manual_steps_count > 0:
                    critical_issues.extend([
                        f"{workflow_type.value}: {issue}" 
                        for issue in validation.issues
                    ])
            
            # Error recovery issues are always critical
            if not validation.error_recovery_available:
                critical_issues.append(
                    f"{workflow_type.value}: No automated error recovery available"
                )
            
            # Performance issues are critical
            if not validation.performance_meets_standards:
                critical_issues.append(
                    f"{workflow_type.value}: Performance standards not met"
                )
        
        return critical_issues
    
    def _generate_improvement_recommendations(
        self, 
        workflow_validations: Dict[WorkflowType, WorkflowValidation],
        automation_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Overall recommendations
        if automation_metrics["overall_percentage"] < 100:
            recommendations.append(
                f"Increase overall automation from {automation_metrics['overall_percentage']:.1f}% to 100%"
            )
        
        # Workflow-specific recommendations
        for validation in workflow_validations.values():
            recommendations.extend(validation.recommendations)
        
        # Prioritize recommendations
        priority_keywords = ["critical", "security", "error", "performance"]
        recommendations.sort(
            key=lambda rec: any(keyword in rec.lower() for keyword in priority_keywords),
            reverse=True
        )
        
        return recommendations[:10]  # Return top 10 recommendations
    
    def _determine_compliance_status(
        self, 
        automation_metrics: Dict[str, Any],
        zero_manual_intervention: bool,
        critical_issues: List[str]
    ) -> str:
        """Determine compliance status"""
        if zero_manual_intervention and automation_metrics["overall_percentage"] >= 100:
            return "FULLY_COMPLIANT"
        elif automation_metrics["overall_percentage"] >= 95 and len(critical_issues) == 0:
            return "SUBSTANTIALLY_COMPLIANT"
        elif automation_metrics["overall_percentage"] >= 80:
            return "PARTIALLY_COMPLIANT"
        else:
            return "NON_COMPLIANT"
    
    def _check_certification_eligibility(
        self, 
        automation_metrics: Dict[str, Any],
        zero_manual_intervention: bool,
        critical_issues: List[str]
    ) -> bool:
        """Check if system is eligible for automation certification"""
        return (
            zero_manual_intervention and
            automation_metrics["overall_percentage"] >= 100 and
            len(critical_issues) == 0
        )
    
    async def _store_validation_results(
        self, 
        result: AutomationValidationResult
    ) -> None:
        """Store validation results in memory"""
        try:
            # Convert result to dictionary for storage
            result_dict = {
                "overall_automation_percentage": result.overall_automation_percentage,
                "zero_manual_intervention": result.zero_manual_intervention,
                "critical_issues": result.critical_issues,
                "improvement_recommendations": result.improvement_recommendations,
                "compliance_status": result.compliance_status,
                "validation_timestamp": result.validation_timestamp.isoformat(),
                "certification_eligible": result.certification_eligible
            }
            
            await self.memory_manager.store_memory(
                "automation_validation_results",
                result_dict
            )
        except Exception as e:
            self.logger.warning("Failed to store validation results: %s", str(e))