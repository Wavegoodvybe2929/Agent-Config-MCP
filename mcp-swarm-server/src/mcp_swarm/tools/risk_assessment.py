"""
Risk Assessment Engine for Dynamic Coordination

This module provides comprehensive risk analysis for coordination strategies,
identifying potential issues, timeline impacts, and mitigation recommendations
to ensure successful multi-agent task execution.
"""

from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np

from .coordination_strategies import Task, Agent, BaseCoordinationStrategy

# Set up logging
logger = logging.getLogger(__name__)


class RiskType(Enum):
    """Types of coordination risks."""
    TIMELINE = "timeline"
    RESOURCE = "resource"
    COORDINATION = "coordination"
    QUALITY = "quality"
    DEPENDENCY = "dependency"
    COMMUNICATION = "communication"
    PERFORMANCE = "performance"
    TECHNICAL = "technical"


class RiskSeverity(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskFactor:
    """Represents a specific risk factor with mitigation strategies."""
    id: str
    type: RiskType
    severity: RiskSeverity
    probability: float  # 0.0 to 1.0
    impact: float  # 0.0 to 1.0
    description: str
    mitigation_strategies: List[str]
    detection_indicators: List[str]
    contingency_plans: List[str]
    timeline_impact_hours: float
    affected_components: List[str]


@dataclass
class RiskAssessmentResult:
    """Complete risk assessment for a coordination strategy."""
    overall_risk_score: float
    risk_factors: List[RiskFactor]
    recommendations: List[str]
    contingency_strategies: List[str]
    monitoring_requirements: List[str]
    success_probability: float
    timeline_buffer_recommendation: float  # hours


class RiskAssessmentEngine:
    """Comprehensive risk assessment for coordination strategies."""
    
    def __init__(self):
        self.risk_patterns = self._load_historical_patterns()
        self.mitigation_library = self._initialize_mitigation_library()
        self.risk_weights = self._initialize_risk_weights()
        self.assessment_history: List[Dict[str, Any]] = []
    
    def _load_historical_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load historical risk patterns for pattern matching."""
        return {
            "high_complexity_patterns": [
                {
                    "context": {"task_complexity": "> 0.7", "team_size": "< 5"},
                    "common_risks": ["coordination_bottleneck", "expertise_shortage"],
                    "success_indicators": ["task_breakdown", "expert_allocation"]
                }
            ],
            "tight_deadline_patterns": [
                {
                    "context": {"time_pressure": "> 0.8", "dependency_density": "> 0.5"},
                    "common_risks": ["cascade_delays", "quality_compromise"],
                    "mitigation": ["parallel_execution", "quality_gates"]
                }
            ],
            "resource_constraint_patterns": [
                {
                    "context": {"resource_availability": "< 0.5", "task_count": "> 10"},
                    "common_risks": ["resource_contention", "agent_burnout"],
                    "solutions": ["load_balancing", "task_prioritization"]
                }
            ]
        }
    
    def _initialize_mitigation_library(self) -> Dict[str, List[str]]:
        """Initialize library of mitigation strategies."""
        return {
            "timeline": [
                "Add timeline buffer of 20-30%",
                "Implement parallel execution where possible",
                "Identify and optimize critical path",
                "Prepare fast-track alternatives",
                "Set up early warning systems for delays"
            ],
            "resource": [
                "Implement dynamic load balancing",
                "Prepare agent backup assignments",
                "Set up resource monitoring alerts",
                "Create resource sharing protocols",
                "Establish priority queuing system"
            ],
            "coordination": [
                "Implement regular sync checkpoints",
                "Use standardized communication protocols",
                "Create clear escalation procedures",
                "Set up automated status reporting",
                "Establish conflict resolution processes"
            ],
            "quality": [
                "Implement staged quality gates",
                "Set up automated testing protocols",
                "Create peer review processes",
                "Establish quality metrics tracking",
                "Prepare quality recovery procedures"
            ],
            "dependency": [
                "Map and validate all dependencies",
                "Create dependency monitoring system",
                "Prepare dependency bypass procedures",
                "Implement dependency breaking strategies",
                "Set up dependency change notifications"
            ],
            "communication": [
                "Establish clear communication channels",
                "Implement redundant communication paths",
                "Create communication escalation matrix",
                "Set up automated status broadcasts",
                "Prepare communication failure procedures"
            ]
        }
    
    def _initialize_risk_weights(self) -> Dict[str, float]:
        """Initialize weights for different risk factors."""
        return {
            "probability": 0.4,
            "impact": 0.4,
            "detection_difficulty": 0.1,
            "mitigation_complexity": 0.1
        }
    
    async def assess_coordination_risks(
        self, 
        strategy: BaseCoordinationStrategy,
        tasks: List[Task],
        agents: List[Agent],
        context: Optional[Dict[str, Any]] = None
    ) -> RiskAssessmentResult:
        """Perform comprehensive risk assessment for coordination strategy."""
        
        try:
            context = context or {}
            
            # Identify all risk factors
            risk_factors = []
            
            # Timeline risks
            timeline_risks = self._analyze_timeline_risks(strategy, tasks, agents, context)
            risk_factors.extend(timeline_risks)
            
            # Resource risks
            resource_risks = self._analyze_resource_risks(strategy, tasks, agents, context)
            risk_factors.extend(resource_risks)
            
            # Coordination risks
            coordination_risks = self._analyze_coordination_risks(strategy, tasks, agents, context)
            risk_factors.extend(coordination_risks)
            
            # Dependency risks
            dependency_risks = self._analyze_dependency_risks(tasks, context)
            risk_factors.extend(dependency_risks)
            
            # Quality risks
            quality_risks = self._analyze_quality_risks(strategy, tasks, agents, context)
            risk_factors.extend(quality_risks)
            
            # Performance risks
            performance_risks = self._analyze_performance_risks(strategy, tasks, agents, context)
            risk_factors.extend(performance_risks)
            
            # Calculate overall risk score
            overall_risk = self._calculate_overall_risk_score(risk_factors)
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(risk_factors, strategy)
            
            # Generate contingency strategies
            contingency_strategies = self._generate_contingency_strategies(risk_factors, strategy)
            
            # Generate monitoring requirements
            monitoring_requirements = self._generate_monitoring_requirements(risk_factors)
            
            # Calculate success probability
            success_probability = self._calculate_success_probability(overall_risk, risk_factors)
            
            # Calculate timeline buffer recommendation
            timeline_buffer = self._calculate_timeline_buffer(risk_factors)
            
            result = RiskAssessmentResult(
                overall_risk_score=overall_risk,
                risk_factors=risk_factors,
                recommendations=recommendations,
                contingency_strategies=contingency_strategies,
                monitoring_requirements=monitoring_requirements,
                success_probability=success_probability,
                timeline_buffer_recommendation=timeline_buffer
            )
            
            # Record assessment for learning
            self._record_assessment(result, strategy, tasks, agents, context)
            
            return result
            
        except RuntimeError as e:
            logger.error("Risk assessment failed: %s", str(e))
            return self._create_fallback_assessment()
    
    def _analyze_timeline_risks(
        self,
        strategy: BaseCoordinationStrategy,
        tasks: List[Task],
        agents: List[Agent],  # pylint: disable=unused-argument
        context: Dict[str, Any]
    ) -> List[RiskFactor]:
        """Analyze timeline-related risks."""
        
        risks = []
        
        # Deadline pressure analysis
        time_pressure = context.get('time_pressure', 0.5)
        if time_pressure > 0.7:
            severity = RiskSeverity.HIGH if time_pressure > 0.9 else RiskSeverity.MEDIUM
            
            risks.append(RiskFactor(
                id="tight_deadlines",
                type=RiskType.TIMELINE,
                severity=severity,
                probability=time_pressure,
                impact=0.8,
                description=f"High time pressure (score: {time_pressure:.2f}) may lead to rushed execution and quality issues",
                mitigation_strategies=[
                    "Add 20-30% timeline buffer",
                    "Implement parallel execution where possible",
                    "Prepare fast-track procedures",
                    "Set up early warning systems"
                ],
                detection_indicators=[
                    "Tasks consistently taking longer than estimated",
                    "Quality metrics declining",
                    "Agent stress levels increasing"
                ],
                contingency_plans=[
                    "Switch to more aggressive parallel strategy",
                    "Reduce scope if possible",
                    "Add additional resources"
                ],
                timeline_impact_hours=float(sum(task.estimated_duration for task in tasks) * 0.3 / 60),
                affected_components=["all_tasks"]
            ))
        
        # Strategy-specific timeline risks
        if strategy.name == "Sequential" and len(tasks) > 5:
            risks.append(RiskFactor(
                id="sequential_bottleneck",
                type=RiskType.TIMELINE,
                severity=RiskSeverity.MEDIUM,
                probability=0.6,
                impact=0.7,
                description="Sequential execution with many tasks creates timeline bottlenecks",
                mitigation_strategies=[
                    "Break tasks into smaller chunks",
                    "Identify parallelizable sub-tasks",
                    "Optimize task ordering"
                ],
                detection_indicators=[
                    "Critical path becoming longer",
                    "Idle agents waiting for dependencies"
                ],
                contingency_plans=[
                    "Switch to hybrid parallel-sequential approach",
                    "Implement task pipelining"
                ],
                timeline_impact_hours=float(len(tasks) * 0.5),
                affected_components=["task_execution"]
            ))
        
        # Dependency cascade risks
        dependency_count = sum(len(task.dependencies) for task in tasks)
        if dependency_count > len(tasks) * 0.5:  # High dependency density
            risks.append(RiskFactor(
                id="dependency_cascade_delays",
                type=RiskType.TIMELINE,
                severity=RiskSeverity.HIGH,
                probability=0.7,
                impact=0.9,
                description="High dependency density may cause cascade delays",
                mitigation_strategies=[
                    "Map critical path dependencies",
                    "Create dependency bypass procedures",
                    "Implement dependency monitoring"
                ],
                detection_indicators=[
                    "Multiple tasks blocked by single dependency",
                    "Dependency resolution taking longer than expected"
                ],
                contingency_plans=[
                    "Activate dependency bypass procedures",
                    "Re-prioritize to resolve blocking dependencies first"
                ],
                timeline_impact_hours=float(dependency_count * 0.3),
                affected_components=["dependent_tasks"]
            ))
        
        return risks
    
    def _analyze_resource_risks(
        self,
        strategy: BaseCoordinationStrategy,  # pylint: disable=unused-argument
        tasks: List[Task],
        agents: List[Agent],
        context: Dict[str, Any]  # pylint: disable=unused-argument
    ) -> List[RiskFactor]:
        """Analyze resource-related risks."""
        
        risks = []
        
        # Agent availability risks
        available_agents = [agent for agent in agents if agent.availability]
        availability_ratio = len(available_agents) / max(len(agents), 1)
        
        if availability_ratio < 0.7:
            risks.append(RiskFactor(
                id="agent_shortage",
                type=RiskType.RESOURCE,
                severity=RiskSeverity.HIGH if availability_ratio < 0.5 else RiskSeverity.MEDIUM,
                probability=0.8,
                impact=0.7,
                description=f"Low agent availability ({availability_ratio:.1%}) may cause resource bottlenecks",
                mitigation_strategies=[
                    "Implement dynamic load balancing",
                    "Prepare backup agent assignments",
                    "Create priority task queuing"
                ],
                detection_indicators=[
                    "Tasks waiting for available agents",
                    "Agent utilization exceeding 90%"
                ],
                contingency_plans=[
                    "Recruit additional agents",
                    "Reduce parallel task execution",
                    "Implement time-sharing protocols"
                ],
                timeline_impact_hours=float(len(tasks) * (1 - availability_ratio) * 2),
                affected_components=["agent_allocation"]
            ))
        
        # Agent load distribution risks
        if available_agents:
            loads = [agent.current_load for agent in available_agents]
            max_load = max(loads)
            load_variance = float(np.var(loads))
            
            if max_load > 0.8 or load_variance > 0.3:
                risks.append(RiskFactor(
                    id="uneven_load_distribution",
                    type=RiskType.RESOURCE,
                    severity=RiskSeverity.MEDIUM,
                    probability=0.6,
                    impact=0.5,
                    description="Uneven agent load distribution may cause performance issues",
                    mitigation_strategies=[
                        "Implement load balancing algorithms",
                        "Monitor agent performance metrics",
                        "Redistribute tasks dynamically"
                    ],
                    detection_indicators=[
                        "Some agents consistently overloaded",
                        "Performance degradation in high-load agents"
                    ],
                    contingency_plans=[
                        "Immediately redistribute high-priority tasks",
                        "Temporary agent reassignment"
                    ],
                    timeline_impact_hours=float(max_load * 5),
                    affected_components=["load_balancing"]
                ))
        
        # Capability coverage risks
        required_capabilities = set()
        for task in tasks:
            required_capabilities.update(task.required_capabilities)
        
        available_capabilities = set()
        for agent in available_agents:
            available_capabilities.update(agent.capabilities)
        
        missing_capabilities = required_capabilities - available_capabilities
        if missing_capabilities:
            risks.append(RiskFactor(
                id="capability_gaps",
                type=RiskType.RESOURCE,
                severity=RiskSeverity.CRITICAL,
                probability=1.0,
                impact=1.0,
                description=f"Missing required capabilities: {', '.join(missing_capabilities)}",
                mitigation_strategies=[
                    "Acquire agents with missing capabilities",
                    "Cross-train existing agents",
                    "Outsource tasks requiring missing capabilities"
                ],
                detection_indicators=[
                    "Tasks cannot be assigned to any agent",
                    "Capability matching failures"
                ],
                contingency_plans=[
                    "Emergency agent recruitment",
                    "Task scope reduction",
                    "External capability sourcing"
                ],
                timeline_impact_hours=float(len(missing_capabilities) * 8),
                affected_components=["capability_matching"]
            ))
        
        return risks
    
    def _analyze_coordination_risks(
        self,
        strategy: BaseCoordinationStrategy,
        tasks: List[Task],
        agents: List[Agent],
        context: Dict[str, Any]
    ) -> List[RiskFactor]:
        """Analyze coordination-related risks."""
        
        risks = []
        
        # Team size coordination risks
        team_size = len(agents)
        
        if strategy.name == "Parallel" and team_size > 10:
            risks.append(RiskFactor(
                id="coordination_complexity",
                type=RiskType.COORDINATION,
                severity=RiskSeverity.MEDIUM,
                probability=0.6,
                impact=0.6,
                description="Large team size with parallel strategy may create coordination overhead",
                mitigation_strategies=[
                    "Implement hierarchical coordination",
                    "Create sub-team structures",
                    "Use automated coordination tools"
                ],
                detection_indicators=[
                    "Increased communication overhead",
                    "Coordination meetings taking longer",
                    "Synchronization delays"
                ],
                contingency_plans=[
                    "Switch to hierarchical strategy",
                    "Break team into smaller sub-teams"
                ],
                timeline_impact_hours=float(team_size * 0.2),
                affected_components=["team_coordination"]
            ))
        
        # Communication risks
        collaboration_history = context.get('collaboration_history', 0.5)
        if collaboration_history < 0.4:
            risks.append(RiskFactor(
                id="poor_collaboration_history",
                type=RiskType.COORDINATION,
                severity=RiskSeverity.MEDIUM,
                probability=0.7,
                impact=0.5,
                description="Poor collaboration history may lead to coordination failures",
                mitigation_strategies=[
                    "Implement structured communication protocols",
                    "Provide team coordination training",
                    "Set up regular sync meetings"
                ],
                detection_indicators=[
                    "Miscommunication incidents",
                    "Duplicated work",
                    "Missed handoffs"
                ],
                contingency_plans=[
                    "Assign dedicated coordination facilitator",
                    "Implement stricter communication protocols"
                ],
                timeline_impact_hours=float(len(tasks) * 0.5),
                affected_components=["team_communication"]
            ))
        
        # Strategy-specific coordination risks
        if strategy.name == "Swarm-Based" and team_size < 3:
            risks.append(RiskFactor(
                id="insufficient_swarm_size",
                type=RiskType.COORDINATION,
                severity=RiskSeverity.HIGH,
                probability=0.8,
                impact=0.7,
                description="Swarm strategy requires minimum team size for effective coordination",
                mitigation_strategies=[
                    "Switch to sequential or parallel strategy",
                    "Add additional agents to team",
                    "Use hybrid coordination approach"
                ],
                detection_indicators=[
                    "Swarm algorithms not converging",
                    "Poor task assignment optimization"
                ],
                contingency_plans=[
                    "Fall back to sequential strategy",
                    "Emergency team expansion"
                ],
                timeline_impact_hours=float(len(tasks) * 1.5),
                affected_components=["swarm_coordination"]
            ))
        
        return risks
    
    def _analyze_dependency_risks(self, tasks: List[Task], context: Dict[str, Any]) -> List[RiskFactor]:  # pylint: disable=unused-argument
        """Analyze risks from task dependencies."""
        
        risks = []
        
        # Circular dependency detection
        circular_deps = self._detect_circular_dependencies(tasks)
        
        if circular_deps:
            risks.append(RiskFactor(
                id="circular_dependencies",
                type=RiskType.DEPENDENCY,
                severity=RiskSeverity.CRITICAL,
                probability=1.0,
                impact=1.0,
                description=f"Circular dependencies detected: {circular_deps}",
                mitigation_strategies=[
                    "Break circular dependencies",
                    "Restructure task relationships",
                    "Implement dependency versioning"
                ],
                detection_indicators=[
                    "Tasks waiting indefinitely for dependencies",
                    "Dependency resolution loops"
                ],
                contingency_plans=[
                    "Emergency dependency breaking",
                    "Task restructuring",
                    "Manual dependency override"
                ],
                timeline_impact_hours=float(len(circular_deps) * 4),
                affected_components=["dependency_resolution"]
            ))
        
        # Critical path risks
        critical_path_length = self._calculate_critical_path_length(tasks)
        total_work = sum(task.estimated_duration for task in tasks)
        
        if critical_path_length > total_work * 0.7:  # Long critical path
            risks.append(RiskFactor(
                id="long_critical_path",
                type=RiskType.DEPENDENCY,
                severity=RiskSeverity.MEDIUM,
                probability=0.5,
                impact=0.6,
                description="Long critical path limits parallelization opportunities",
                mitigation_strategies=[
                    "Optimize critical path task ordering",
                    "Break down critical path tasks",
                    "Identify parallel execution opportunities"
                ],
                detection_indicators=[
                    "Critical path delays affecting overall timeline",
                    "Limited parallelization effectiveness"
                ],
                contingency_plans=[
                    "Critical path task prioritization",
                    "Resource concentration on critical path"
                ],
                timeline_impact_hours=float(critical_path_length * 0.2 / 60),
                affected_components=["critical_path"]
            ))
        
        return risks
    
    def _analyze_quality_risks(
        self,
        strategy: BaseCoordinationStrategy,  # pylint: disable=unused-argument
        tasks: List[Task],
        agents: List[Agent],  # pylint: disable=unused-argument
        context: Dict[str, Any]
    ) -> List[RiskFactor]:
        """Analyze quality-related risks."""
        
        risks = []
        
        # Time pressure quality risks
        time_pressure = context.get('time_pressure', 0.5)
        if time_pressure > 0.8:
            risks.append(RiskFactor(
                id="quality_compromise_time_pressure",
                type=RiskType.QUALITY,
                severity=RiskSeverity.HIGH,
                probability=0.7,
                impact=0.8,
                description="High time pressure may compromise work quality",
                mitigation_strategies=[
                    "Implement mandatory quality gates",
                    "Set up automated quality checks",
                    "Prepare quality recovery procedures"
                ],
                detection_indicators=[
                    "Quality metrics declining",
                    "Increased error rates",
                    "Review failures increasing"
                ],
                contingency_plans=[
                    "Implement emergency quality review",
                    "Slow down execution to maintain quality",
                    "Add quality assurance resources"
                ],
                timeline_impact_hours=float(len(tasks) * 0.3),
                affected_components=["quality_assurance"]
            ))
        
        # Complexity quality risks
        high_complexity_tasks = [task for task in tasks if task.complexity > 0.7]
        if len(high_complexity_tasks) > len(tasks) * 0.3:
            risks.append(RiskFactor(
                id="complexity_quality_risk",
                type=RiskType.QUALITY,
                severity=RiskSeverity.MEDIUM,
                probability=0.6,
                impact=0.7,
                description="High task complexity may lead to quality issues",
                mitigation_strategies=[
                    "Assign most experienced agents to complex tasks",
                    "Implement peer review for complex tasks",
                    "Break down complex tasks into simpler components"
                ],
                detection_indicators=[
                    "Higher error rates in complex tasks",
                    "Longer than expected completion times"
                ],
                contingency_plans=[
                    "Emergency expert assignment",
                    "Task complexity reduction",
                    "Additional review cycles"
                ],
                timeline_impact_hours=float(len(high_complexity_tasks) * 0.5),
                affected_components=["complex_tasks"]
            ))
        
        return risks
    
    def _analyze_performance_risks(
        self,
        strategy: BaseCoordinationStrategy,
        tasks: List[Task],
        agents: List[Agent],
        context: Dict[str, Any]  # pylint: disable=unused-argument
    ) -> List[RiskFactor]:
        """Analyze performance-related risks."""
        
        risks = []
        
        # Agent performance risks
        low_performers = [
            agent for agent in agents 
            if agent.performance_history.get('success_rate', 0.5) < 0.6
        ]
        
        if len(low_performers) > len(agents) * 0.3:
            risks.append(RiskFactor(
                id="agent_performance_concerns",
                type=RiskType.PERFORMANCE,
                severity=RiskSeverity.MEDIUM,
                probability=0.6,
                impact=0.6,
                description="Multiple agents with poor performance history",
                mitigation_strategies=[
                    "Provide additional training and support",
                    "Pair low performers with high performers",
                    "Implement closer monitoring"
                ],
                detection_indicators=[
                    "Consistently missed deadlines",
                    "Quality issues from specific agents",
                    "Task reassignment frequency increasing"
                ],
                contingency_plans=[
                    "Agent replacement if available",
                    "Increased supervision and support",
                    "Task difficulty adjustment"
                ],
                timeline_impact_hours=float(len(low_performers) * 2),
                affected_components=["agent_performance"]
            ))
        
        # Strategy performance fit risks
        strategy_complexity_fit = self._assess_strategy_complexity_fit(strategy, tasks)
        if strategy_complexity_fit < 0.4:
            risks.append(RiskFactor(
                id="strategy_complexity_mismatch",
                type=RiskType.PERFORMANCE,
                severity=RiskSeverity.MEDIUM,
                probability=0.5,
                impact=0.5,
                description="Strategy may not be optimal for current task complexity",
                mitigation_strategies=[
                    "Consider alternative coordination strategy",
                    "Adjust task complexity distribution",
                    "Implement hybrid approach"
                ],
                detection_indicators=[
                    "Strategy performing below expectations",
                    "Coordination overhead higher than anticipated"
                ],
                contingency_plans=[
                    "Switch to better-fit strategy",
                    "Implement strategy adaptation"
                ],
                timeline_impact_hours=float(len(tasks) * 0.4),
                affected_components=["strategy_execution"]
            ))
        
        return risks
    
    def _detect_circular_dependencies(self, tasks: List[Task]) -> List[str]:
        """Detect circular dependencies in task list."""
        # Simple cycle detection using DFS
        task_dict = {task.id: task for task in tasks}
        visited = set()
        recursion_stack = set()
        cycles = []
        
        def dfs(task_id: str, path: List[str]) -> bool:
            if task_id in recursion_stack:
                cycle_start = path.index(task_id)
                cycles.append(" -> ".join(path[cycle_start:] + [task_id]))
                return True
            
            if task_id in visited:
                return False
            
            visited.add(task_id)
            recursion_stack.add(task_id)
            
            task = task_dict.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    if dep_id in task_dict:  # Only check dependencies that exist
                        if dfs(dep_id, path + [task_id]):
                            return True
            
            recursion_stack.remove(task_id)
            return False
        
        for task in tasks:
            if task.id not in visited:
                dfs(task.id, [])
        
        return cycles
    
    def _calculate_critical_path_length(self, tasks: List[Task]) -> float:
        """Calculate the length of the critical path."""
        # Simplified critical path calculation
        task_dict = {task.id: task for task in tasks}
        
        # Calculate earliest start times
        earliest_start = {}
        
        def calculate_earliest_start(task_id: str) -> float:
            if task_id in earliest_start:
                return earliest_start[task_id]
            
            task = task_dict.get(task_id)
            if not task:
                return 0.0
            
            if not task.dependencies:
                earliest_start[task_id] = 0.0
            else:
                max_predecessor_finish = 0.0
                for dep_id in task.dependencies:
                    if dep_id in task_dict:
                        dep_start = calculate_earliest_start(dep_id)
                        dep_duration = task_dict[dep_id].estimated_duration
                        max_predecessor_finish = max(max_predecessor_finish, dep_start + dep_duration)
                earliest_start[task_id] = max_predecessor_finish
            
            return earliest_start[task_id]
        
        # Calculate for all tasks
        for task in tasks:
            calculate_earliest_start(task.id)
        
        # Find the critical path length
        max_finish_time = 0.0
        for task in tasks:
            finish_time = earliest_start.get(task.id, 0.0) + task.estimated_duration
            max_finish_time = max(max_finish_time, finish_time)
        
        return max_finish_time
    
    def _assess_strategy_complexity_fit(self, strategy: BaseCoordinationStrategy, tasks: List[Task]) -> float:
        """Assess how well the strategy fits the task complexity."""
        if not tasks:
            return 0.5
        
        avg_complexity = sum(task.complexity for task in tasks) / len(tasks)
        
        # Strategy complexity preferences (simplified)
        strategy_preferences = {
            "Sequential": (0.0, 0.6),    # Good for low-medium complexity
            "Parallel": (0.3, 0.8),     # Good for medium-high complexity
            "Swarm-Based": (0.5, 1.0)   # Good for high complexity
        }
        
        min_pref, max_pref = strategy_preferences.get(strategy.name, (0.0, 1.0))
        
        if min_pref <= avg_complexity <= max_pref:
            return 1.0
        elif avg_complexity < min_pref:
            return max(0.0, 1.0 - (min_pref - avg_complexity) * 2)
        else:
            return max(0.0, 1.0 - (avg_complexity - max_pref) * 2)
    
    def _calculate_overall_risk_score(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate overall risk score from individual risk factors."""
        if not risk_factors:
            return 0.0
        
        # Weight risks by severity and probability
        weighted_risks = []
        for risk in risk_factors:
            severity_weight = {
                RiskSeverity.LOW: 0.25,
                RiskSeverity.MEDIUM: 0.5,
                RiskSeverity.HIGH: 0.75,
                RiskSeverity.CRITICAL: 1.0
            }[risk.severity]
            
            weighted_risk = risk.probability * risk.impact * severity_weight
            weighted_risks.append(weighted_risk)
        
        # Use a compound risk formula (not simple average)
        overall_risk = 1.0 - np.prod([1.0 - risk for risk in weighted_risks])
        
        return float(min(overall_risk, 1.0))
    
    def _generate_risk_recommendations(
        self, 
        risk_factors: List[RiskFactor], 
        strategy: BaseCoordinationStrategy
    ) -> List[str]:
        """Generate actionable recommendations based on risk assessment."""
        
        recommendations = []
        
        # Group risks by type
        risk_by_type = {}
        for risk in risk_factors:
            if risk.type not in risk_by_type:
                risk_by_type[risk.type] = []
            risk_by_type[risk.type].append(risk)
        
        # Generate type-specific recommendations
        for risk_type, risks in risk_by_type.items():
            high_risks = [r for r in risks if r.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]]
            
            if high_risks:
                type_name = risk_type.value
                recommendations.append(f"Address {len(high_risks)} high-severity {type_name} risks immediately")
                
                # Add top mitigation strategies
                all_mitigations = []
                for risk in high_risks:
                    all_mitigations.extend(risk.mitigation_strategies[:2])  # Top 2 per risk
                
                # Remove duplicates and add top recommendations
                unique_mitigations = list(dict.fromkeys(all_mitigations))[:3]
                recommendations.extend(unique_mitigations)
        
        # Strategy-specific recommendations
        if strategy.name == "Sequential" and any(r.id == "sequential_bottleneck" for r in risk_factors):
            recommendations.append("Consider switching to parallel strategy for independent tasks")
        
        if strategy.name == "Parallel" and any(r.id == "coordination_complexity" for r in risk_factors):
            recommendations.append("Implement hierarchical coordination to manage complexity")
        
        # General recommendations based on overall risk
        critical_risks = [r for r in risk_factors if r.severity == RiskSeverity.CRITICAL]
        if critical_risks:
            recommendations.insert(0, f"CRITICAL: {len(critical_risks)} critical risks require immediate attention")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _generate_contingency_strategies(
        self, 
        risk_factors: List[RiskFactor], 
        strategy: BaseCoordinationStrategy
    ) -> List[str]:
        """Generate contingency strategies for high-risk scenarios."""
        
        contingencies = []
        
        # Collect all contingency plans from high-severity risks
        high_severity_risks = [
            risk for risk in risk_factors 
            if risk.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]
        ]
        
        for risk in high_severity_risks:
            contingencies.extend(risk.contingency_plans)
        
        # Remove duplicates while preserving order
        unique_contingencies = list(dict.fromkeys(contingencies))
        
        # Add strategy-level contingencies
        if strategy.name == "Sequential":
            unique_contingencies.append("Switch to parallel execution if timeline becomes critical")
        elif strategy.name == "Parallel":
            unique_contingencies.append("Fall back to sequential execution if coordination fails")
        elif strategy.name == "Swarm-Based":
            unique_contingencies.append("Use simplified assignment algorithm if swarm optimization fails")
        
        return unique_contingencies[:8]  # Limit to top 8 contingencies
    
    def _generate_monitoring_requirements(self, risk_factors: List[RiskFactor]) -> List[str]:
        """Generate monitoring requirements for risk factors."""
        
        monitoring = []
        
        # Collect detection indicators from all risks
        all_indicators = []
        for risk in risk_factors:
            all_indicators.extend(risk.detection_indicators)
        
        # Remove duplicates and add monitoring requirements
        unique_indicators = list(dict.fromkeys(all_indicators))
        
        for indicator in unique_indicators:
            monitoring.append(f"Monitor: {indicator}")
        
        # Add general monitoring requirements
        monitoring.extend([
            "Track overall project timeline against milestones",
            "Monitor agent performance metrics and load distribution",
            "Track quality metrics and error rates",
            "Monitor communication effectiveness and coordination overhead"
        ])
        
        return monitoring[:12]  # Limit to top 12 monitoring requirements
    
    def _calculate_success_probability(self, overall_risk: float, risk_factors: List[RiskFactor]) -> float:
        """Calculate probability of successful completion."""
        
        # Base success probability inversely related to risk
        base_probability = 1.0 - overall_risk * 0.7  # Risk doesn't eliminate all success chance
        
        # Adjust for critical risks
        critical_risks = [r for r in risk_factors if r.severity == RiskSeverity.CRITICAL]
        critical_penalty = len(critical_risks) * 0.15
        
        # Adjust for high risks
        high_risks = [r for r in risk_factors if r.severity == RiskSeverity.HIGH]
        high_penalty = len(high_risks) * 0.05
        
        success_probability = base_probability - critical_penalty - high_penalty
        
        return float(max(0.1, min(0.95, success_probability)))  # Keep within reasonable bounds
    
    def _calculate_timeline_buffer(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate recommended timeline buffer in hours."""
        
        total_timeline_impact = sum(risk.timeline_impact_hours for risk in risk_factors)
        
        # Add baseline buffer
        baseline_buffer = total_timeline_impact * 0.2
        
        # Add buffer based on risk severity
        severity_buffer = 0.0
        for risk in risk_factors:
            if risk.severity == RiskSeverity.CRITICAL:
                severity_buffer += risk.timeline_impact_hours * 0.5
            elif risk.severity == RiskSeverity.HIGH:
                severity_buffer += risk.timeline_impact_hours * 0.3
            elif risk.severity == RiskSeverity.MEDIUM:
                severity_buffer += risk.timeline_impact_hours * 0.1
        
        total_buffer = baseline_buffer + severity_buffer
        
        return float(max(1.0, total_buffer))  # Minimum 1 hour buffer
    
    def _record_assessment(
        self, 
        result: RiskAssessmentResult, 
        strategy: BaseCoordinationStrategy, 
        tasks: List[Task], 
        agents: List[Agent], 
        context: Dict[str, Any]
    ):
        """Record assessment for learning and improvement."""
        
        record = {
            "timestamp": datetime.now(),
            "strategy": strategy.name,
            "task_count": len(tasks),
            "agent_count": len(agents),
            "overall_risk": result.overall_risk_score,
            "success_probability": result.success_probability,
            "risk_count_by_severity": {
                severity.value: len([r for r in result.risk_factors if r.severity == severity])
                for severity in RiskSeverity
            },
            "context": context
        }
        
        self.assessment_history.append(record)
        
        # Keep only recent history
        if len(self.assessment_history) > 100:
            self.assessment_history = self.assessment_history[-100:]
    
    def _create_fallback_assessment(self) -> RiskAssessmentResult:
        """Create a fallback assessment when analysis fails."""
        
        return RiskAssessmentResult(
            overall_risk_score=0.7,  # Assume medium-high risk
            risk_factors=[
                RiskFactor(
                    id="assessment_failure",
                    type=RiskType.TECHNICAL,
                    severity=RiskSeverity.HIGH,
                    probability=1.0,
                    impact=0.5,
                    description="Risk assessment system failure - manual assessment required",
                    mitigation_strategies=["Perform manual risk assessment", "Fix assessment system"],
                    detection_indicators=["Assessment system errors"],
                    contingency_plans=["Use conservative approach", "Manual oversight"],
                    timeline_impact_hours=2.0,
                    affected_components=["risk_assessment"]
                )
            ],
            recommendations=["Perform manual risk assessment", "Use conservative execution approach"],
            contingency_strategies=["Manual coordination oversight", "Conservative timeline planning"],
            monitoring_requirements=["Manual risk monitoring", "Frequent status checks"],
            success_probability=0.5,
            timeline_buffer_recommendation=4.0
        )
    
    def get_assessment_summary(self) -> Dict[str, Any]:
        """Get summary of historical assessments."""
        
        if not self.assessment_history:
            return {"message": "No assessment history available"}
        
        recent_assessments = self.assessment_history[-20:]  # Last 20 assessments
        
        avg_risk = float(np.mean([a["overall_risk"] for a in recent_assessments]))
        avg_success_prob = float(np.mean([a["success_probability"] for a in recent_assessments]))
        
        return {
            "total_assessments": len(self.assessment_history),
            "recent_average_risk": avg_risk,
            "recent_average_success_probability": avg_success_prob,
            "most_common_strategy": max(
                set(a["strategy"] for a in recent_assessments),
                key=lambda s: sum(1 for a in recent_assessments if a["strategy"] == s)
            ) if recent_assessments else "Unknown"
        }