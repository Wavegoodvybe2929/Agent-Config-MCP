"""
Assignment Explanation Generator

This module generates detailed, human-readable explanations for agent assignments
with reasoning factors, weights, and justifications to help users understand
why specific assignments were made.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AssignmentReason:
    """Represents a specific reason for an assignment decision."""
    factor: str
    weight: float
    contribution: float
    explanation: str
    supporting_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.supporting_data is None:
            self.supporting_data = {}


@dataclass
class AssignmentExplanation:
    """Complete explanation for an agent assignment."""
    task_id: str
    agent_id: str
    overall_score: float
    confidence: float
    primary_reasons: List[AssignmentReason]
    secondary_reasons: List[AssignmentReason]
    alternative_agents: List[Dict[str, Any]]
    summary: str
    timestamp: datetime


class AssignmentExplainer:
    """
    Generate detailed explanations for agent assignments.
    
    Provides human-readable justifications for assignment decisions,
    including primary and secondary factors, alternative options,
    and confidence assessments.
    """
    
    def __init__(self):
        """Initialize assignment explainer with reasoning templates."""
        self.reason_templates = {
            "capability_match": {
                "high": "Agent {agent} has {match_score:.1%} capability match for {requirements}, demonstrating strong competency in all required areas",
                "medium": "Agent {agent} has {match_score:.1%} capability match for {requirements}, covering most required skills adequately", 
                "low": "Agent {agent} has {match_score:.1%} capability match for {requirements}, with limited coverage of required skills"
            },
            "load_balance": {
                "low": "Agent {agent} has {load:.1%} current load, providing ample capacity for additional work",
                "medium": "Agent {agent} has {load:.1%} current load, within acceptable working capacity",
                "high": "Agent {agent} has {load:.1%} current load, approaching maximum capacity but still available"
            },
            "success_rate": {
                "high": "Agent {agent} has {rate:.1%} historical success rate for similar tasks, indicating reliable performance",
                "medium": "Agent {agent} has {rate:.1%} historical success rate, showing consistent but not exceptional performance",
                "low": "Agent {agent} has {rate:.1%} historical success rate, suggesting potential challenges with task completion"
            },
            "availability": {
                "immediate": "Agent {agent} is immediately available with {response_time:.1f}s response time",
                "quick": "Agent {agent} can respond within {response_time:.1f}s, ensuring prompt task initiation",
                "delayed": "Agent {agent} has {response_time:.1f}s response time, which may delay task start"
            },
            "expertise": {
                "expert": "Agent {agent} has expert-level expertise in required domain with specialization score of {level:.2f}",
                "proficient": "Agent {agent} has proficient expertise in required domain with competency score of {level:.2f}",
                "novice": "Agent {agent} has basic expertise in required domain with learning potential score of {level:.2f}"
            },
            "task_complexity": {
                "well_suited": "Task complexity ({complexity:.1%}) is well-suited to agent's capabilities",
                "manageable": "Task complexity ({complexity:.1%}) is manageable for agent with some effort required",
                "challenging": "Task complexity ({complexity:.1%}) presents a significant challenge for agent"
            },
            "optimization": {
                "optimal": "This assignment was identified as optimal through swarm intelligence optimization",
                "near_optimal": "This assignment scored among the top solutions in optimization analysis",
                "acceptable": "This assignment meets minimum requirements while balancing other constraints"
            }
        }
        
        self.factor_weights = {
            "capability_match": 0.30,
            "load_balance": 0.20,
            "success_rate": 0.20,
            "expertise": 0.15,
            "availability": 0.10,
            "task_complexity": 0.05
        }
    
    async def generate_explanation(
        self,
        agent: Any,
        task: Any,
        assignment_context: Dict[str, Any]
    ) -> List[AssignmentReason]:
        """
        Generate detailed explanation for a single assignment.
        
        Args:
            agent: Agent object with capabilities and performance data
            task: Task object with requirements and complexity
            assignment_context: Context information about the assignment process
            
        Returns:
            List of assignment reasons explaining the decision
        """
        reasons = []
        
        try:
            # Analyze capability match
            capability_reason = await self._analyze_capability_match(agent, task)
            if capability_reason:
                reasons.append(capability_reason)
            
            # Analyze load balance
            load_reason = await self._analyze_load_balance(agent)
            if load_reason:
                reasons.append(load_reason)
            
            # Analyze success rate
            success_reason = await self._analyze_success_rate(agent, task)
            if success_reason:
                reasons.append(success_reason)
            
            # Analyze expertise level
            expertise_reason = await self._analyze_expertise(agent, task)
            if expertise_reason:
                reasons.append(expertise_reason)
            
            # Analyze availability
            availability_reason = await self._analyze_availability(agent)
            if availability_reason:
                reasons.append(availability_reason)
            
            # Analyze task complexity fit
            complexity_reason = await self._analyze_task_complexity(agent, task)
            if complexity_reason:
                reasons.append(complexity_reason)
            
            # Add optimization context
            optimization_reason = await self._analyze_optimization_context(assignment_context)
            if optimization_reason:
                reasons.append(optimization_reason)
            
            return reasons
            
        except Exception as e:
            logger.error("Error generating assignment explanation: %s", str(e))
            return [AssignmentReason(
                factor="error",
                weight=0.0,
                contribution=0.0,
                explanation=f"Error generating explanation: {str(e)}"
            )]
    
    async def _analyze_capability_match(self, agent: Any, task: Any) -> Optional[AssignmentReason]:
        """Analyze capability match between agent and task."""
        try:
            # Calculate capability match score
            required_caps = set(getattr(task, 'requirements', []))
            agent_caps = set(getattr(agent, 'capabilities', []))
            
            if not required_caps:
                match_score = 1.0
                matched_caps = set()
            else:
                matched_caps = required_caps.intersection(agent_caps)
                match_score = len(matched_caps) / len(required_caps)
            
            # Determine match level
            if match_score >= 0.8:
                level = "high"
            elif match_score >= 0.5:
                level = "medium"
            else:
                level = "low"
            
            # Generate explanation
            explanation = self.reason_templates["capability_match"][level].format(
                agent=getattr(agent, 'id', 'unknown'),
                match_score=match_score,
                requirements=", ".join(required_caps) if required_caps else "none"
            )
            
            # Calculate contribution
            contribution = match_score * self.factor_weights["capability_match"]
            
            return AssignmentReason(
                factor="capability_match",
                weight=self.factor_weights["capability_match"],
                contribution=contribution,
                explanation=explanation,
                supporting_data={
                    "match_score": match_score,
                    "required_capabilities": list(required_caps),
                    "agent_capabilities": list(agent_caps),
                    "matched_capabilities": list(matched_caps) if required_caps else []
                }
            )
            
        except Exception as e:
            logger.error("Error analyzing capability match: %s", str(e))
            return None
    
    async def _analyze_load_balance(self, agent: Any) -> Optional[AssignmentReason]:
        """Analyze agent's current load balance."""
        try:
            load = getattr(agent, 'current_load', 0.5)
            
            # Determine load level
            if load <= 0.3:
                level = "low"
            elif load <= 0.7:
                level = "medium"
            else:
                level = "high"
            
            explanation = self.reason_templates["load_balance"][level].format(
                agent=getattr(agent, 'id', 'unknown'),
                load=load
            )
            
            # Higher contribution for lower load (inverted)
            contribution = (1.0 - load) * self.factor_weights["load_balance"]
            
            return AssignmentReason(
                factor="load_balance",
                weight=self.factor_weights["load_balance"],
                contribution=contribution,
                explanation=explanation,
                supporting_data={
                    "current_load": load,
                    "load_level": level
                }
            )
            
        except Exception as e:
            logger.error("Error analyzing load balance: %s", str(e))
            return None
    
    async def _analyze_success_rate(self, agent: Any, task: Any) -> Optional[AssignmentReason]:
        """Analyze agent's historical success rate."""
        try:
            success_rate = getattr(agent, 'success_rate', 0.8)
            
            # Determine success level
            if success_rate >= 0.8:
                level = "high"
            elif success_rate >= 0.6:
                level = "medium"
            else:
                level = "low"
            
            explanation = self.reason_templates["success_rate"][level].format(
                agent=getattr(agent, 'id', 'unknown'),
                rate=success_rate
            )
            
            contribution = success_rate * self.factor_weights["success_rate"]
            
            return AssignmentReason(
                factor="success_rate",
                weight=self.factor_weights["success_rate"],
                contribution=contribution,
                explanation=explanation,
                supporting_data={
                    "success_rate": success_rate,
                    "success_level": level,
                    "total_tasks": getattr(agent, 'total_tasks_completed', 0)
                }
            )
            
        except Exception as e:
            logger.error("Error analyzing success rate: %s", str(e))
            return None
    
    async def _analyze_expertise(self, agent: Any, task: Any) -> Optional[AssignmentReason]:
        """Analyze agent's expertise level for the task."""
        try:
            # Get specialty weights for required capabilities
            specialty_weights = getattr(agent, 'specialty_weights', {})
            requirements = getattr(task, 'requirements', [])
            
            if requirements:
                expertise_scores = [specialty_weights.get(req, 0.5) for req in requirements]
                avg_expertise = sum(expertise_scores) / len(expertise_scores)
            else:
                avg_expertise = 0.5
            
            # Determine expertise level
            if avg_expertise >= 0.8:
                level = "expert"
            elif avg_expertise >= 0.5:
                level = "proficient"
            else:
                level = "novice"
            
            explanation = self.reason_templates["expertise"][level].format(
                agent=getattr(agent, 'id', 'unknown'),
                level=avg_expertise
            )
            
            contribution = avg_expertise * self.factor_weights["expertise"]
            
            return AssignmentReason(
                factor="expertise",
                weight=self.factor_weights["expertise"],
                contribution=contribution,
                explanation=explanation,
                supporting_data={
                    "expertise_score": avg_expertise,
                    "expertise_level": level,
                    "specialty_weights": specialty_weights
                }
            )
            
        except Exception as e:
            logger.error("Error analyzing expertise: %s", str(e))
            return None
    
    async def _analyze_availability(self, agent: Any) -> Optional[AssignmentReason]:
        """Analyze agent's availability and response time."""
        try:
            response_time = getattr(agent, 'average_completion_time', 2.0)
            availability = getattr(agent, 'availability', True)
            
            if not availability:
                explanation = f"Agent {getattr(agent, 'id', 'unknown')} is currently unavailable"
                contribution = 0.0
            else:
                # Determine response level
                if response_time <= 1.0:
                    level = "immediate"
                elif response_time <= 3.0:
                    level = "quick"
                else:
                    level = "delayed"
                
                explanation = self.reason_templates["availability"][level].format(
                    agent=getattr(agent, 'id', 'unknown'),
                    response_time=response_time
                )
                
                # Lower response time = higher contribution
                contribution = (1.0 / (1.0 + response_time)) * self.factor_weights["availability"]
            
            return AssignmentReason(
                factor="availability",
                weight=self.factor_weights["availability"],
                contribution=contribution,
                explanation=explanation,
                supporting_data={
                    "response_time": response_time,
                    "availability": availability
                }
            )
            
        except Exception as e:
            logger.error("Error analyzing availability: %s", str(e))
            return None
    
    async def _analyze_task_complexity(self, agent: Any, task: Any) -> Optional[AssignmentReason]:
        """Analyze fit between task complexity and agent capabilities."""
        try:
            task_complexity = getattr(task, 'complexity', 0.5)
            agent_success_rate = getattr(agent, 'success_rate', 0.8)
            
            # Determine complexity fit
            if agent_success_rate >= 0.8 and task_complexity <= 0.6:
                level = "well_suited"
                contribution = 1.0
            elif agent_success_rate >= 0.6:
                level = "manageable"
                contribution = 0.7
            else:
                level = "challenging"
                contribution = 0.4
            
            explanation = self.reason_templates["task_complexity"][level].format(
                complexity=task_complexity
            )
            
            contribution *= self.factor_weights["task_complexity"]
            
            return AssignmentReason(
                factor="task_complexity",
                weight=self.factor_weights["task_complexity"],
                contribution=contribution,
                explanation=explanation,
                supporting_data={
                    "task_complexity": task_complexity,
                    "complexity_fit": level
                }
            )
            
        except Exception as e:
            logger.error("Error analyzing task complexity: %s", str(e))
            return None
    
    async def _analyze_optimization_context(self, context: Dict[str, Any]) -> Optional[AssignmentReason]:
        """Analyze optimization context and algorithm used."""
        try:
            optimization_used = context.get("optimization_used", "unknown")
            total_agents = context.get("total_agents", 1)
            total_tasks = context.get("total_tasks", 1)
            
            if optimization_used == "ACO":
                level = "optimal"
                explanation = "This assignment was identified as optimal through Ant Colony Optimization algorithm"
            elif optimization_used in ["PSO", "MCDA"]:
                level = "near_optimal"
                explanation = f"This assignment scored among the top solutions using {optimization_used} analysis"
            else:
                level = "acceptable"
                explanation = "This assignment meets minimum requirements while balancing constraints"
            
            return AssignmentReason(
                factor="optimization",
                weight=0.0,  # Not counted in weighted score
                contribution=0.0,
                explanation=explanation,
                supporting_data={
                    "optimization_method": optimization_used,
                    "total_agents": total_agents,
                    "total_tasks": total_tasks
                }
            )
            
        except Exception as e:
            logger.error("Error analyzing optimization context: %s", str(e))
            return None
    
    def generate_summary(self, reasons: List[AssignmentReason], overall_score: float) -> str:
        """Generate a concise summary of the assignment decision."""
        if not reasons:
            return "Assignment made with limited information available."
        
        # Find top contributing factors
        primary_factors = sorted(reasons, key=lambda r: r.contribution, reverse=True)[:3]
        
        summary_parts = []
        
        # Overall assessment
        if overall_score >= 0.8:
            summary_parts.append("Excellent assignment match.")
        elif overall_score >= 0.6:
            summary_parts.append("Good assignment match.")
        elif overall_score >= 0.4:
            summary_parts.append("Acceptable assignment match.")
        else:
            summary_parts.append("Suboptimal assignment match.")
        
        # Top factors
        if primary_factors:
            factor_names = [self._humanize_factor_name(f.factor) for f in primary_factors]
            summary_parts.append(f"Primary factors: {', '.join(factor_names)}.")
        
        return " ".join(summary_parts)
    
    def _humanize_factor_name(self, factor: str) -> str:
        """Convert factor name to human-readable format."""
        human_names = {
            "capability_match": "capability alignment",
            "load_balance": "workload management",
            "success_rate": "performance history",
            "expertise": "domain expertise",
            "availability": "immediate availability",
            "task_complexity": "complexity suitability",
            "optimization": "algorithmic optimization"
        }
        return human_names.get(factor, factor.replace("_", " "))
    
    def update_factor_weights(self, new_weights: Dict[str, float]) -> bool:
        """
        Update factor weights for explanation generation.
        
        Args:
            new_weights: Dictionary mapping factor names to new weights
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Validate weights
            for factor, weight in new_weights.items():
                if factor in self.factor_weights and 0.0 <= weight <= 1.0:
                    self.factor_weights[factor] = weight
            
            # Normalize weights to sum to 1.0
            total_weight = sum(self.factor_weights.values())
            if total_weight > 0:
                for factor in self.factor_weights:
                    self.factor_weights[factor] /= total_weight
                return True
            
            return False
            
        except Exception as e:
            logger.error("Error updating factor weights: %s", str(e))
            return False