"""
Fuzzy Logic Capability Matcher

This module implements fuzzy logic-based capability matching for sophisticated
agent-task assignment with uncertainty handling and multi-valued logic.
"""

import numpy as np
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MembershipFunction(Enum):
    """Types of membership functions for fuzzy sets."""
    TRIANGULAR = "triangular"
    TRAPEZOIDAL = "trapezoidal"
    GAUSSIAN = "gaussian"
    SIGMOID = "sigmoid"


@dataclass
class FuzzySet:
    """Represents a fuzzy set with a membership function."""
    name: str
    membership_function: Callable[[float], float]
    description: str = ""
    
    def membership(self, value: float) -> float:
        """Calculate membership degree for a value."""
        try:
            return max(0.0, min(1.0, self.membership_function(value)))
        except Exception as e:
            logger.error("Error calculating membership for %s: %s", self.name, str(e))
            return 0.0


@dataclass
class CapabilityMatch:
    """Represents a fuzzy capability match result."""
    agent_id: str
    capability: str
    match_degree: float  # 0.0 to 1.0
    confidence: float    # 0.0 to 1.0
    linguistic_term: str = ""
    supporting_factors: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.supporting_factors is None:
            self.supporting_factors = []


class FuzzyCapabilityMatcher:
    """
    Fuzzy logic-based capability matching for agent assignment.
    
    Uses fuzzy sets and inference rules to handle uncertainty and
    provide more nuanced capability matching than binary logic.
    """
    
    def __init__(self):
        """Initialize fuzzy capability matcher with predefined fuzzy sets."""
        self.fuzzy_sets = self._initialize_fuzzy_sets()
        self.inference_rules = self._initialize_inference_rules()
        
    def _initialize_fuzzy_sets(self) -> Dict[str, List[FuzzySet]]:
        """Initialize predefined fuzzy sets for different criteria."""
        return {
            "expertise_level": [
                FuzzySet(
                    "novice", 
                    lambda x: max(0, min(1, (0.3 - x) / 0.3)),
                    "Low expertise level"
                ),
                FuzzySet(
                    "intermediate", 
                    lambda x: max(0, min((x - 0.2) / 0.3, (0.8 - x) / 0.3)) if 0.2 <= x <= 0.8 else 0,
                    "Medium expertise level"
                ),
                FuzzySet(
                    "expert", 
                    lambda x: max(0, min(1, (x - 0.7) / 0.3)),
                    "High expertise level"
                )
            ],
            "load_level": [
                FuzzySet(
                    "low", 
                    lambda x: max(0, min(1, (0.4 - x) / 0.4)),
                    "Low system load"
                ),
                FuzzySet(
                    "medium", 
                    lambda x: max(0, min((x - 0.2) / 0.3, (0.8 - x) / 0.3)) if 0.2 <= x <= 0.8 else 0,
                    "Medium system load"
                ),
                FuzzySet(
                    "high", 
                    lambda x: max(0, min(1, (x - 0.6) / 0.4)),
                    "High system load"
                )
            ],
            "task_complexity": [
                FuzzySet(
                    "simple", 
                    lambda x: max(0, min(1, (0.3 - x) / 0.3)),
                    "Simple task"
                ),
                FuzzySet(
                    "moderate", 
                    lambda x: max(0, min((x - 0.2) / 0.3, (0.7 - x) / 0.2)) if 0.2 <= x <= 0.7 else 0,
                    "Moderate complexity task"
                ),
                FuzzySet(
                    "complex", 
                    lambda x: max(0, min(1, (x - 0.6) / 0.4)),
                    "Complex task"
                )
            ],
            "success_probability": [
                FuzzySet(
                    "low", 
                    lambda x: max(0, min(1, (0.4 - x) / 0.4)),
                    "Low success probability"
                ),
                FuzzySet(
                    "medium", 
                    lambda x: max(0, min((x - 0.3) / 0.3, (0.8 - x) / 0.2)) if 0.3 <= x <= 0.8 else 0,
                    "Medium success probability"
                ),
                FuzzySet(
                    "high", 
                    lambda x: max(0, min(1, (x - 0.7) / 0.3)),
                    "High success probability"
                )
            ],
            "response_time": [
                FuzzySet(
                    "fast", 
                    lambda x: max(0, min(1, (2.0 - x) / 2.0)),
                    "Fast response time"
                ),
                FuzzySet(
                    "normal", 
                    lambda x: max(0, min((x - 1.0) / 2.0, (5.0 - x) / 2.0)) if 1.0 <= x <= 5.0 else 0,
                    "Normal response time"
                ),
                FuzzySet(
                    "slow", 
                    lambda x: max(0, min(1, (x - 3.0) / 3.0)),
                    "Slow response time"
                )
            ]
        }
    
    def _initialize_inference_rules(self) -> List[Dict[str, Any]]:
        """Initialize fuzzy inference rules for capability matching."""
        return [
            {
                "conditions": [
                    ("expertise_level", "expert"),
                    ("load_level", "low"),
                    ("task_complexity", "complex")
                ],
                "conclusion": ("success_probability", "high"),
                "weight": 0.9,
                "description": "Expert agent with low load can handle complex tasks"
            },
            {
                "conditions": [
                    ("expertise_level", "novice"),
                    ("task_complexity", "complex")
                ],
                "conclusion": ("success_probability", "low"),
                "weight": 0.8,
                "description": "Novice agent struggles with complex tasks"
            },
            {
                "conditions": [
                    ("load_level", "high"),
                    ("response_time", "slow")
                ],
                "conclusion": ("success_probability", "low"),
                "weight": 0.7,
                "description": "High load leads to slower response and lower success"
            },
            {
                "conditions": [
                    ("expertise_level", "intermediate"),
                    ("load_level", "medium"),
                    ("task_complexity", "moderate")
                ],
                "conclusion": ("success_probability", "medium"),
                "weight": 0.6,
                "description": "Balanced conditions lead to moderate success"
            },
            {
                "conditions": [
                    ("expertise_level", "expert"),
                    ("task_complexity", "simple")
                ],
                "conclusion": ("success_probability", "high"),
                "weight": 0.8,
                "description": "Expert agent excels at simple tasks"
            }
        ]
    
    async def match_capabilities(
        self, 
        agent_capabilities: Dict[str, float],
        required_capabilities: List[str],
        task_complexity: float,
        agent_load: float = 0.5,
        agent_success_rate: float = 0.8,
        agent_response_time: float = 2.0
    ) -> List[CapabilityMatch]:
        """
        Match agent capabilities using fuzzy logic.
        
        Args:
            agent_capabilities: Dict mapping capability names to proficiency levels (0.0-1.0)
            required_capabilities: List of required capability names
            task_complexity: Task complexity level (0.0-1.0)
            agent_load: Current agent load (0.0-1.0)
            agent_success_rate: Historical success rate (0.0-1.0)
            agent_response_time: Average response time in seconds
            
        Returns:
            List of capability matches with fuzzy scores
        """
        matches = []
        
        for capability in required_capabilities:
            # Get agent's proficiency in this capability
            proficiency = agent_capabilities.get(capability, 0.0)
            
            # Calculate fuzzy match using inference
            match_result = await self._calculate_fuzzy_match(
                capability=capability,
                proficiency=proficiency,
                task_complexity=task_complexity,
                agent_load=agent_load,
                agent_success_rate=agent_success_rate,
                agent_response_time=agent_response_time
            )
            
            matches.append(match_result)
        
        return matches
    
    async def _calculate_fuzzy_match(
        self,
        capability: str,
        proficiency: float,
        task_complexity: float,
        agent_load: float,
        agent_success_rate: float,
        agent_response_time: float
    ) -> CapabilityMatch:
        """Calculate fuzzy match for a single capability."""
        
        # Determine expertise level based on proficiency
        expertise_memberships = self._calculate_memberships("expertise_level", proficiency)
        
        # Determine load level
        load_memberships = self._calculate_memberships("load_level", agent_load)
        
        # Determine task complexity level
        complexity_memberships = self._calculate_memberships("task_complexity", task_complexity)
        
        # Determine response time level
        response_memberships = self._calculate_memberships("response_time", agent_response_time)
        
        # Apply inference rules
        inferred_success = await self._apply_inference_rules({
            "expertise_level": expertise_memberships,
            "load_level": load_memberships,
            "task_complexity": complexity_memberships,
            "response_time": response_memberships
        })
        
        # Calculate overall match degree
        match_degree = self._aggregate_match_scores([
            proficiency,
            agent_success_rate,
            inferred_success
        ])
        
        # Calculate confidence based on membership strength
        confidence = self._calculate_confidence([
            max(expertise_memberships.values()),
            max(load_memberships.values()),
            max(complexity_memberships.values())
        ])
        
        # Determine linguistic term
        linguistic_term = self._get_linguistic_term(match_degree)
        
        # Generate supporting factors
        supporting_factors = self._generate_supporting_factors(
            proficiency, agent_load, task_complexity, agent_success_rate
        )
        
        return CapabilityMatch(
            agent_id="",  # Will be set by caller
            capability=capability,
            match_degree=match_degree,
            confidence=confidence,
            linguistic_term=linguistic_term,
            supporting_factors=supporting_factors
        )
    
    def _calculate_memberships(self, fuzzy_set_name: str, value: float) -> Dict[str, float]:
        """Calculate membership degrees for a value in a fuzzy set."""
        memberships = {}
        
        if fuzzy_set_name in self.fuzzy_sets:
            for fuzzy_set in self.fuzzy_sets[fuzzy_set_name]:
                memberships[fuzzy_set.name] = fuzzy_set.membership(value)
        
        return memberships
    
    async def _apply_inference_rules(self, memberships: Dict[str, Dict[str, float]]) -> float:
        """Apply fuzzy inference rules to calculate success probability."""
        rule_outputs = []
        
        for rule in self.inference_rules:
            # Calculate rule activation strength
            activation_strength = 1.0
            
            for condition_var, condition_term in rule["conditions"]:
                if condition_var in memberships and condition_term in memberships[condition_var]:
                    membership_degree = memberships[condition_var][condition_term]
                    activation_strength = min(activation_strength, membership_degree)
                else:
                    activation_strength = 0.0
                    break
            
            # Apply rule weight
            weighted_strength = activation_strength * rule["weight"]
            
            if weighted_strength > 0:
                rule_outputs.append(weighted_strength)
        
        # Aggregate rule outputs
        if rule_outputs:
            return sum(rule_outputs) / len(rule_outputs)
        else:
            return 0.5  # Default neutral value
    
    def _aggregate_match_scores(self, scores: List[float]) -> float:
        """Aggregate multiple match scores using weighted average."""
        if not scores:
            return 0.0
        
        # Use different weights for different factors
        weights = [0.4, 0.3, 0.3]  # proficiency, success_rate, inferred_success
        
        if len(scores) != len(weights):
            # Fallback to simple average
            return sum(scores) / len(scores)
        
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        return min(1.0, max(0.0, weighted_sum))
    
    def _calculate_confidence(self, membership_strengths: List[float]) -> float:
        """Calculate confidence based on membership strength."""
        if not membership_strengths:
            return 0.0
        
        # Confidence is higher when memberships are clear (close to 0 or 1)
        clarity_scores = [min(strength, 1.0 - strength) for strength in membership_strengths]
        average_clarity = sum(clarity_scores) / len(clarity_scores)
        
        # Convert clarity to confidence (inverted)
        confidence = 1.0 - (2.0 * average_clarity)
        return max(0.0, min(1.0, confidence))
    
    def _get_linguistic_term(self, match_degree: float) -> str:
        """Convert numeric match degree to linguistic term."""
        if match_degree >= 0.8:
            return "excellent_match"
        elif match_degree >= 0.6:
            return "good_match"
        elif match_degree >= 0.4:
            return "fair_match"
        elif match_degree >= 0.2:
            return "poor_match"
        else:
            return "no_match"
    
    def _generate_supporting_factors(
        self, 
        proficiency: float, 
        agent_load: float, 
        task_complexity: float, 
        success_rate: float
    ) -> List[str]:
        """Generate human-readable supporting factors for the match."""
        factors = []
        
        if proficiency >= 0.8:
            factors.append("High proficiency in required capability")
        elif proficiency >= 0.5:
            factors.append("Adequate proficiency in required capability")
        else:
            factors.append("Limited proficiency in required capability")
        
        if agent_load <= 0.3:
            factors.append("Low current workload allows focus")
        elif agent_load <= 0.7:
            factors.append("Moderate workload manageable")
        else:
            factors.append("High workload may impact performance")
        
        if task_complexity <= 0.3:
            factors.append("Simple task within agent capabilities")
        elif task_complexity <= 0.7:
            factors.append("Moderate complexity requires attention")
        else:
            factors.append("Complex task requires expert skills")
        
        if success_rate >= 0.8:
            factors.append("Strong track record of success")
        elif success_rate >= 0.6:
            factors.append("Reliable performance history")
        else:
            factors.append("Performance history shows room for improvement")
        
        return factors
    
    def add_fuzzy_set(self, category: str, fuzzy_set: FuzzySet) -> None:
        """Add a new fuzzy set to a category."""
        if category not in self.fuzzy_sets:
            self.fuzzy_sets[category] = []
        self.fuzzy_sets[category].append(fuzzy_set)
    
    def add_inference_rule(self, rule: Dict[str, Any]) -> None:
        """Add a new inference rule."""
        required_keys = ["conditions", "conclusion", "weight"]
        if all(key in rule for key in required_keys):
            self.inference_rules.append(rule)
        else:
            logger.error("Invalid inference rule format")
    
    def get_fuzzy_sets_info(self) -> Dict[str, List[Dict[str, str]]]:
        """Get information about all fuzzy sets."""
        info = {}
        for category, fuzzy_sets in self.fuzzy_sets.items():
            info[category] = [
                {
                    "name": fs.name,
                    "description": fs.description
                }
                for fs in fuzzy_sets
            ]
        return info