"""
Minority Opinion Preservation for Swarm Consensus

This module preserves and analyzes minority opinions from consensus processes
to ensure valuable dissenting views are captured, documented, and can inform
future decisions in the MCP Swarm Intelligence Server.
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter
import statistics
import hashlib

from .consensus_algorithms import ConsensusResult, Proposal

logger = logging.getLogger(__name__)


@dataclass
class MinorityOpinion:
    """Represents a minority opinion in a consensus process."""
    opinion_id: str
    agent_ids: List[str]
    proposal_supported: Proposal
    reasoning: List[str]
    confidence_level: float
    strength: float  # How strongly they disagreed with majority
    timestamp: datetime = field(default_factory=datetime.now)
    validation_triggers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert minority opinion to dictionary."""
        return {
            "opinion_id": self.opinion_id,
            "agent_ids": self.agent_ids,
            "proposal_supported": self.proposal_supported.to_dict(),
            "reasoning": self.reasoning,
            "confidence_level": self.confidence_level,
            "strength": self.strength,
            "timestamp": self.timestamp.isoformat(),
            "validation_triggers": self.validation_triggers
        }


@dataclass
class DissentReason:
    """Represents a reason for dissenting from majority opinion."""
    reason_type: str  # 'technical', 'strategic', 'ethical', 'practical', 'risk'
    description: str
    agents_citing: List[str]
    frequency: int
    impact_assessment: str  # 'low', 'medium', 'high'
    related_past_decisions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dissent reason to dictionary."""
        return {
            "reason_type": self.reason_type,
            "description": self.description,
            "agents_citing": self.agents_citing,
            "frequency": self.frequency,
            "impact_assessment": self.impact_assessment,
            "related_past_decisions": self.related_past_decisions
        }


@dataclass
class ValidationTrigger:
    """Represents a trigger for future validation of minority views."""
    trigger_id: str
    condition: str
    minority_opinion_id: str
    trigger_type: str  # 'time_based', 'outcome_based', 'metric_based', 'event_based'
    trigger_parameters: Dict[str, Any]
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation trigger to dictionary."""
        return {
            "trigger_id": self.trigger_id,
            "condition": self.condition,
            "minority_opinion_id": self.minority_opinion_id,
            "trigger_type": self.trigger_type,
            "trigger_parameters": self.trigger_parameters,
            "active": self.active,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class MinorityOpinionRecord:
    """Complete record of minority opinions from a consensus process."""
    consensus_id: str
    minority_opinions: List[MinorityOpinion]
    dissent_reasons: List[DissentReason]
    validation_triggers: List[ValidationTrigger]
    preservation_strategies_used: List[str]
    analysis_summary: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert minority opinion record to dictionary."""
        return {
            "consensus_id": self.consensus_id,
            "minority_opinions": [mo.to_dict() for mo in self.minority_opinions],
            "dissent_reasons": [dr.to_dict() for dr in self.dissent_reasons],
            "validation_triggers": [vt.to_dict() for vt in self.validation_triggers],
            "preservation_strategies_used": self.preservation_strategies_used,
            "analysis_summary": self.analysis_summary,
            "created_at": self.created_at.isoformat()
        }


class MinorityOpinionPreserver:
    """Preserves and analyzes minority opinions from consensus processes."""

    def __init__(self):
        """Initialize minority opinion preserving system."""
        self.preservation_strategies = {
            "documentation": self._document_minority_views,
            "alternative_scenarios": self._create_alternative_scenarios,
            "dissent_analysis": self._analyze_dissent_patterns,
            "future_validation": self._setup_future_validation
        }
        
        # Historical storage
        self.historical_minority_opinions: List[MinorityOpinionRecord] = []
        self.dissent_pattern_analysis: Dict[str, Any] = {}
        self.validation_triggers: List[ValidationTrigger] = []
        
        # Pattern tracking
        self.agent_dissent_patterns: Dict[str, List[str]] = defaultdict(list)
        self.topic_dissent_patterns: Dict[str, List[str]] = defaultdict(list)

    async def preserve_minority_opinions(
        self,
        consensus_result: ConsensusResult
    ) -> MinorityOpinionRecord:
        """
        Preserve and analyze minority opinions from consensus process.
        
        Args:
            consensus_result: Result of consensus process to analyze
            
        Returns:
            MinorityOpinionRecord with preserved minority views
        """
        # Extract minority opinions
        minority_opinions = self._identify_minority_opinions(
            consensus_result.vote_summary,
            consensus_result.winning_proposal
        )
        
        # Analyze dissent reasoning
        dissent_reasons = self._analyze_dissent_reasoning(minority_opinions)
        
        # Apply preservation strategies
        strategies_applied = []
        analysis_results = {}
        
        for strategy_name, strategy_func in self.preservation_strategies.items():
            try:
                result = await strategy_func(minority_opinions, consensus_result)
                analysis_results[strategy_name] = result
                strategies_applied.append(strategy_name)
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning("Error applying strategy %s: %s", strategy_name, e)
                
        # Create validation triggers
        validation_triggers = self._create_validation_triggers(
            minority_opinions, consensus_result
        )
        
        # Create minority opinion record
        consensus_id = self._generate_consensus_id(consensus_result)
        minority_record = MinorityOpinionRecord(
            consensus_id=consensus_id,
            minority_opinions=minority_opinions,
            dissent_reasons=dissent_reasons,
            validation_triggers=validation_triggers,
            preservation_strategies_used=strategies_applied,
            analysis_summary=analysis_results
        )
        
        # Store for historical analysis
        self._update_historical_records(minority_record)
        
        return minority_record

    def _identify_minority_opinions(
        self,
        vote_summary: Dict[str, Any],
        winning_proposal: Proposal
    ) -> List[MinorityOpinion]:
        """Identify and categorize minority opinions."""
        minority_opinions = []
        
        # Extract vote information
        vote_scores = vote_summary.get("vote_scores", {})
        
        # Identify non-winning proposals as minority opinions
        for proposal_id, score in vote_scores.items():
            if proposal_id != str(winning_proposal.id):
                # Create a minority opinion for this proposal
                # In real implementation, this would extract actual agent IDs and reasoning
                
                # Simulate agent IDs who voted for this proposal
                supporting_agents = [f"agent_{i}" for i in range(max(1, int(score)))]
                
                # Create proposal object (simplified)
                minority_proposal = Proposal(
                    id=int(proposal_id),
                    content=f"Alternative proposal {proposal_id}",
                    topic=winning_proposal.topic
                )
                
                # Calculate strength of dissent
                total_score = sum(vote_scores.values())
                dissent_strength = score / total_score if total_score > 0 else 0
                
                # Extract reasoning (simplified)
                reasoning = [f"Support for proposal {proposal_id} based on alternative analysis"]
                
                # Calculate average confidence
                avg_confidence = min(1.0, score / len(supporting_agents)) if supporting_agents else 0.5
                
                opinion_id = self._generate_opinion_id(minority_proposal, supporting_agents)
                
                minority_opinion = MinorityOpinion(
                    opinion_id=opinion_id,
                    agent_ids=supporting_agents,
                    proposal_supported=minority_proposal,
                    reasoning=reasoning,
                    confidence_level=avg_confidence,
                    strength=dissent_strength
                )
                
                minority_opinions.append(minority_opinion)
                
        return minority_opinions

    def _analyze_dissent_reasoning(
        self,
        minority_opinions: List[MinorityOpinion]
    ) -> List[DissentReason]:
        """Analyze reasoning behind minority dissent."""
        dissent_reasons = []
        
        # Collect all reasoning statements
        all_reasoning = []
        for opinion in minority_opinions:
            all_reasoning.extend(opinion.reasoning)
            
        # Categorize dissent reasons
        reason_categories = {
            "technical": ["implementation", "architecture", "performance", "scalability"],
            "strategic": ["direction", "priority", "timeline", "resource"],
            "ethical": ["privacy", "security", "fairness", "transparency"],
            "practical": ["feasibility", "cost", "complexity", "maintenance"],
            "risk": ["uncertainty", "failure", "impact", "consequence"]
        }
        
        # Analyze reasoning patterns
        for reason_type, keywords in reason_categories.items():
            matching_reasoning = []
            citing_agents = set()
            
            for opinion in minority_opinions:
                for reasoning_text in opinion.reasoning:
                    if any(keyword in reasoning_text.lower() for keyword in keywords):
                        matching_reasoning.append(reasoning_text)
                        citing_agents.update(opinion.agent_ids)
                        
            if matching_reasoning:
                # Assess impact based on frequency and agent involvement
                frequency = len(matching_reasoning)
                agent_count = len(citing_agents)
                
                if frequency > 3 or agent_count > 2:
                    impact = "high"
                elif frequency > 1 or agent_count > 1:
                    impact = "medium"
                else:
                    impact = "low"
                    
                dissent_reason = DissentReason(
                    reason_type=reason_type,
                    description=f"Dissent based on {reason_type} concerns",
                    agents_citing=list(citing_agents),
                    frequency=frequency,
                    impact_assessment=impact
                )
                
                dissent_reasons.append(dissent_reason)
                
        return dissent_reasons

    async def _document_minority_views(
        self,
        minority_opinions: List[MinorityOpinion],
        consensus_result: ConsensusResult
    ) -> Dict[str, Any]:
        """Document minority views for future reference."""
        documentation = {
            "total_minority_opinions": len(minority_opinions),
            "consensus_algorithm": consensus_result.algorithm_used,
            "consensus_strength": consensus_result.consensus_strength,
            "minority_strength_distribution": [],
            "documented_alternatives": []
        }
        
        for opinion in minority_opinions:
            documentation["minority_strength_distribution"].append(opinion.strength)
            
            alternative_doc = {
                "proposal_id": opinion.proposal_supported.id,
                "supporting_agents": len(opinion.agent_ids),
                "confidence": opinion.confidence_level,
                "key_reasoning": opinion.reasoning[:3]  # Top 3 reasons
            }
            documentation["documented_alternatives"].append(alternative_doc)
            
        # Add summary statistics
        if documentation["minority_strength_distribution"]:
            strengths = documentation["minority_strength_distribution"]
            documentation["minority_summary"] = {
                "average_strength": statistics.mean(strengths),
                "max_strength": max(strengths),
                "total_alternatives": len(minority_opinions)
            }
            
        return documentation

    async def _create_alternative_scenarios(
        self,
        minority_opinions: List[MinorityOpinion],
        _consensus_result: ConsensusResult
    ) -> Dict[str, Any]:
        """Create alternative scenarios based on minority opinions."""
        scenarios = {
            "alternative_outcomes": [],
            "risk_assessments": [],
            "implementation_variants": []
        }
        
        for opinion in minority_opinions:
            # Create alternative outcome scenario
            outcome_scenario = {
                "scenario_id": f"alt_{opinion.opinion_id}",
                "proposal": opinion.proposal_supported.to_dict(),
                "supporting_rationale": opinion.reasoning,
                "estimated_probability": opinion.confidence_level,
                "potential_benefits": self._extract_benefits(opinion.reasoning),
                "potential_risks": self._extract_risks(opinion.reasoning)
            }
            scenarios["alternative_outcomes"].append(outcome_scenario)
            
            # Risk assessment for not considering this alternative
            risk_assessment = {
                "ignored_alternative": opinion.proposal_supported.id,
                "risk_level": self._assess_alternative_risk(opinion),
                "mitigation_strategies": self._suggest_mitigation(opinion),
                "monitoring_recommendations": self._suggest_monitoring(opinion)
            }
            scenarios["risk_assessments"].append(risk_assessment)
            
        return scenarios

    async def _analyze_dissent_patterns(
        self,
        minority_opinions: List[MinorityOpinion],
        consensus_result: ConsensusResult
    ) -> Dict[str, Any]:
        """Analyze dissent patterns for insights."""
        pattern_analysis = {
            "agent_dissent_frequency": {},
            "topic_dissent_patterns": {},
            "reasoning_clusters": {},
            "temporal_patterns": {}
        }
        
        # Analyze agent dissent patterns
        for opinion in minority_opinions:
            for agent_id in opinion.agent_ids:
                if agent_id not in pattern_analysis["agent_dissent_frequency"]:
                    pattern_analysis["agent_dissent_frequency"][agent_id] = 0
                pattern_analysis["agent_dissent_frequency"][agent_id] += 1
                
                # Update historical tracking
                self.agent_dissent_patterns[agent_id].append(consensus_result.winning_proposal.topic)
                
        # Analyze topic patterns
        topic = consensus_result.winning_proposal.topic
        if topic not in pattern_analysis["topic_dissent_patterns"]:
            pattern_analysis["topic_dissent_patterns"][topic] = {
                "frequency": 0,
                "common_alternatives": [],
                "typical_reasoning": []
            }
            
        pattern_analysis["topic_dissent_patterns"][topic]["frequency"] += len(minority_opinions)
        
        # Cluster similar reasoning
        reasoning_clusters = self._cluster_reasoning([
            reason for opinion in minority_opinions for reason in opinion.reasoning
        ])
        pattern_analysis["reasoning_clusters"] = reasoning_clusters
        
        return pattern_analysis

    async def _setup_future_validation(
        self,
        minority_opinions: List[MinorityOpinion],
        _consensus_result: ConsensusResult
    ) -> Dict[str, Any]:
        """Setup future validation triggers for minority opinions."""
        validation_setup = {
            "triggers_created": 0,
            "validation_strategies": [],
            "monitoring_points": []
        }
        
        for opinion in minority_opinions:
            # Create time-based validation trigger
            time_trigger = ValidationTrigger(
                trigger_id=f"time_{opinion.opinion_id}",
                condition="Check minority opinion validity after implementation period",
                minority_opinion_id=opinion.opinion_id,
                trigger_type="time_based",
                trigger_parameters={
                    "check_after_days": 30,
                    "check_until_days": 90
                }
            )
            self.validation_triggers.append(time_trigger)
            validation_setup["triggers_created"] += 1
            
            # Create outcome-based validation trigger
            outcome_trigger = ValidationTrigger(
                trigger_id=f"outcome_{opinion.opinion_id}",
                condition="Validate if predicted concerns materialized",
                minority_opinion_id=opinion.opinion_id,
                trigger_type="outcome_based",
                trigger_parameters={
                    "monitor_metrics": ["success_rate", "performance", "user_satisfaction"],
                    "threshold_for_validation": 0.7
                }
            )
            self.validation_triggers.append(outcome_trigger)
            validation_setup["triggers_created"] += 1
            
        return validation_setup

    def _create_validation_triggers(
        self,
        minority_opinions: List[MinorityOpinion],
        _consensus_result: ConsensusResult
    ) -> List[ValidationTrigger]:
        """Create triggers for future validation of minority views."""
        triggers = []
        
        for opinion in minority_opinions:
            # Event-based trigger for high-confidence minority opinions
            if opinion.confidence_level > 0.7:
                event_trigger = ValidationTrigger(
                    trigger_id=f"event_{opinion.opinion_id}",
                    condition="High-confidence minority opinion requires validation",
                    minority_opinion_id=opinion.opinion_id,
                    trigger_type="event_based",
                    trigger_parameters={
                        "events_to_monitor": ["system_failure", "performance_degradation", "user_complaints"],
                        "confidence_threshold": opinion.confidence_level
                    }
                )
                triggers.append(event_trigger)
                
        return triggers

    def _generate_consensus_id(self, consensus_result: ConsensusResult) -> str:
        """Generate unique ID for consensus process."""
        content = f"{consensus_result.winning_proposal.id}_{consensus_result.algorithm_used}_{consensus_result.completed_at}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _generate_opinion_id(self, proposal: Proposal, agent_ids: List[str]) -> str:
        """Generate unique ID for minority opinion."""
        content = f"{proposal.id}_{proposal.topic}_{''.join(sorted(agent_ids))}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _extract_benefits(self, reasoning: List[str]) -> List[str]:
        """Extract potential benefits from reasoning."""
        benefits = []
        benefit_keywords = ["advantage", "benefit", "improvement", "better", "optimize", "enhance"]
        
        for reason in reasoning:
            if any(keyword in reason.lower() for keyword in benefit_keywords):
                benefits.append(reason)
                
        return benefits[:3]  # Limit to top 3

    def _extract_risks(self, reasoning: List[str]) -> List[str]:
        """Extract potential risks from reasoning."""
        risks = []
        risk_keywords = ["risk", "problem", "issue", "concern", "danger", "failure"]
        
        for reason in reasoning:
            if any(keyword in reason.lower() for keyword in risk_keywords):
                risks.append(reason)
                
        return risks[:3]  # Limit to top 3

    def _assess_alternative_risk(self, opinion: MinorityOpinion) -> str:
        """Assess risk level of ignoring alternative."""
        if opinion.confidence_level > 0.8 and opinion.strength > 0.3:
            return "high"
        elif opinion.confidence_level > 0.6 and opinion.strength > 0.2:
            return "medium"
        else:
            return "low"

    def _suggest_mitigation(self, opinion: MinorityOpinion) -> List[str]:
        """Suggest mitigation strategies for minority concerns."""
        strategies = [
            "Monitor implementation for concerns raised in minority opinion",
            f"Consider hybrid approach incorporating elements of proposal {opinion.proposal_supported.id}",
            "Establish feedback mechanism for minority opinion validation"
        ]
        
        if opinion.confidence_level > 0.7:
            strategies.append("Plan contingency implementation of minority proposal")
            
        return strategies

    def _suggest_monitoring(self, _opinion: MinorityOpinion) -> List[str]:
        """Suggest monitoring recommendations."""
        return [
            "Track key performance indicators relevant to minority concerns",
            "Regular check-ins with dissenting agents",
            "Monitor for emergence of predicted issues",
            "Document lessons learned for future similar decisions"
        ]

    def _cluster_reasoning(self, reasoning_list: List[str]) -> Dict[str, Any]:
        """Cluster similar reasoning statements."""
        # Simplified clustering based on keywords
        clusters = defaultdict(list)
        
        cluster_keywords = {
            "technical": ["implementation", "code", "architecture", "system"],
            "business": ["cost", "revenue", "market", "customer"],
            "operational": ["process", "workflow", "team", "resource"],
            "strategic": ["long-term", "vision", "direction", "goal"]
        }
        
        for reasoning in reasoning_list:
            categorized = False
            for cluster_name, keywords in cluster_keywords.items():
                if any(keyword in reasoning.lower() for keyword in keywords):
                    clusters[cluster_name].append(reasoning)
                    categorized = True
                    break
                    
            if not categorized:
                clusters["other"].append(reasoning)
                
        return dict(clusters)

    def _update_historical_records(self, minority_record: MinorityOpinionRecord) -> None:
        """Update historical records for pattern analysis."""
        self.historical_minority_opinions.append(minority_record)
        
        # Keep only recent history (last 100 records)
        if len(self.historical_minority_opinions) > 100:
            self.historical_minority_opinions = self.historical_minority_opinions[-100:]

    def get_minority_opinion_statistics(self) -> Dict[str, Any]:
        """Get statistics about minority opinion preservation."""
        if not self.historical_minority_opinions:
            return {"message": "No minority opinion data available"}
            
        total_opinions = sum(len(record.minority_opinions) for record in self.historical_minority_opinions)
        total_triggers = sum(len(record.validation_triggers) for record in self.historical_minority_opinions)
        
        return {
            "total_consensus_processes": len(self.historical_minority_opinions),
            "total_minority_opinions": total_opinions,
            "total_validation_triggers": total_triggers,
            "average_opinions_per_consensus": total_opinions / len(self.historical_minority_opinions),
            "most_frequent_dissent_agents": self._get_frequent_dissenters(),
            "most_frequent_dissent_topics": self._get_frequent_dissent_topics(),
            "validation_trigger_types": self._get_trigger_type_distribution()
        }

    def _get_frequent_dissenters(self) -> List[Dict[str, Any]]:
        """Get agents who frequently hold minority opinions."""
        agent_counts = Counter()
        for record in self.historical_minority_opinions:
            for opinion in record.minority_opinions:
                for agent_id in opinion.agent_ids:
                    agent_counts[agent_id] += 1
                    
        return [{"agent_id": agent, "dissent_count": count} 
                for agent, count in agent_counts.most_common(5)]

    def _get_frequent_dissent_topics(self) -> List[Dict[str, Any]]:
        """Get topics that frequently generate minority opinions."""
        topic_counts = Counter()
        for record in self.historical_minority_opinions:
            for opinion in record.minority_opinions:
                topic_counts[opinion.proposal_supported.topic] += 1
                
        return [{"topic": topic, "dissent_count": count} 
                for topic, count in topic_counts.most_common(5)]

    def _get_trigger_type_distribution(self) -> Dict[str, int]:
        """Get distribution of validation trigger types."""
        trigger_types = Counter()
        for trigger in self.validation_triggers:
            trigger_types[trigger.trigger_type] += 1
            
        return dict(trigger_types)