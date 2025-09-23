"""
Decision Confidence Scoring for Swarm Consensus

This module implements comprehensive confidence scoring for consensus decisions
in the MCP Swarm Intelligence Server, analyzing multiple factors to quantify
decision reliability and uncertainty.
"""

import numpy as np
import logging
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter, defaultdict
import statistics

from .consensus_algorithms import ConsensusResult
from ..swarm.decisions import Vote

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """Comprehensive confidence score for a consensus decision."""
    overall_confidence: float  # 0.0 to 1.0
    vote_distribution_confidence: float
    agent_expertise_confidence: float
    historical_accuracy_confidence: float
    information_quality_confidence: float
    consensus_strength_confidence: float
    uncertainty_level: float  # 0.0 (certain) to 1.0 (uncertain)
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    calculation_timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert confidence score to dictionary."""
        return {
            "overall_confidence": self.overall_confidence,
            "vote_distribution_confidence": self.vote_distribution_confidence,
            "agent_expertise_confidence": self.agent_expertise_confidence,
            "historical_accuracy_confidence": self.historical_accuracy_confidence,
            "information_quality_confidence": self.information_quality_confidence,
            "consensus_strength_confidence": self.consensus_strength_confidence,
            "uncertainty_level": self.uncertainty_level,
            "confidence_factors": self.confidence_factors,
            "calculation_timestamp": self.calculation_timestamp.isoformat()
        }


class DecisionConfidenceScorer:
    """Analyzes and scores confidence in consensus decisions."""

    def __init__(self):
        """Initialize decision confidence scorer."""
        self.confidence_factors = {
            "vote_distribution": self._analyze_vote_distribution,
            "agent_expertise": self._analyze_agent_expertise,
            "historical_accuracy": self._analyze_historical_accuracy,
            "information_quality": self._analyze_information_quality,
            "consensus_strength": self._analyze_consensus_strength
        }
        
        # Historical data for learning
        self.decision_history: List[Dict[str, Any]] = []
        self.agent_accuracy_history: Dict[str, List[float]] = defaultdict(list)
        self.confidence_weights = {
            "vote_distribution": 0.25,
            "agent_expertise": 0.20,
            "historical_accuracy": 0.20,
            "information_quality": 0.15,
            "consensus_strength": 0.20
        }

    async def calculate_decision_confidence(
        self,
        consensus_result: ConsensusResult
    ) -> ConfidenceScore:
        """
        Calculate comprehensive confidence score for consensus decision.
        
        Args:
            consensus_result: Result of consensus process to analyze
            
        Returns:
            ConfidenceScore with detailed confidence analysis
        """
        # Extract votes from consensus result
        votes = self._extract_votes_from_result(consensus_result)
        
        # Calculate individual confidence factors
        confidence_scores = {}
        for factor_name, analyzer_func in self.confidence_factors.items():
            try:
                score = await analyzer_func(consensus_result, votes)
                confidence_scores[factor_name] = max(0.0, min(1.0, score))
            except (ValueError, TypeError, ZeroDivisionError) as e:
                logger.warning("Error calculating %s: %s", factor_name, e)
                confidence_scores[factor_name] = 0.5  # Neutral score on error

        # Calculate weighted overall confidence
        overall_confidence = sum(
            score * self.confidence_weights.get(factor, 0.2)
            for factor, score in confidence_scores.items()
        )

        # Calculate uncertainty level
        uncertainty = self._calculate_uncertainty(confidence_scores)

        # Create confidence score object
        confidence_score = ConfidenceScore(
            overall_confidence=overall_confidence,
            vote_distribution_confidence=confidence_scores.get("vote_distribution", 0.5),
            agent_expertise_confidence=confidence_scores.get("agent_expertise", 0.5),
            historical_accuracy_confidence=confidence_scores.get("historical_accuracy", 0.5),
            information_quality_confidence=confidence_scores.get("information_quality", 0.5),
            consensus_strength_confidence=confidence_scores.get("consensus_strength", 0.5),
            uncertainty_level=uncertainty,
            confidence_factors=confidence_scores
        )

        # Store for historical learning
        self._update_decision_history(consensus_result, confidence_score)

        return confidence_score

    def _extract_votes_from_result(self, consensus_result: ConsensusResult) -> List[Vote]:
        """Extract votes from consensus result for analysis."""
        # This is a simplified extraction - in real implementation,
        # votes would be stored in the consensus result
        votes = []
        
        # Create simulated votes from vote summary
        if "vote_scores" in consensus_result.vote_summary:
            vote_scores = consensus_result.vote_summary["vote_scores"]
            for proposal_id, score in vote_scores.items():
                # Create a representative vote
                vote = Vote(
                    agent_id=f"agent_{len(votes)}",
                    option_id=proposal_id,
                    confidence=min(1.0, score / 10.0),  # Normalize score to confidence
                    expertise_weight=1.0,
                    reasoning=f"Vote for proposal {proposal_id}"
                )
                votes.append(vote)
        
        return votes

    async def _analyze_vote_distribution(
        self,
        _consensus_result: ConsensusResult,
        votes: List[Vote]
    ) -> float:
        """Analyze distribution of votes for confidence assessment."""
        if not votes:
            return 0.0

        # Count votes per option
        vote_counts = Counter(vote.option_id for vote in votes)
        total_votes = len(votes)
        
        if total_votes == 0:
            return 0.0

        # Calculate distribution metrics
        winning_count = vote_counts.most_common(1)[0][1]
        winning_ratio = winning_count / total_votes
        
        # Calculate entropy (higher entropy = more uncertainty)
        proportions = [count / total_votes for count in vote_counts.values()]
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in proportions)
        max_entropy = np.log2(len(vote_counts)) if len(vote_counts) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Calculate confidence based on winning ratio and entropy
        # High winning ratio and low entropy = high confidence
        distribution_confidence = (
            0.6 * winning_ratio +  # Reward clear majority
            0.4 * (1 - normalized_entropy)  # Reward low uncertainty
        )
        
        # Bonus for unanimous decisions
        if winning_ratio == 1.0:
            distribution_confidence = min(1.0, distribution_confidence * 1.1)
            
        return distribution_confidence

    async def _analyze_agent_expertise(
        self,
        _consensus_result: ConsensusResult,
        votes: List[Vote]
    ) -> float:
        """Analyze agent expertise levels for confidence assessment."""
        if not votes:
            return 0.0

        # Calculate expertise-weighted confidence
        total_expertise_weight = 0.0
        weighted_confidence_sum = 0.0
        
        for vote in votes:
            expertise_weight = vote.expertise_weight
            vote_confidence = vote.confidence
            
            weighted_confidence_sum += expertise_weight * vote_confidence
            total_expertise_weight += expertise_weight
            
        if total_expertise_weight == 0:
            return 0.5  # Neutral when no expertise information
            
        expertise_weighted_confidence = weighted_confidence_sum / total_expertise_weight
        
        # Factor in expertise distribution
        expertise_values = [vote.expertise_weight for vote in votes]
        expertise_variance = np.var(expertise_values) if len(expertise_values) > 1 else 0
        
        # Lower variance in expertise = more reliable (all experts agree on difficulty)
        variance_factor = max(0.0, 1.0 - expertise_variance)
        
        return float(0.7 * expertise_weighted_confidence + 0.3 * variance_factor)

    async def _analyze_historical_accuracy(
        self,
        _consensus_result: ConsensusResult,
        votes: List[Vote]
    ) -> float:
        """Analyze historical accuracy of participating agents."""
        if not votes:
            return 0.5  # Neutral when no historical data

        # Calculate average historical accuracy of participating agents
        participating_agents = {vote.agent_id for vote in votes}
        historical_accuracies = []
        
        for agent_id in participating_agents:
            agent_history = self.agent_accuracy_history.get(agent_id, [])
            if agent_history:
                avg_accuracy = statistics.mean(agent_history)
                historical_accuracies.append(avg_accuracy)
            else:
                # New agent - use neutral score
                historical_accuracies.append(0.6)
                
        if not historical_accuracies:
            return 0.5
            
        # Weight by vote confidence
        weighted_accuracies = []
        for vote in votes:
            agent_history = self.agent_accuracy_history.get(vote.agent_id, [0.6])
            agent_accuracy = statistics.mean(agent_history)
            weighted_accuracy = agent_accuracy * vote.confidence
            weighted_accuracies.append(weighted_accuracy)
            
        return statistics.mean(weighted_accuracies)

    async def _analyze_information_quality(
        self,
        _consensus_result: ConsensusResult,
        votes: List[Vote]
    ) -> float:
        """Analyze quality of information used in decision."""
        if not votes:
            return 0.0

        quality_factors = []
        
        # Factor 1: Reasoning quality (length and detail of reasoning)
        reasoning_scores = []
        for vote in votes:
            if vote.reasoning:
                # Simple heuristic: longer reasoning suggests more thought
                reasoning_length = len(vote.reasoning.strip())
                reasoning_score = min(1.0, reasoning_length / 100.0)  # Normalize to 100 chars
            else:
                reasoning_score = 0.3  # Penalty for no reasoning
            reasoning_scores.append(reasoning_score)
            
        avg_reasoning_quality = statistics.mean(reasoning_scores) if reasoning_scores else 0.3
        quality_factors.append(avg_reasoning_quality)
        
        # Factor 2: Vote confidence levels
        confidence_levels = [vote.confidence for vote in votes]
        avg_confidence = statistics.mean(confidence_levels) if confidence_levels else 0.5
        quality_factors.append(avg_confidence)
        
        # Factor 3: Vote strength consistency
        vote_strengths = [vote.vote_strength for vote in votes]
        strength_variance = np.var(vote_strengths) if len(vote_strengths) > 1 else 0
        consistency_score = max(0.0, 1.0 - strength_variance)
        quality_factors.append(consistency_score)
        
        # Factor 4: Participation completeness
        # This could be enhanced to check if all relevant experts participated
        participation_score = min(1.0, len(votes) / 10.0)  # Assume 10 is good participation
        quality_factors.append(participation_score)
        
        return statistics.mean(quality_factors)

    async def _analyze_consensus_strength(
        self,
        consensus_result: ConsensusResult,
        votes: List[Vote]
    ) -> float:
        """Analyze strength of consensus for confidence."""
        if not votes:
            return 0.0

        # Use the consensus strength from the result
        base_strength = consensus_result.consensus_strength
        
        # Factor in participation rate
        total_possible_agents = consensus_result.participation_count
        actual_participation = len(votes)
        participation_ratio = actual_participation / max(1, total_possible_agents)
        
        # Factor in vote confidence variance
        confidence_levels = [vote.confidence for vote in votes]
        if len(confidence_levels) > 1:
            confidence_variance = np.var(confidence_levels)
            confidence_consistency = max(0.0, 1.0 - confidence_variance)
        else:
            confidence_consistency = 1.0
            
        # Combine factors
        strength_score = (
            0.5 * base_strength +
            0.3 * participation_ratio +
            0.2 * confidence_consistency
        )
        
        return float(min(1.0, strength_score))

    def _calculate_uncertainty(self, confidence_factors: Dict[str, float]) -> float:
        """Calculate uncertainty level in the decision."""
        if not confidence_factors:
            return 1.0  # Maximum uncertainty
            
        # Calculate variance in confidence factors
        factor_values = list(confidence_factors.values())
        confidence_variance = np.var(factor_values) if len(factor_values) > 1 else 0
        
        # Calculate average confidence
        avg_confidence = statistics.mean(factor_values)
        
        # Uncertainty increases with variance and decreases with average confidence
        uncertainty = (
            0.6 * (1.0 - avg_confidence) +  # Low confidence = high uncertainty
            0.4 * confidence_variance       # High variance = high uncertainty
        )
        
        return float(min(1.0, max(0.0, uncertainty)))

    def _update_decision_history(
        self,
        consensus_result: ConsensusResult,
        confidence_score: ConfidenceScore
    ) -> None:
        """Update historical records for learning."""
        decision_record = {
            "timestamp": datetime.now(),
            "algorithm": consensus_result.algorithm_used,
            "consensus_strength": consensus_result.consensus_strength,
            "confidence_score": confidence_score.overall_confidence,
            "participation_count": consensus_result.participation_count,
            "winning_proposal_id": consensus_result.winning_proposal.id
        }
        
        self.decision_history.append(decision_record)
        
        # Keep only recent history (last 1000 decisions)
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

    def get_confidence_statistics(self) -> Dict[str, Any]:
        """Get statistics about confidence scoring performance."""
        if not self.decision_history:
            return {"message": "No historical data available"}
            
        recent_decisions = self.decision_history[-100:]  # Last 100 decisions
        
        confidence_scores = [d["confidence_score"] for d in recent_decisions]
        consensus_strengths = [d["consensus_strength"] for d in recent_decisions]
        
        return {
            "total_decisions": len(self.decision_history),
            "recent_decisions": len(recent_decisions),
            "average_confidence": statistics.mean(confidence_scores) if confidence_scores else 0,
            "confidence_std": statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0,
            "average_consensus_strength": statistics.mean(consensus_strengths) if consensus_strengths else 0,
            "algorithms_used": Counter(d["algorithm"] for d in recent_decisions),
            "confidence_trend": self._calculate_confidence_trend(confidence_scores)
        }

    def _calculate_confidence_trend(self, confidence_scores: List[float]) -> str:
        """Calculate trend in confidence scores."""
        if len(confidence_scores) < 10:
            return "insufficient_data"
            
        # Simple trend calculation using first and last halves
        mid_point = len(confidence_scores) // 2
        first_half_avg = statistics.mean(confidence_scores[:mid_point])
        second_half_avg = statistics.mean(confidence_scores[mid_point:])
        
        diff = second_half_avg - first_half_avg
        
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"

    def update_agent_accuracy(self, agent_id: str, accuracy: float) -> None:
        """Update historical accuracy for an agent."""
        self.agent_accuracy_history[agent_id].append(accuracy)
        
        # Keep only recent history per agent
        if len(self.agent_accuracy_history[agent_id]) > 50:
            self.agent_accuracy_history[agent_id] = self.agent_accuracy_history[agent_id][-50:]