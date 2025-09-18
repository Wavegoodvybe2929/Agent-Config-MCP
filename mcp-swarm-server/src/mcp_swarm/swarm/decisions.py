"""
Collective Decision Making for Swarm Intelligence

This module implements collective decision-making protocols for coordinating
agent consensus in the MCP Swarm Intelligence Server. It supports multiple
voting mechanisms, weighted expertise, conflict resolution, and consensus
building strategies.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of collective decision-making mechanisms."""
    SIMPLE_MAJORITY = "simple_majority"
    WEIGHTED_MAJORITY = "weighted_majority"
    CONSENSUS = "consensus"
    EXPERTISE_WEIGHTED = "expertise_weighted"
    PHEROMONE_GUIDED = "pheromone_guided"
    RANKED_CHOICE = "ranked_choice"
    APPROVAL_VOTING = "approval_voting"
    BORDA_COUNT = "borda_count"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving voting conflicts."""
    REVOTE = "revote"
    EXPERT_ARBITRATION = "expert_arbitration"
    RANDOM_SELECTION = "random_selection"
    COMPROMISE = "compromise"
    DEFER_TO_AUTHORITY = "defer_to_authority"
    WEIGHTED_COMPROMISE = "weighted_compromise"


@dataclass
class Vote:
    """Represents a single vote from an agent."""
    agent_id: str
    option_id: str
    confidence: float  # 0.0 to 1.0
    expertise_weight: float = 1.0
    reasoning: Optional[str] = None
    vote_strength: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RankedVote:
    """Represents a ranked vote with preferences."""
    agent_id: str
    rankings: List[str]  # Ordered list of option IDs (most preferred first)
    confidence: float
    expertise_weight: float = 1.0
    reasoning: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ApprovalVote:
    """Represents an approval vote for multiple options."""
    agent_id: str
    approved_options: Set[str]
    confidence: float
    expertise_weight: float = 1.0
    reasoning: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DecisionOption:
    """Represents a decision option with metadata."""
    id: str
    title: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    estimated_effort: Optional[float] = None
    risk_level: float = 0.5  # 0.0 (low) to 1.0 (high)
    expected_outcome: Optional[str] = None


@dataclass
class DecisionResult:
    """Results of a collective decision-making process."""
    winning_option: DecisionOption
    decision_type: DecisionType
    vote_counts: Dict[str, int]
    weighted_scores: Dict[str, float]
    confidence_score: float
    unanimous: bool
    participation_rate: float
    dissenting_agents: List[str]
    conflict_resolution_used: Optional[ConflictResolutionStrategy]
    decision_time: datetime
    total_votes: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionContext:
    """Context for decision-making process."""
    decision_id: str
    title: str
    description: str
    deadline: Optional[datetime] = None
    required_consensus_threshold: float = 0.5
    minimum_participation: int = 1
    allow_abstentions: bool = True
    enable_discussion: bool = True
    max_voting_rounds: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


class VotingMechanism(ABC):
    """Abstract base class for voting mechanisms."""
    
    @abstractmethod
    async def process_votes(
        self,
        votes: List[Union[Vote, RankedVote, ApprovalVote]],
        options: List[DecisionOption],
        context: DecisionContext
    ) -> Tuple[str, Dict[str, Any]]:
        """Process votes and return winning option ID and metadata."""
        raise NotImplementedError("Subclasses must implement process_votes")


class SimpleMajorityVoting(VotingMechanism):
    """Simple majority voting mechanism."""
    
    async def process_votes(
        self,
        votes: List[Union[Vote, RankedVote, ApprovalVote]],
        options: List[DecisionOption],
        context: DecisionContext
    ) -> Tuple[str, Dict[str, Any]]:
        """Process votes using simple majority rule."""
        vote_counts = {}
        total_votes = 0
        
        for vote in votes:
            if isinstance(vote, Vote):
                option_id = vote.option_id
                vote_counts[option_id] = vote_counts.get(option_id, 0) + 1
                total_votes += 1
        
        if not vote_counts:
            raise ValueError("No valid votes received")
        
        winning_option = max(vote_counts.keys(), key=lambda k: vote_counts[k])
        winner_count = vote_counts[winning_option]
        
        metadata = {
            "vote_counts": vote_counts,
            "total_votes": total_votes,
            "winner_percentage": winner_count / total_votes if total_votes > 0 else 0,
            "majority_achieved": winner_count > total_votes / 2
        }
        
        return winning_option, metadata


class WeightedMajorityVoting(VotingMechanism):
    """Weighted majority voting with expertise consideration."""
    
    async def process_votes(
        self,
        votes: List[Union[Vote, RankedVote, ApprovalVote]],
        options: List[DecisionOption],
        context: DecisionContext
    ) -> Tuple[str, Dict[str, Any]]:
        """Process votes using weighted majority rule."""
        weighted_scores = {}
        total_weight = 0
        
        for vote in votes:
            if isinstance(vote, Vote):
                option_id = vote.option_id
                weight = vote.expertise_weight * vote.confidence * vote.vote_strength
                weighted_scores[option_id] = weighted_scores.get(option_id, 0) + weight
                total_weight += weight
        
        if not weighted_scores:
            raise ValueError("No valid votes received")
        
        winning_option = max(weighted_scores.keys(), key=lambda k: weighted_scores[k])
        winner_score = weighted_scores[winning_option]
        
        metadata = {
            "weighted_scores": weighted_scores,
            "total_weight": total_weight,
            "winner_percentage": winner_score / total_weight if total_weight > 0 else 0,
            "normalized_scores": {
                opt: score / total_weight for opt, score in weighted_scores.items()
            } if total_weight > 0 else {}
        }
        
        return winning_option, metadata


class RankedChoiceVoting(VotingMechanism):
    """Ranked choice voting with instant runoff."""
    
    async def process_votes(
        self,
        votes: List[Union[Vote, RankedVote, ApprovalVote]],
        options: List[DecisionOption],
        context: DecisionContext
    ) -> Tuple[str, Dict[str, Any]]:
        """Process votes using ranked choice with instant runoff."""
        ranked_votes = [v for v in votes if isinstance(v, RankedVote)]
        
        if not ranked_votes:
            raise ValueError("No ranked votes received")
        
        # Initialize with all options
        remaining_options = {opt.id for opt in options}
        elimination_rounds = []
        
        while len(remaining_options) > 1:
            # Count first-choice votes for remaining options
            first_choice_counts = {}
            total_votes = 0
            
            for vote in ranked_votes:
                # Find first remaining choice in rankings
                first_choice = None
                for option_id in vote.rankings:
                    if option_id in remaining_options:
                        first_choice = option_id
                        break
                
                if first_choice:
                    weight = vote.expertise_weight * vote.confidence
                    first_choice_counts[first_choice] = first_choice_counts.get(first_choice, 0) + weight
                    total_votes += weight
            
            if not first_choice_counts:
                break
            
            # Check if any option has majority
            for option_id, count in first_choice_counts.items():
                if count > total_votes / 2:
                    return option_id, {
                        "elimination_rounds": elimination_rounds,
                        "final_counts": first_choice_counts,
                        "majority_winner": True
                    }
            
            # Eliminate option with fewest votes
            min_option = min(first_choice_counts.keys(), key=lambda k: first_choice_counts[k])
            remaining_options.remove(min_option)
            elimination_rounds.append({
                "eliminated": min_option,
                "counts": first_choice_counts.copy(),
                "remaining": list(remaining_options)
            })
        
        # Return last remaining option
        winning_option = list(remaining_options)[0] if remaining_options else None
        
        if not winning_option:
            raise ValueError("No winner determined in ranked choice voting")
        
        metadata = {
            "elimination_rounds": elimination_rounds,
            "majority_winner": False,
            "final_winner": winning_option
        }
        
        return winning_option, metadata


class BordaCountVoting(VotingMechanism):
    """Borda count voting mechanism."""
    
    async def process_votes(
        self,
        votes: List[Union[Vote, RankedVote, ApprovalVote]],
        options: List[DecisionOption],
        context: DecisionContext
    ) -> Tuple[str, Dict[str, Any]]:
        """Process votes using Borda count."""
        ranked_votes = [v for v in votes if isinstance(v, RankedVote)]
        
        if not ranked_votes:
            raise ValueError("No ranked votes received")
        
        num_options = len(options)
        borda_scores = {}
        
        for vote in ranked_votes:
            weight = vote.expertise_weight * vote.confidence
            
            for position, option_id in enumerate(vote.rankings):
                if option_id in {opt.id for opt in options}:
                    # Higher position = lower index = more points
                    points = (num_options - position - 1) * weight
                    borda_scores[option_id] = borda_scores.get(option_id, 0) + points
        
        if not borda_scores:
            raise ValueError("No valid Borda scores calculated")
        
        winning_option = max(borda_scores.keys(), key=lambda k: borda_scores[k])
        
        metadata = {
            "borda_scores": borda_scores,
            "normalized_scores": {
                opt: score / max(borda_scores.values()) for opt, score in borda_scores.items()
            } if borda_scores else {}
        }
        
        return winning_option, metadata


class ApprovalVoting(VotingMechanism):
    """Approval voting mechanism."""
    
    async def process_votes(
        self,
        votes: List[Union[Vote, RankedVote, ApprovalVote]],
        options: List[DecisionOption],
        context: DecisionContext
    ) -> Tuple[str, Dict[str, Any]]:
        """Process votes using approval voting."""
        approval_votes = [v for v in votes if isinstance(v, ApprovalVote)]
        
        if not approval_votes:
            raise ValueError("No approval votes received")
        
        approval_counts = {}
        
        for vote in approval_votes:
            weight = vote.expertise_weight * vote.confidence
            
            for option_id in vote.approved_options:
                if option_id in {opt.id for opt in options}:
                    approval_counts[option_id] = approval_counts.get(option_id, 0) + weight
        
        if not approval_counts:
            raise ValueError("No valid approval counts")
        
        winning_option = max(approval_counts.keys(), key=lambda k: approval_counts[k])
        
        metadata = {
            "approval_counts": approval_counts,
            "total_approvals": sum(approval_counts.values())
        }
        
        return winning_option, metadata


class CollectiveDecisionMaker:
    """
    Coordinate collective decision making among agents.
    
    This class provides comprehensive decision-making capabilities including:
    - Multiple voting mechanisms (majority, weighted, ranked choice, etc.)
    - Conflict resolution strategies
    - Expertise weighting and confidence scoring
    - Multi-round voting support
    - Decision audit trails and explanations
    """
    
    def __init__(
        self,
        default_decision_type: DecisionType = DecisionType.WEIGHTED_MAJORITY,
        default_conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.REVOTE,
        expertise_weight_factor: float = 1.0,
        confidence_weight_factor: float = 1.0,
        minimum_consensus_threshold: float = 0.6,
        decision_timeout: float = 300.0  # 5 minutes
    ):
        """
        Initialize collective decision maker.
        
        Args:
            default_decision_type: Default voting mechanism
            default_conflict_resolution: Default conflict resolution strategy
            expertise_weight_factor: Factor for expertise weighting
            confidence_weight_factor: Factor for confidence weighting
            minimum_consensus_threshold: Minimum threshold for consensus
            decision_timeout: Maximum time for decision process (seconds)
        """
        self.default_decision_type = default_decision_type
        self.default_conflict_resolution = default_conflict_resolution
        self.expertise_weight_factor = expertise_weight_factor
        self.confidence_weight_factor = confidence_weight_factor
        self.minimum_consensus_threshold = minimum_consensus_threshold
        self.decision_timeout = decision_timeout
        
        # Voting mechanisms
        self.voting_mechanisms: Dict[DecisionType, VotingMechanism] = {
            DecisionType.SIMPLE_MAJORITY: SimpleMajorityVoting(),
            DecisionType.WEIGHTED_MAJORITY: WeightedMajorityVoting(),
            DecisionType.RANKED_CHOICE: RankedChoiceVoting(),
            DecisionType.BORDA_COUNT: BordaCountVoting(),
            DecisionType.APPROVAL_VOTING: ApprovalVoting(),
        }
        
        # Decision history
        self.decision_history: List[DecisionResult] = []
        self.active_decisions: Dict[str, DecisionContext] = {}
    
    async def make_decision(
        self,
        context: DecisionContext,
        options: List[DecisionOption],
        votes: List[Union[Vote, RankedVote, ApprovalVote]],
        decision_type: Optional[DecisionType] = None,
        conflict_resolution: Optional[ConflictResolutionStrategy] = None
    ) -> DecisionResult:
        """
        Make a collective decision based on votes and options.
        
        Args:
            context: Decision context and parameters
            options: Available decision options
            votes: Votes from participating agents
            decision_type: Voting mechanism to use
            conflict_resolution: Conflict resolution strategy
            
        Returns:
            DecisionResult with the outcome and metadata
        """
        if not options:
            raise ValueError("At least one decision option is required")
        
        if not votes:
            raise ValueError("At least one vote is required")
        
        decision_type = decision_type or self.default_decision_type
        conflict_resolution = conflict_resolution or self.default_conflict_resolution
        
        logger.info("Starting collective decision: %s with %d options, %d votes", 
                   context.title, len(options), len(votes))
        
        # Validate votes
        validated_votes = self._validate_votes(votes, options)
        
        # Check participation requirements
        if len(validated_votes) < context.minimum_participation:
            raise ValueError(f"Insufficient participation: {len(validated_votes)} < {context.minimum_participation}")
        
        # Process votes through multiple rounds if needed
        result = await self._process_decision_rounds(
            context, options, validated_votes, decision_type, conflict_resolution
        )
        
        # Store in history
        self.decision_history.append(result)
        
        # Clean up active decision
        if context.decision_id in self.active_decisions:
            del self.active_decisions[context.decision_id]
        
        logger.info("Decision completed: %s -> %s (confidence: %.3f)",
                   context.title, result.winning_option.title, result.confidence_score)
        
        return result
    
    def _validate_votes(
        self,
        votes: List[Union[Vote, RankedVote, ApprovalVote]],
        options: List[DecisionOption]
    ) -> List[Union[Vote, RankedVote, ApprovalVote]]:
        """Validate and filter votes."""
        valid_option_ids = {opt.id for opt in options}
        validated_votes = []
        
        for vote in votes:
            if isinstance(vote, Vote):
                if vote.option_id in valid_option_ids and vote.confidence > 0:
                    validated_votes.append(vote)
            elif isinstance(vote, RankedVote):
                # Filter rankings to only include valid options
                valid_rankings = [opt_id for opt_id in vote.rankings if opt_id in valid_option_ids]
                if valid_rankings and vote.confidence > 0:
                    filtered_vote = RankedVote(
                        agent_id=vote.agent_id,
                        rankings=valid_rankings,
                        confidence=vote.confidence,
                        expertise_weight=vote.expertise_weight,
                        reasoning=vote.reasoning,
                        timestamp=vote.timestamp
                    )
                    validated_votes.append(filtered_vote)
            elif isinstance(vote, ApprovalVote):
                valid_approvals = vote.approved_options & valid_option_ids
                if valid_approvals and vote.confidence > 0:
                    filtered_vote = ApprovalVote(
                        agent_id=vote.agent_id,
                        approved_options=valid_approvals,
                        confidence=vote.confidence,
                        expertise_weight=vote.expertise_weight,
                        reasoning=vote.reasoning,
                        timestamp=vote.timestamp
                    )
                    validated_votes.append(filtered_vote)
        
        return validated_votes
    
    async def _process_decision_rounds(
        self,
        context: DecisionContext,
        options: List[DecisionOption],
        votes: List[Union[Vote, RankedVote, ApprovalVote]],
        decision_type: DecisionType,
        conflict_resolution: ConflictResolutionStrategy
    ) -> DecisionResult:
        """Process decision through multiple rounds if needed."""
        voting_round = 1
        current_votes = votes
        conflict_resolution_used = None
        
        while voting_round <= context.max_voting_rounds:
            try:
                # Get voting mechanism
                mechanism = self.voting_mechanisms.get(decision_type)
                if not mechanism:
                    raise ValueError(f"Unsupported decision type: {decision_type}")
                
                # Process votes
                winning_option_id, vote_metadata = await mechanism.process_votes(
                    current_votes, options, context
                )
                
                # Find winning option
                winning_option = next((opt for opt in options if opt.id == winning_option_id), None)
                if not winning_option:
                    raise ValueError(f"Winning option {winning_option_id} not found")
                
                # Calculate metrics
                result_metrics = self._calculate_decision_metrics(
                    current_votes, winning_option
                )
                
                # Check if consensus is achieved
                if result_metrics["confidence_score"] >= context.required_consensus_threshold:
                    return DecisionResult(
                        winning_option=winning_option,
                        decision_type=decision_type,
                        vote_counts=vote_metadata.get("vote_counts", {}),
                        weighted_scores=vote_metadata.get("weighted_scores", {}),
                        confidence_score=result_metrics["confidence_score"],
                        unanimous=result_metrics["unanimous"],
                        participation_rate=result_metrics["participation_rate"],
                        dissenting_agents=result_metrics["dissenting_agents"],
                        conflict_resolution_used=conflict_resolution_used,
                        decision_time=datetime.now(),
                        total_votes=len(current_votes),
                        metadata={
                            "voting_rounds": voting_round,
                            "vote_metadata": vote_metadata,
                            **result_metrics
                        }
                    )
                
                # Handle conflict if not final round
                if voting_round < context.max_voting_rounds:
                    current_votes = await self._resolve_conflict(
                        current_votes, conflict_resolution
                    )
                    conflict_resolution_used = conflict_resolution
                    voting_round += 1
                else:
                    # Final round - return best result even if below threshold
                    return DecisionResult(
                        winning_option=winning_option,
                        decision_type=decision_type,
                        vote_counts=vote_metadata.get("vote_counts", {}),
                        weighted_scores=vote_metadata.get("weighted_scores", {}),
                        confidence_score=result_metrics["confidence_score"],
                        unanimous=result_metrics["unanimous"],
                        participation_rate=result_metrics["participation_rate"],
                        dissenting_agents=result_metrics["dissenting_agents"],
                        conflict_resolution_used=conflict_resolution_used,
                        decision_time=datetime.now(),
                        total_votes=len(current_votes),
                        metadata={
                            "voting_rounds": voting_round,
                            "vote_metadata": vote_metadata,
                            "consensus_threshold_met": False,
                            **result_metrics
                        }
                    )
                
            except (ValueError, KeyError, AttributeError) as e:
                logger.error("Error in voting round %d: %s", voting_round, e)
                if voting_round == context.max_voting_rounds:
                    raise
                voting_round += 1
        
        raise RuntimeError("Decision process failed after maximum rounds")
    
    def _calculate_decision_metrics(
        self,
        votes: List[Union[Vote, RankedVote, ApprovalVote]],
        winning_option: DecisionOption,
    ) -> Dict[str, Any]:
        """Calculate decision quality metrics."""
        total_agents = len(set(
            vote.agent_id for vote in votes
        ))
        
        # Count votes for winning option
        winning_votes = 0
        total_votes = len(votes)
        
        supporting_agents = set()
        all_agents = set()
        
        for vote in votes:
            all_agents.add(vote.agent_id)
            
            if isinstance(vote, Vote) and vote.option_id == winning_option.id:
                winning_votes += 1
                supporting_agents.add(vote.agent_id)
            elif isinstance(vote, RankedVote) and vote.rankings and vote.rankings[0] == winning_option.id:
                winning_votes += 1
                supporting_agents.add(vote.agent_id)
            elif isinstance(vote, ApprovalVote) and winning_option.id in vote.approved_options:
                supporting_agents.add(vote.agent_id)
        
        dissenting_agents = list(all_agents - supporting_agents)
        unanimous = len(dissenting_agents) == 0
        
        # Calculate confidence score based on support and consensus
        support_ratio = len(supporting_agents) / len(all_agents) if all_agents else 0
        
        # Factor in vote strength and confidence
        weighted_support = 0
        total_weight = 0
        
        for vote in votes:
            weight = getattr(vote, 'expertise_weight', 1.0) * getattr(vote, 'confidence', 1.0)
            total_weight += weight
            
            if isinstance(vote, Vote) and vote.option_id == winning_option.id:
                weighted_support += weight
            elif isinstance(vote, RankedVote) and vote.rankings and vote.rankings[0] == winning_option.id:
                weighted_support += weight
            elif isinstance(vote, ApprovalVote) and winning_option.id in vote.approved_options:
                weighted_support += weight * 0.8  # Slightly lower weight for approval
        
        weighted_ratio = weighted_support / total_weight if total_weight > 0 else 0
        
        # Combined confidence score
        confidence_score = (support_ratio + weighted_ratio) / 2
        
        participation_rate = len(all_agents) / total_agents if total_agents > 0 else 1.0
        
        return {
            "confidence_score": confidence_score,
            "unanimous": unanimous,
            "participation_rate": participation_rate,
            "dissenting_agents": dissenting_agents,
            "support_ratio": support_ratio,
            "weighted_ratio": weighted_ratio,
            "winning_votes": winning_votes,
            "total_votes": total_votes
        }
    
    async def _resolve_conflict(
        self,
        votes: List[Union[Vote, RankedVote, ApprovalVote]],
        strategy: ConflictResolutionStrategy
    ) -> List[Union[Vote, RankedVote, ApprovalVote]]:
        """Resolve voting conflicts using specified strategy."""
        
        if strategy == ConflictResolutionStrategy.REVOTE:
            # In a real implementation, this would trigger a new voting round
            # For now, return the same votes
            logger.info("Conflict resolution: REVOTE - triggering new voting round")
            return votes
        
        elif strategy == ConflictResolutionStrategy.EXPERT_ARBITRATION:
            # Weight votes more heavily by expertise
            logger.info("Conflict resolution: EXPERT_ARBITRATION - increasing expertise weights")
            modified_votes = []
            
            for vote in votes:
                if hasattr(vote, 'expertise_weight'):
                    # Double the expertise weight for high-expertise voters
                    if vote.expertise_weight > 0.7:
                        if isinstance(vote, Vote):
                            modified_vote = Vote(
                                agent_id=vote.agent_id,
                                option_id=vote.option_id,
                                confidence=vote.confidence,
                                expertise_weight=vote.expertise_weight * 2.0,
                                reasoning=vote.reasoning,
                                vote_strength=vote.vote_strength,
                                timestamp=vote.timestamp,
                                metadata=vote.metadata
                            )
                        else:
                            modified_vote = vote  # Keep other vote types unchanged for now
                        modified_votes.append(modified_vote)
                    else:
                        modified_votes.append(vote)
                else:
                    modified_votes.append(vote)
            
            return modified_votes
        
        elif strategy == ConflictResolutionStrategy.WEIGHTED_COMPROMISE:
            # Adjust vote strengths based on confidence and expertise
            logger.info("Conflict resolution: WEIGHTED_COMPROMISE - adjusting vote strengths")
            return votes  # For now, return unchanged
        
        else:
            # For other strategies, return votes unchanged
            logger.info("Conflict resolution: %s - no vote modification", strategy.value)
            return votes
    
    def get_decision_history(
        self,
        limit: Optional[int] = None,
        decision_type: Optional[DecisionType] = None
    ) -> List[DecisionResult]:
        """Get decision history with optional filtering."""
        filtered_history = self.decision_history
        
        if decision_type:
            filtered_history = [
                result for result in filtered_history
                if result.decision_type == decision_type
            ]
        
        if limit:
            filtered_history = filtered_history[-limit:]
        
        return filtered_history
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about decision-making performance."""
        if not self.decision_history:
            return {"status": "no_decisions"}
        
        total_decisions = len(self.decision_history)
        unanimous_decisions = sum(1 for result in self.decision_history if result.unanimous)
        
        confidence_scores = [result.confidence_score for result in self.decision_history]
        participation_rates = [result.participation_rate for result in self.decision_history]
        
        decision_types = {}
        for result in self.decision_history:
            decision_types[result.decision_type.value] = decision_types.get(result.decision_type.value, 0) + 1
        
        conflict_resolutions = {}
        for result in self.decision_history:
            if result.conflict_resolution_used:
                strategy = result.conflict_resolution_used.value
                conflict_resolutions[strategy] = conflict_resolutions.get(strategy, 0) + 1
        
        return {
            "total_decisions": total_decisions,
            "unanimous_decisions": unanimous_decisions,
            "unanimity_rate": unanimous_decisions / total_decisions,
            "average_confidence": float(np.mean(confidence_scores)),
            "confidence_std": float(np.std(confidence_scores)),
            "average_participation": float(np.mean(participation_rates)),
            "decision_type_distribution": decision_types,
            "conflict_resolution_usage": conflict_resolutions,
            "min_confidence": float(np.min(confidence_scores)),
            "max_confidence": float(np.max(confidence_scores))
        }
    
    def clear_history(self) -> None:
        """Clear decision history."""
        self.decision_history.clear()
        logger.info("Decision history cleared")