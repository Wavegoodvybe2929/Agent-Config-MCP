"""
Consensus Algorithms for Swarm Intelligence

This module implements various consensus algorithms for collective decision-making
in the MCP Swarm Intelligence Server, including weighted voting, Byzantine fault
tolerance, RAFT consensus, and swarm-based consensus protocols.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime
import random
from collections import defaultdict, Counter

from ..swarm.decisions import Vote
from ..swarm import SwarmCoordinator
from ..swarm.aco import Agent

logger = logging.getLogger(__name__)


class ConsensusAlgorithm(Enum):
    """Available consensus algorithms."""
    WEIGHTED_VOTING = "weighted_voting"
    BYZANTINE_FAULT_TOLERANT = "byzantine_ft"
    RAFT_CONSENSUS = "raft"
    PRACTICAL_BFT = "pbft"
    SWARM_CONSENSUS = "swarm_consensus"


@dataclass
class Proposal:
    """Represents a proposal for consensus."""
    id: int
    content: str
    topic: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    proposer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert proposal to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "topic": self.topic,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat(),
            "proposer": self.proposer
        }


@dataclass
class ConsensusResult:
    """Result of a consensus process."""
    winning_proposal: Proposal
    vote_summary: Dict[str, Any]
    consensus_strength: float
    participation_count: int
    algorithm_used: str
    completed_at: datetime = field(default_factory=datetime.now)
    confidence_score: Optional[float] = None
    minority_opinions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "winning_proposal": self.winning_proposal.to_dict(),
            "vote_summary": self.vote_summary,
            "consensus_strength": self.consensus_strength,
            "participation_count": self.participation_count,
            "algorithm_used": self.algorithm_used,
            "completed_at": self.completed_at.isoformat(),
            "confidence_score": self.confidence_score,
            "minority_opinions": self.minority_opinions
        }


class ConsensusTimeoutError(Exception):
    """Exception raised when consensus timeout is reached."""


class BaseConsensusAlgorithm(ABC):
    """Base class for all consensus algorithms."""

    @abstractmethod
    async def reach_consensus(
        self,
        proposals: List[Proposal],
        agents: List[Agent],
        timeout: float = 30.0
    ) -> ConsensusResult:
        """
        Reach consensus on proposals using this algorithm.
        
        Args:
            proposals: List of proposals to vote on
            agents: List of participating agents
            timeout: Maximum time to reach consensus
            
        Returns:
            ConsensusResult with winning proposal and details
            
        Raises:
            ConsensusTimeoutError: If consensus cannot be reached in time
        """
        raise NotImplementedError("Subclasses must implement reach_consensus")


class WeightedVotingConsensus(BaseConsensusAlgorithm):
    """Weighted voting consensus based on agent expertise and track records."""

    def __init__(self, expertise_weights: Optional[Dict[str, float]] = None):
        """
        Initialize weighted voting consensus.
        
        Args:
            expertise_weights: Optional predefined weights for agents
        """
        self.expertise_weights = expertise_weights or {}
        self.historical_accuracy: Dict[str, List[float]] = defaultdict(list)

    async def reach_consensus(
        self,
        proposals: List[Proposal],
        agents: List[Agent],
        timeout: float = 30.0
    ) -> ConsensusResult:
        """Weighted voting consensus based on agent expertise."""
        # Collect votes from all agents
        votes = []
        for agent in agents:
            # Simulate agent voting (in real implementation, this would call agent)
            vote = await self._get_agent_vote(agent, proposals)
            if vote:
                votes.append(vote)
                
        if not votes:
            raise ConsensusTimeoutError("No votes received from agents")
            
        # Calculate weights for each agent
        weights = {}
        for vote in votes:
            weights[vote.agent_id] = self._calculate_agent_weight(
                vote.agent_id, proposals[0].topic
            )
            
        # Aggregate weighted votes
        vote_scores = self._aggregate_votes(votes, weights)
        
        # Determine winning proposal
        winning_id = max(vote_scores.keys(), key=lambda k: vote_scores[k])
        winning_proposal = next(p for p in proposals if p.id == int(winning_id))
        
        # Calculate consensus strength
        total_weight = sum(weights.values())
        consensus_strength = vote_scores[winning_id] / total_weight if total_weight > 0 else 0
        
        return ConsensusResult(
            winning_proposal=winning_proposal,
            vote_summary={
                "total_votes": len(votes),
                "vote_scores": vote_scores,
                "weights_used": weights
            },
            consensus_strength=consensus_strength,
            participation_count=len(votes),
            algorithm_used="weighted_voting"
        )

    def _calculate_agent_weight(self, agent_id: str, domain: str) -> float:
        """Calculate voting weight for agent in specific domain."""
        # Base weight from predefined expertise
        base_weight = self.expertise_weights.get(agent_id, 1.0)
        
        # Historical accuracy bonus
        historical_scores = self.historical_accuracy.get(agent_id, [])
        if historical_scores:
            accuracy_bonus = np.mean(historical_scores)
        else:
            accuracy_bonus = 0.5  # Neutral starting point
            
        # Domain-specific adjustment (simplified)
        domain_multiplier = 1.0
        if domain in ["technical", "engineering"]:
            domain_multiplier = 1.2
        elif domain in ["strategic", "planning"]:
            domain_multiplier = 1.1
            
        return float(base_weight * (0.7 + 0.3 * accuracy_bonus) * domain_multiplier)

    def _aggregate_votes(
        self,
        votes: List[Vote],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Aggregate weighted votes for each proposal."""
        proposal_scores = defaultdict(float)
        
        for vote in votes:
            weight = weights.get(vote.agent_id, 1.0)
            confidence_adjusted_weight = weight * vote.confidence * vote.vote_strength
            proposal_scores[vote.option_id] += confidence_adjusted_weight
            
        return dict(proposal_scores)

    async def _get_agent_vote(self, agent: Agent, proposals: List[Proposal]) -> Optional[Vote]:
        """Get vote from agent (simulated for now)."""
        # In real implementation, this would query the actual agent
        if proposals:
            # Simulate voting with some randomness
            chosen_proposal = random.choice(proposals)
            confidence = random.uniform(0.6, 1.0)
            return Vote(
                agent_id=agent.id,
                option_id=str(chosen_proposal.id),
                confidence=confidence,
                expertise_weight=self.expertise_weights.get(agent.id, 1.0),
                reasoning=f"Selected proposal {chosen_proposal.id} based on evaluation",
                vote_strength=1.0
            )
        return None


class ByzantineFaultTolerantConsensus(BaseConsensusAlgorithm):
    """Byzantine fault tolerant consensus algorithm."""

    def __init__(self, fault_tolerance_ratio: float = 0.33):
        """
        Initialize Byzantine fault tolerant consensus.
        
        Args:
            fault_tolerance_ratio: Maximum ratio of malicious agents tolerated
        """
        self.fault_tolerance_ratio = fault_tolerance_ratio
        self.suspicious_agents: Set[str] = set()

    async def reach_consensus(
        self,
        proposals: List[Proposal],
        agents: List[Agent],
        timeout: float = 30.0
    ) -> ConsensusResult:
        """Byzantine fault tolerant consensus implementation."""
        max_malicious = int(len(agents) * self.fault_tolerance_ratio)
        required_agreement = len(agents) - max_malicious
        
        # Multiple rounds of voting to detect Byzantine behavior
        round_results = []
        for _ in range(3):  # 3 rounds for BFT
            votes = []
            for agent in agents:
                if agent.id not in self.suspicious_agents:
                    vote = await self._get_agent_vote(agent, proposals)
                    if vote:
                        votes.append(vote)
                        
            round_results.append(votes)
            
        # Analyze voting patterns for consistency
        consistent_agents = self._identify_consistent_agents(round_results)
        
        if len(consistent_agents) < required_agreement:
            raise ConsensusTimeoutError(
                f"Insufficient consistent agents: {len(consistent_agents)} < {required_agreement}"
            )
            
        # Use votes from consistent agents only
        final_votes = [v for v in round_results[-1] if v.agent_id in consistent_agents]
        
        # Simple majority from consistent agents
        vote_counts = Counter(vote.option_id for vote in final_votes)
        winning_id = vote_counts.most_common(1)[0][0]
        winning_proposal = next(p for p in proposals if p.id == int(winning_id))
        
        consensus_strength = vote_counts[winning_id] / len(final_votes)
        
        return ConsensusResult(
            winning_proposal=winning_proposal,
            vote_summary={
                "total_rounds": len(round_results),
                "consistent_agents": len(consistent_agents),
                "final_votes": len(final_votes),
                "vote_distribution": dict(vote_counts),
                "suspicious_agents": list(self.suspicious_agents)
            },
            consensus_strength=consensus_strength,
            participation_count=len(final_votes),
            algorithm_used="byzantine_fault_tolerant"
        )

    def _identify_consistent_agents(self, round_results: List[List[Vote]]) -> Set[str]:
        """Identify agents with consistent voting patterns across rounds."""
        agent_votes = defaultdict(list)
        
        # Collect votes by agent across rounds
        for round_votes in round_results:
            for vote in round_votes:
                agent_votes[vote.agent_id].append(vote.option_id)
                
        # Identify consistent agents (same vote in all rounds or reasonable variation)
        consistent_agents = set()
        for agent_id, votes in agent_votes.items():
            if len(votes) >= 2:  # Participated in at least 2 rounds
                vote_consistency = len(set(votes)) / len(votes)
                if vote_consistency <= 0.5:  # At most 50% variation
                    consistent_agents.add(agent_id)
                else:
                    self.suspicious_agents.add(agent_id)
                    
        return consistent_agents

    async def _get_agent_vote(self, agent: Agent, proposals: List[Proposal]) -> Optional[Vote]:
        """Get vote from agent (simulated for Byzantine testing)."""
        if proposals:
            # Simulate potential Byzantine behavior
            if agent.id in self.suspicious_agents or random.random() < 0.1:
                # Potentially malicious vote
                chosen_proposal = random.choice(proposals)
            else:
                # Honest vote (simplified)
                chosen_proposal = proposals[0]  # Simplified: prefer first proposal
                
            confidence = random.uniform(0.7, 1.0)
            return Vote(
                agent_id=agent.id,
                option_id=str(chosen_proposal.id),
                confidence=confidence,
                reasoning=f"BFT vote for proposal {chosen_proposal.id}"
            )
        return None


class SwarmConsensusAlgorithm(BaseConsensusAlgorithm):
    """Swarm intelligence based consensus with pheromone trails."""

    def __init__(self, swarm_coordinator: SwarmCoordinator):
        """
        Initialize swarm consensus algorithm.
        
        Args:
            swarm_coordinator: SwarmCoordinator instance for pheromone trails
        """
        self.swarm_coordinator = swarm_coordinator

    async def reach_consensus(
        self,
        proposals: List[Proposal],
        agents: List[Agent],
        timeout: float = 30.0
    ) -> ConsensusResult:
        """Swarm intelligence based consensus with pheromone trails."""
        start_time = datetime.now()
        
        # Initialize pheromone trails for proposals
        proposal_pheromones = {str(p.id): 1.0 for p in proposals}
        
        # Multiple iterations of swarm voting
        iteration_votes = []
        max_iterations = 5
        
        for _ in range(max_iterations):
            # Check timeout
            if (datetime.now() - start_time).total_seconds() > timeout:
                raise ConsensusTimeoutError("Swarm consensus timeout")
                
            # Collect votes with pheromone influence
            votes = []
            for agent in agents:
                vote = await self._get_pheromone_influenced_vote(
                    agent, proposals, proposal_pheromones
                )
                if vote:
                    votes.append(vote)
                    
            iteration_votes.append(votes)
            
            # Update pheromone trails based on votes
            proposal_pheromones = self._update_pheromone_trails(
                votes, proposal_pheromones
            )
            
            # Check for convergence
            if self._check_convergence(votes, threshold=0.7):
                break
                
        # Final vote counting
        final_votes = iteration_votes[-1] if iteration_votes else []
        if not final_votes:
            raise ConsensusTimeoutError("No votes received in swarm consensus")
            
        # Weight votes by pheromone strength
        weighted_scores = defaultdict(float)
        for vote in final_votes:
            pheromone_weight = proposal_pheromones.get(vote.option_id, 1.0)
            weighted_scores[vote.option_id] += vote.confidence * pheromone_weight
            
        winning_id = max(weighted_scores.keys(), key=lambda k: weighted_scores[k])
        winning_proposal = next(p for p in proposals if p.id == int(winning_id))
        
        # Calculate consensus strength
        total_weight = sum(weighted_scores.values())
        consensus_strength = weighted_scores[winning_id] / total_weight if total_weight > 0 else 0
        
        return ConsensusResult(
            winning_proposal=winning_proposal,
            vote_summary={
                "iterations": len(iteration_votes),
                "final_pheromones": proposal_pheromones,
                "weighted_scores": dict(weighted_scores),
                "convergence_achieved": len(iteration_votes) < max_iterations
            },
            consensus_strength=consensus_strength,
            participation_count=len(final_votes),
            algorithm_used="swarm_consensus"
        )

    def _apply_pheromone_influence(
        self,
        votes: List[Vote],
        pheromone_trails: Dict[str, float]
    ) -> List[Vote]:
        """Apply pheromone trail influence to voting decisions."""
        influenced_votes = []
        
        for vote in votes:
            pheromone_strength = pheromone_trails.get(vote.option_id, 1.0)
            # Enhance confidence based on pheromone strength
            influenced_confidence = min(1.0, vote.confidence * (1 + 0.3 * pheromone_strength))
            
            influenced_vote = Vote(
                agent_id=vote.agent_id,
                option_id=vote.option_id,
                confidence=influenced_confidence,
                expertise_weight=vote.expertise_weight,
                reasoning=f"{vote.reasoning} (pheromone influenced: {pheromone_strength:.2f})",
                vote_strength=vote.vote_strength
            )
            influenced_votes.append(influenced_vote)
            
        return influenced_votes

    async def _get_pheromone_influenced_vote(
        self,
        agent: Agent,
        proposals: List[Proposal],
        pheromone_trails: Dict[str, float]
    ) -> Optional[Vote]:
        """Get vote from agent influenced by pheromone trails."""
        if not proposals:
            return None
            
        # Calculate pheromone-weighted probabilities
        proposal_weights = []
        for proposal in proposals:
            pheromone_strength = pheromone_trails.get(str(proposal.id), 1.0)
            # Combine random preference with pheromone influence
            base_preference = random.uniform(0.3, 1.0)
            weighted_preference = base_preference * (1 + pheromone_strength)
            proposal_weights.append(weighted_preference)
            
        # Select proposal based on weights
        total_weight = sum(proposal_weights)
        probabilities = [w / total_weight for w in proposal_weights]
        chosen_idx = np.random.choice(len(proposals), p=probabilities)
        chosen_proposal = proposals[chosen_idx]
        
        # Confidence influenced by pheromone strength
        pheromone_strength = pheromone_trails.get(str(chosen_proposal.id), 1.0)
        base_confidence = random.uniform(0.6, 0.9)
        pheromone_bonus = min(0.2, 0.1 * pheromone_strength)
        confidence = min(1.0, base_confidence + pheromone_bonus)
        
        return Vote(
            agent_id=agent.id,
            option_id=str(chosen_proposal.id),
            confidence=confidence,
            reasoning=f"Swarm vote for proposal {chosen_proposal.id} (pheromone: {pheromone_strength:.2f})"
        )

    def _update_pheromone_trails(
        self,
        votes: List[Vote],
        current_pheromones: Dict[str, float]
    ) -> Dict[str, float]:
        """Update pheromone trails based on voting results."""
        # Decay all pheromones
        decay_rate = 0.1
        new_pheromones = {
            proposal_id: max(0.1, strength * (1 - decay_rate))
            for proposal_id, strength in current_pheromones.items()
        }
        
        # Reinforce pheromones for voted proposals
        vote_counts = Counter(vote.option_id for vote in votes)
        total_votes = len(votes)
        
        for proposal_id, count in vote_counts.items():
            reinforcement = (count / total_votes) * 2.0  # Scaling factor
            new_pheromones[proposal_id] = min(10.0, new_pheromones.get(proposal_id, 1.0) + reinforcement)
            
        return new_pheromones

    def _check_convergence(self, votes: List[Vote], threshold: float = 0.7) -> bool:
        """Check if votes have converged to a consensus."""
        if not votes:
            return False
            
        vote_counts = Counter(vote.option_id for vote in votes)
        most_common_count = vote_counts.most_common(1)[0][1]
        convergence_ratio = most_common_count / len(votes)
        
        return convergence_ratio >= threshold


# Factory function for creating consensus algorithms
def create_consensus_algorithm(
    algorithm_type: ConsensusAlgorithm,
    swarm_coordinator: Optional[SwarmCoordinator] = None,
    **kwargs
) -> BaseConsensusAlgorithm:
    """
    Factory function to create consensus algorithms.
    
    Args:
        algorithm_type: Type of consensus algorithm to create
        swarm_coordinator: SwarmCoordinator instance for swarm-based algorithms
        **kwargs: Additional algorithm-specific parameters
        
    Returns:
        Consensus algorithm instance
    """
    if algorithm_type == ConsensusAlgorithm.WEIGHTED_VOTING:
        return WeightedVotingConsensus(
            expertise_weights=kwargs.get('expertise_weights')
        )
    elif algorithm_type == ConsensusAlgorithm.BYZANTINE_FAULT_TOLERANT:
        return ByzantineFaultTolerantConsensus(
            fault_tolerance_ratio=kwargs.get('fault_tolerance_ratio', 0.33)
        )
    elif algorithm_type == ConsensusAlgorithm.SWARM_CONSENSUS:
        if not swarm_coordinator:
            raise ValueError("SwarmCoordinator required for swarm consensus")
        return SwarmConsensusAlgorithm(swarm_coordinator)
    else:
        raise ValueError(f"Unsupported consensus algorithm: {algorithm_type}")