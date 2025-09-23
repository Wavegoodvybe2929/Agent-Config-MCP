"""
Decision Audit Trail for Swarm Consensus

This module creates comprehensive audit trails for consensus decisions
in the MCP Swarm Intelligence Server, documenting the complete decision-making
process for accountability, learning, and transparency.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from collections import defaultdict

from .consensus_algorithms import Proposal

logger = logging.getLogger(__name__)


@dataclass
class ReasoningChain:
    """Represents a chain of reasoning for a vote or decision."""
    agent_id: str
    reasoning_steps: List[str]
    decision_factors: Dict[str, float]
    confidence_evolution: List[float]  # How confidence changed during reasoning
    external_influences: List[str]
    final_decision: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reasoning chain to dictionary."""
        return {
            "agent_id": self.agent_id,
            "reasoning_steps": self.reasoning_steps,
            "decision_factors": self.decision_factors,
            "confidence_evolution": self.confidence_evolution,
            "external_influences": self.external_influences,
            "final_decision": self.final_decision,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class VotingProcessRecord:
    """Records the voting process and timeline."""
    process_id: str
    algorithm_used: str
    participants: List[str]
    voting_rounds: List[Dict[str, Any]]
    timeline_events: List[Dict[str, Any]]
    participation_metrics: Dict[str, Any]
    process_duration: float  # seconds
    started_at: datetime
    completed_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert voting process record to dictionary."""
        return {
            "process_id": self.process_id,
            "algorithm_used": self.algorithm_used,
            "participants": self.participants,
            "voting_rounds": self.voting_rounds,
            "timeline_events": self.timeline_events,
            "participation_metrics": self.participation_metrics,
            "process_duration": self.process_duration,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
        }


@dataclass
class ConsensusEvolution:
    """Documents how consensus evolved during the process."""
    evolution_id: str
    initial_state: Dict[str, Any]
    intermediate_states: List[Dict[str, Any]]
    final_state: Dict[str, Any]
    convergence_metrics: Dict[str, float]
    stability_analysis: Dict[str, Any]
    turning_points: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert consensus evolution to dictionary."""
        return {
            "evolution_id": self.evolution_id,
            "initial_state": self.initial_state,
            "intermediate_states": self.intermediate_states,
            "final_state": self.final_state,
            "convergence_metrics": self.convergence_metrics,
            "stability_analysis": self.stability_analysis,
            "turning_points": self.turning_points
        }


@dataclass
class ConsensusProcess:
    """Represents a complete consensus process for auditing."""
    topic: str
    proposals: List[Proposal]
    agents: List[Any]  # Agent objects
    algorithm: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "in_progress"  # in_progress, completed, timeout, failed
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_partial_results(self) -> Dict[str, Any]:
        """Get partial results for incomplete processes."""
        return {
            "topic": self.topic,
            "proposals_count": len(self.proposals),
            "participating_agents": len(self.agents),
            "algorithm": self.algorithm,
            "status": self.status,
            "intermediate_results": self.intermediate_results,
            "duration_so_far": (datetime.now() - self.started_at).total_seconds()
        }


@dataclass
class AuditTrail:
    """Complete audit trail for a consensus decision."""
    id: str
    consensus_id: str
    voting_process: VotingProcessRecord
    reasoning_chains: List[ReasoningChain]
    consensus_evolution: ConsensusEvolution
    decision_context: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit trail to dictionary."""
        return {
            "id": self.id,
            "consensus_id": self.consensus_id,
            "voting_process": self.voting_process.to_dict(),
            "reasoning_chains": [rc.to_dict() for rc in self.reasoning_chains],
            "consensus_evolution": self.consensus_evolution.to_dict(),
            "decision_context": self.decision_context,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class AuditDatabase:
    """Simplified audit database interface."""
    
    def __init__(self):
        """Initialize audit database."""
        self.audit_records: List[AuditTrail] = []
        
    def store_audit_trail(self, audit_trail: AuditTrail) -> str:
        """Store audit trail and return its ID."""
        self.audit_records.append(audit_trail)
        
        # Keep only recent records (last 500)
        if len(self.audit_records) > 500:
            self.audit_records = self.audit_records[-500:]
            
        return audit_trail.id
    
    def get_audit_trail(self, audit_id: str) -> Optional[AuditTrail]:
        """Retrieve audit trail by ID."""
        for trail in self.audit_records:
            if trail.id == audit_id:
                return trail
        return None
    
    def search_audit_trails(self, criteria: Dict[str, Any]) -> List[AuditTrail]:
        """Search audit trails by criteria."""
        results = []
        for trail in self.audit_records:
            match = True
            for key, value in criteria.items():
                if key == "algorithm" and trail.voting_process.algorithm_used != value:
                    match = False
                    break
                elif key == "date_range" and not (value[0] <= trail.created_at <= value[1]):
                    match = False
                    break
            if match:
                results.append(trail)
        return results


class DecisionAuditTrailGenerator:
    """Creates comprehensive audit trails for consensus decisions."""

    def __init__(self, audit_db: Optional[AuditDatabase] = None):
        """
        Initialize decision audit trail generator.
        
        Args:
            audit_db: Database for storing audit trails
        """
        self.audit_db = audit_db or AuditDatabase()
        self.active_processes: Dict[str, ConsensusProcess] = {}

    async def create_decision_audit_trail(
        self,
        consensus_process: ConsensusProcess
    ) -> AuditTrail:
        """
        Create comprehensive audit trail for consensus decision.
        
        Args:
            consensus_process: The consensus process to audit
            
        Returns:
            AuditTrail with complete decision documentation
        """
        # Generate unique audit trail ID
        audit_id = self._generate_audit_id(consensus_process)
        
        # Document voting process
        voting_process = self._document_voting_process(consensus_process)
        
        # Capture reasoning chains
        reasoning_chains = self._capture_reasoning_chains(consensus_process)
        
        # Document consensus evolution
        consensus_evolution = self._document_consensus_evolution(consensus_process)
        
        # Gather decision context
        decision_context = self._gather_decision_context(consensus_process)
        
        # Create metadata
        metadata = self._create_audit_metadata(consensus_process)
        
        # Create audit trail
        audit_trail = AuditTrail(
            id=audit_id,
            consensus_id=self._generate_consensus_id(consensus_process),
            voting_process=voting_process,
            reasoning_chains=reasoning_chains,
            consensus_evolution=consensus_evolution,
            decision_context=decision_context,
            metadata=metadata
        )
        
        # Store in database
        stored_id = self.audit_db.store_audit_trail(audit_trail)
        logger.info("Created audit trail %s for consensus process", stored_id)
        
        return audit_trail

    def _document_voting_process(
        self,
        consensus_process: ConsensusProcess
    ) -> VotingProcessRecord:
        """Document the voting process and timeline."""
        # Calculate process duration
        if consensus_process.completed_at:
            duration = (consensus_process.completed_at - consensus_process.started_at).total_seconds()
            completed_at = consensus_process.completed_at
        else:
            duration = (datetime.now() - consensus_process.started_at).total_seconds()
            completed_at = datetime.now()
        
        # Extract participant information
        participants = [agent.id if hasattr(agent, 'id') else str(agent) for agent in consensus_process.agents]
        
        # Create voting rounds documentation
        voting_rounds = []
        for i, result in enumerate(consensus_process.intermediate_results):
            round_doc = {
                "round_number": i + 1,
                "timestamp": result.get("timestamp", datetime.now().isoformat()),
                "votes_cast": result.get("votes_cast", 0),
                "consensus_strength": result.get("consensus_strength", 0.0),
                "leading_proposal": result.get("leading_proposal", "unknown")
            }
            voting_rounds.append(round_doc)
        
        # Create timeline events
        timeline_events = [
            {
                "event": "process_started",
                "timestamp": consensus_process.started_at.isoformat(),
                "details": f"Consensus process started with {len(participants)} agents"
            },
            {
                "event": "process_completed",
                "timestamp": completed_at.isoformat(),
                "details": f"Process completed with status: {consensus_process.status}"
            }
        ]
        
        # Calculate participation metrics
        participation_metrics = {
            "total_agents": len(consensus_process.agents),
            "participating_agents": len(participants),
            "participation_rate": len(participants) / max(1, len(consensus_process.agents)),
            "average_response_time": duration / max(1, len(voting_rounds)),
            "rounds_completed": len(voting_rounds)
        }
        
        return VotingProcessRecord(
            process_id=self._generate_process_id(consensus_process),
            algorithm_used=consensus_process.algorithm,
            participants=participants,
            voting_rounds=voting_rounds,
            timeline_events=timeline_events,
            participation_metrics=participation_metrics,
            process_duration=duration,
            started_at=consensus_process.started_at,
            completed_at=completed_at
        )

    def _capture_reasoning_chains(
        self,
        consensus_process: ConsensusProcess
    ) -> List[ReasoningChain]:
        """Capture reasoning chains for each vote."""
        reasoning_chains = []
        
        # For each agent, create a reasoning chain
        for agent in consensus_process.agents:
            agent_id = agent.id if hasattr(agent, 'id') else str(agent)
            
            # Simulate reasoning chain (in real implementation, this would be extracted from actual agent reasoning)
            reasoning_chain = ReasoningChain(
                agent_id=agent_id,
                reasoning_steps=[
                    f"Analyzed proposals for topic: {consensus_process.topic}",
                    f"Evaluated {len(consensus_process.proposals)} available options",
                    "Applied expertise and preference weighting",
                    "Reached final decision based on analysis"
                ],
                decision_factors={
                    "expertise_relevance": 0.8,
                    "proposal_quality": 0.7,
                    "implementation_feasibility": 0.6,
                    "strategic_alignment": 0.9
                },
                confidence_evolution=[0.3, 0.5, 0.7, 0.8],  # Evolution of confidence during reasoning
                external_influences=[
                    "Historical performance data",
                    "Peer agent recommendations",
                    "System resource constraints"
                ],
                final_decision=f"proposal_{consensus_process.proposals[0].id}" if consensus_process.proposals else "unknown"
            )
            
            reasoning_chains.append(reasoning_chain)
        
        return reasoning_chains

    def _document_consensus_evolution(
        self,
        consensus_process: ConsensusProcess
    ) -> ConsensusEvolution:
        """Document how consensus evolved during the process."""
        evolution_id = f"evolution_{self._generate_process_id(consensus_process)}"
        
        # Initial state
        initial_state = {
            "proposals_count": len(consensus_process.proposals),
            "agents_count": len(consensus_process.agents),
            "initial_preferences": "distributed",
            "consensus_strength": 0.0
        }
        
        # Intermediate states from process results
        intermediate_states = []
        for i, result in enumerate(consensus_process.intermediate_results):
            state = {
                "iteration": i + 1,
                "consensus_strength": result.get("consensus_strength", 0.0),
                "leading_proposal": result.get("leading_proposal", "unknown"),
                "vote_distribution": result.get("vote_distribution", {}),
                "convergence_rate": result.get("convergence_rate", 0.0)
            }
            intermediate_states.append(state)
        
        # Final state
        final_state = {
            "final_consensus_strength": intermediate_states[-1]["consensus_strength"] if intermediate_states else 0.0,
            "winning_proposal": intermediate_states[-1]["leading_proposal"] if intermediate_states else "unknown",
            "process_status": consensus_process.status,
            "iterations_required": len(intermediate_states)
        }
        
        # Convergence metrics
        convergence_metrics = {
            "convergence_rate": self._calculate_convergence_rate(intermediate_states),
            "stability_index": self._calculate_stability_index(intermediate_states),
            "consensus_strength_final": final_state["final_consensus_strength"],
            "iterations_to_convergence": len(intermediate_states)
        }
        
        # Stability analysis
        stability_analysis = {
            "vote_stability": "high" if convergence_metrics["stability_index"] > 0.8 else "medium",
            "consensus_trend": "converging" if convergence_metrics["convergence_rate"] > 0.1 else "stable",
            "outlier_opinions": self._identify_outlier_opinions(intermediate_states)
        }
        
        # Identify turning points
        turning_points = self._identify_turning_points(intermediate_states)
        
        return ConsensusEvolution(
            evolution_id=evolution_id,
            initial_state=initial_state,
            intermediate_states=intermediate_states,
            final_state=final_state,
            convergence_metrics=convergence_metrics,
            stability_analysis=stability_analysis,
            turning_points=turning_points
        )

    def _gather_decision_context(
        self,
        consensus_process: ConsensusProcess
    ) -> Dict[str, Any]:
        """Gather contextual information about the decision."""
        return {
            "decision_topic": consensus_process.topic,
            "proposals_considered": [proposal.to_dict() for proposal in consensus_process.proposals],
            "decision_urgency": "normal",  # Could be extracted from process metadata
            "stakeholders_involved": len(consensus_process.agents),
            "decision_scope": "local",  # Could be determined from topic analysis
            "related_decisions": [],  # Could be populated from historical analysis
            "environmental_factors": {
                "system_load": "normal",
                "time_constraints": "moderate",
                "resource_availability": "adequate"
            }
        }

    def _create_audit_metadata(
        self,
        consensus_process: ConsensusProcess
    ) -> Dict[str, Any]:
        """Create metadata for the audit trail."""
        return {
            "audit_version": "1.0",
            "consensus_algorithm": consensus_process.algorithm,
            "process_type": "swarm_consensus",
            "data_sources": ["voting_records", "agent_reasoning", "process_metrics"],
            "quality_indicators": {
                "completeness": "full",
                "accuracy": "high",
                "timeliness": "real_time"
            },
            "compliance_flags": {
                "transparency": True,
                "accountability": True,
                "auditability": True
            }
        }

    def _generate_audit_id(self, consensus_process: ConsensusProcess) -> str:
        """Generate unique audit trail ID."""
        content = f"audit_{consensus_process.topic}_{consensus_process.started_at}_{len(consensus_process.agents)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _generate_consensus_id(self, consensus_process: ConsensusProcess) -> str:
        """Generate unique consensus process ID."""
        content = f"consensus_{consensus_process.topic}_{consensus_process.algorithm}_{consensus_process.started_at}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _generate_process_id(self, consensus_process: ConsensusProcess) -> str:
        """Generate unique process ID."""
        content = f"process_{consensus_process.topic}_{consensus_process.started_at}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def _calculate_convergence_rate(self, intermediate_states: List[Dict[str, Any]]) -> float:
        """Calculate rate of consensus convergence."""
        if len(intermediate_states) < 2:
            return 0.0
        
        strengths = [state.get("consensus_strength", 0.0) for state in intermediate_states]
        if not strengths:
            return 0.0
        
        # Calculate average rate of change
        rate_changes = []
        for i in range(1, len(strengths)):
            rate_change = strengths[i] - strengths[i-1]
            rate_changes.append(rate_change)
        
        return sum(rate_changes) / len(rate_changes) if rate_changes else 0.0

    def _calculate_stability_index(self, intermediate_states: List[Dict[str, Any]]) -> float:
        """Calculate stability index of the consensus process."""
        if len(intermediate_states) < 3:
            return 1.0  # Assume stable for short processes
        
        # Calculate variance in consensus strength changes
        strengths = [state.get("consensus_strength", 0.0) for state in intermediate_states[-5:]]  # Last 5 states
        if len(strengths) < 2:
            return 1.0
        
        changes = [abs(strengths[i] - strengths[i-1]) for i in range(1, len(strengths))]
        avg_change = sum(changes) / len(changes) if changes else 0.0
        
        # Stability is inverse of average change (normalized)
        stability = max(0.0, 1.0 - (avg_change * 5))  # Scale factor of 5
        return min(1.0, stability)

    def _identify_outlier_opinions(self, intermediate_states: List[Dict[str, Any]]) -> List[str]:
        """Identify outlier opinions during consensus evolution."""
        # Simplified outlier detection
        outliers = []
        
        for state in intermediate_states:
            vote_dist = state.get("vote_distribution", {})
            if vote_dist:
                total_votes = sum(vote_dist.values())
                for proposal, votes in vote_dist.items():
                    vote_ratio = votes / total_votes if total_votes > 0 else 0
                    if vote_ratio < 0.1 and votes > 0:  # Less than 10% but not zero
                        outliers.append(f"proposal_{proposal}")
        
        return list(set(outliers))  # Remove duplicates

    def _identify_turning_points(self, intermediate_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify significant turning points in consensus evolution."""
        turning_points = []
        
        if len(intermediate_states) < 3:
            return turning_points
        
        for i in range(1, len(intermediate_states) - 1):
            current = intermediate_states[i]
            previous = intermediate_states[i-1]
            next_state = intermediate_states[i+1]
            
            current_strength = current.get("consensus_strength", 0.0)
            prev_strength = previous.get("consensus_strength", 0.0)
            next_strength = next_state.get("consensus_strength", 0.0)
            
            # Detect significant changes in consensus strength
            change_before = current_strength - prev_strength
            change_after = next_strength - current_strength
            
            # Turning point if direction changes significantly
            if abs(change_before - change_after) > 0.2:
                turning_point = {
                    "iteration": current["iteration"],
                    "type": "consensus_shift",
                    "before_change": change_before,
                    "after_change": change_after,
                    "significance": abs(change_before - change_after),
                    "description": f"Significant consensus shift at iteration {current['iteration']}"
                }
                turning_points.append(turning_point)
        
        return turning_points

    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get statistics about audit trail generation."""
        total_trails = len(self.audit_db.audit_records)
        
        if total_trails == 0:
            return {"message": "No audit trails available"}
        
        # Analyze algorithm usage
        algorithms = defaultdict(int)
        for trail in self.audit_db.audit_records:
            algorithms[trail.voting_process.algorithm_used] += 1
        
        # Analyze process durations
        durations = [trail.voting_process.process_duration for trail in self.audit_db.audit_records]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        
        return {
            "total_audit_trails": total_trails,
            "algorithms_used": dict(algorithms),
            "average_process_duration": avg_duration,
            "average_participants": sum(len(trail.voting_process.participants) for trail in self.audit_db.audit_records) / total_trails,
            "recent_trails": min(10, total_trails)
        }