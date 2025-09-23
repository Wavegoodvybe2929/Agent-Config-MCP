"""
Swarm Consensus MCP Tool

This module provides the main MCP tool interface for swarm consensus decisions
in the MCP Swarm Intelligence Server, integrating all consensus components
including algorithms, confidence scoring, minority preservation, and audit trails.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .consensus_algorithms import (
    ConsensusAlgorithm, 
    Proposal, 
    ConsensusResult,
    ConsensusTimeoutError,
    create_consensus_algorithm
)
from .decision_confidence import DecisionConfidenceScorer
from .minority_opinion import MinorityOpinionPreserver
from .decision_audit import (
    DecisionAuditTrailGenerator, 
    AuditDatabase, 
    ConsensusProcess
)
from ..swarm import SwarmCoordinator
from ..swarm.aco import Agent

logger = logging.getLogger(__name__)


class SwarmConsensusError(Exception):
    """Base exception for swarm consensus errors."""


class InsufficientParticipationError(SwarmConsensusError):
    """Raised when insufficient agents participate in consensus."""


def mcp_tool(name: str):
    """Decorator for MCP tools (placeholder implementation)."""
    def decorator(func):
        func.mcp_tool_name = name  # Use public attribute instead
        return func
    return decorator


# Global instances (in real implementation, these would be properly injected)
swarm_coordinator = SwarmCoordinator()
audit_db = AuditDatabase()


async def get_available_agents() -> List[Agent]:
    """Get list of available agents for consensus participation."""
    # In real implementation, this would query the actual agent registry
    # For now, return simulated agents
    agents = []
    for i in range(10):  # Simulate 10 available agents
        agent = Agent(
            id=f"agent_{i}",
            capabilities=["analysis", "decision_making"],
            current_load=0.3,  # 30% load
            success_rate=0.8,  # 80% success rate
            availability=True
        )
        # Add custom attributes for consensus filtering using hasattr checks
        setattr(agent, 'expertise_domains', ["technical", "strategic"])
        setattr(agent, 'load', agent.current_load)  # Create load alias
        agents.append(agent)
    return agents


async def filter_eligible_agents(
    available_agents: List[Agent], 
    decision_topic: str
) -> List[Agent]:
    """Filter agents eligible for the specific decision topic."""
    # In real implementation, this would check agent capabilities and expertise
    # For now, return agents with low load and relevant expertise
    eligible_agents = []
    
    for agent in available_agents:
        # Check if agent has capacity (load < 80%)
        agent_load = getattr(agent, 'load', agent.current_load)
        if agent_load < 0.8:
            # Check if agent has relevant expertise
            topic_lower = decision_topic.lower()
            expertise_domains = getattr(agent, 'expertise_domains', ['general'])
            if any(domain in topic_lower for domain in expertise_domains):
                eligible_agents.append(agent)
            elif "general" in expertise_domains:
                eligible_agents.append(agent)
    
    return eligible_agents


def generate_implementation_recommendations(
    consensus_result: ConsensusResult,
    confidence_score: Any  # ConfidenceScore type
) -> List[str]:
    """Generate implementation recommendations based on consensus result."""
    recommendations = []
    
    # Base recommendation
    recommendations.append(f"Implement proposal {consensus_result.winning_proposal.id}")
    
    # Confidence-based recommendations
    if confidence_score.overall_confidence > 0.8:
        recommendations.append("High confidence decision - proceed with full implementation")
    elif confidence_score.overall_confidence > 0.6:
        recommendations.append("Moderate confidence - consider phased implementation")
        recommendations.append("Monitor key metrics during implementation")
    else:
        recommendations.append("Low confidence - recommend pilot testing first")
        recommendations.append("Gather additional input before full implementation")
    
    # Consensus strength recommendations
    if consensus_result.consensus_strength < 0.6:
        recommendations.append("Consider consensus building activities before implementation")
        recommendations.append("Address concerns raised by minority opinions")
    
    # Participation-based recommendations
    if consensus_result.participation_count < 5:
        recommendations.append("Low participation - consider broader consultation")
    
    # Algorithm-specific recommendations
    if consensus_result.algorithm_used == "byzantine_fault_tolerant":
        recommendations.append("Review security implications before implementation")
    elif consensus_result.algorithm_used == "swarm_consensus":
        recommendations.append("Leverage swarm intelligence patterns during implementation")
    
    return recommendations


def generate_retry_recommendations(consensus_process: ConsensusProcess) -> List[str]:
    """Generate recommendations for retrying failed consensus."""
    recommendations = []
    
    # Timeout-specific recommendations
    recommendations.append("Increase timeout duration for complex decisions")
    recommendations.append("Consider breaking complex decisions into smaller components")
    
    # Participation recommendations
    if len(consensus_process.agents) < 3:
        recommendations.append("Recruit more agents for consensus participation")
    
    # Algorithm recommendations
    if consensus_process.algorithm == "byzantine_fault_tolerant":
        recommendations.append("Consider switching to weighted voting for faster consensus")
    elif consensus_process.algorithm == "swarm_consensus":
        recommendations.append("Adjust pheromone parameters for faster convergence")
    
    # Process recommendations
    recommendations.append("Review decision complexity and scope")
    recommendations.append("Ensure all agents have necessary context and information")
    
    return recommendations


@mcp_tool("swarm_consensus")
async def swarm_consensus_tool(
    decision_topic: str,
    proposals: List[str],
    consensus_algorithm: str = "weighted_voting",
    timeout_seconds: float = 30.0,
    minimum_participation: float = 0.7,
    preserve_minority: bool = True
) -> Dict[str, Any]:
    """
    MCP tool for reaching consensus decisions using swarm intelligence.
    
    Args:
        decision_topic: Topic or question requiring consensus
        proposals: List of proposal options
        consensus_algorithm: Algorithm to use for consensus
        timeout_seconds: Maximum time to reach consensus
        minimum_participation: Minimum agent participation required
        preserve_minority: Whether to preserve minority opinions
        
    Returns:
        Consensus decision with confidence scores and audit trail
    """
    try:
        # Validate inputs
        if not decision_topic or not proposals:
            return {
                "status": "error",
                "error": "Decision topic and proposals are required",
                "error_type": "invalid_input"
            }
        
        if consensus_algorithm not in [algo.value for algo in ConsensusAlgorithm]:
            return {
                "status": "error",
                "error": f"Unsupported consensus algorithm: {consensus_algorithm}",
                "supported_algorithms": [algo.value for algo in ConsensusAlgorithm],
                "error_type": "invalid_algorithm"
            }
        
        # Initialize consensus components
        try:
            consensus_algo = create_consensus_algorithm(
                ConsensusAlgorithm(consensus_algorithm),
                swarm_coordinator=swarm_coordinator
            )
        except ValueError as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": "algorithm_creation_failed"
            }
        
        confidence_scorer = DecisionConfidenceScorer()
        minority_preserver = MinorityOpinionPreserver()
        audit_generator = DecisionAuditTrailGenerator(audit_db)
        
        # Prepare proposals and get available agents
        proposal_objects = [
            Proposal(id=i, content=content, topic=decision_topic)
            for i, content in enumerate(proposals)
        ]
        
        available_agents = await get_available_agents()
        participating_agents = await filter_eligible_agents(
            available_agents, decision_topic
        )
        
        # Check minimum participation
        participation_rate = len(participating_agents) / max(1, len(available_agents))
        if participation_rate < minimum_participation:
            return {
                "status": "insufficient_participation",
                "participation_rate": participation_rate,
                "required_rate": minimum_participation,
                "available_agents": len(available_agents),
                "participating_agents": len(participating_agents),
                "recommendations": [
                    "Reduce minimum participation requirement",
                    "Expand agent pool for this decision type",
                    "Review agent eligibility criteria"
                ]
            }
        
        # Create consensus process
        consensus_process = ConsensusProcess(
            topic=decision_topic,
            proposals=proposal_objects,
            agents=participating_agents,
            algorithm=consensus_algorithm,
            started_at=datetime.now()
        )
        
        # Execute consensus algorithm
        try:
            logger.info("Starting consensus process for topic: %s", decision_topic)
            
            consensus_result = await asyncio.wait_for(
                consensus_algo.reach_consensus(
                    proposal_objects, participating_agents, timeout_seconds
                ),
                timeout=timeout_seconds + 5  # Extra buffer for cleanup
            )
            
            consensus_process.completed_at = datetime.now()
            consensus_process.status = "completed"
            
            logger.info("Consensus reached for topic: %s", decision_topic)
            
        except ConsensusTimeoutError:
            consensus_process.completed_at = datetime.now()
            consensus_process.status = "timeout"
            
            return {
                "status": "consensus_timeout",
                "partial_results": consensus_process.get_partial_results(),
                "participation_rate": participation_rate,
                "timeout_seconds": timeout_seconds,
                "retry_recommendations": generate_retry_recommendations(consensus_process)
            }
            
        except asyncio.TimeoutError:
            consensus_process.completed_at = datetime.now()
            consensus_process.status = "timeout"
            
            return {
                "status": "consensus_timeout",
                "partial_results": consensus_process.get_partial_results(),
                "participation_rate": participation_rate,
                "timeout_seconds": timeout_seconds,
                "retry_recommendations": generate_retry_recommendations(consensus_process)
            }
            
        except (ValueError, TypeError, AttributeError) as e:
            consensus_process.completed_at = datetime.now()
            consensus_process.status = "failed"
            
            logger.error("Consensus process failed: %s", e)
            return {
                "status": "consensus_failed",
                "error": str(e),
                "error_type": "consensus_execution_error",
                "partial_results": consensus_process.get_partial_results()
            }
        
        # Calculate decision confidence
        try:
            confidence_score = await confidence_scorer.calculate_decision_confidence(
                consensus_result
            )
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to calculate confidence score: %s", e)
            # Create a basic confidence score
            confidence_score = type('ConfidenceScore', (), {
                'overall_confidence': 0.5,
                'uncertainty_level': 0.5,
                'to_dict': lambda: {"overall_confidence": 0.5, "uncertainty_level": 0.5}
            })()
        
        # Preserve minority opinions if requested
        minority_record = None
        if preserve_minority:
            try:
                minority_record = await minority_preserver.preserve_minority_opinions(
                    consensus_result
                )
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning("Failed to preserve minority opinion: %s", e)        # Generate audit trail
        audit_trail = None
        try:
            audit_trail = await audit_generator.create_decision_audit_trail(
                consensus_process
            )
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to create audit trail: %s", e)
        
        # Generate implementation recommendations
        implementation_recommendations = generate_implementation_recommendations(
            consensus_result, confidence_score
        )
        
        return {
            "status": "consensus_reached",
            "decision": consensus_result.winning_proposal.to_dict(),
            "confidence_score": getattr(confidence_score, '__dict__', {"overall_confidence": 0.5}) if confidence_score else {"overall_confidence": 0.5},
            "participation_rate": participation_rate,
            "algorithm_used": consensus_algorithm,
            "voting_results": consensus_result.vote_summary,
            "minority_opinions": minority_record.to_dict() if minority_record else None,
            "audit_trail_id": audit_trail.id if audit_trail else None,
            "decision_timestamp": consensus_result.completed_at.isoformat(),
            "implementation_recommendations": implementation_recommendations,
            "process_metrics": {
                "total_agents": len(available_agents),
                "participating_agents": len(participating_agents),
                "consensus_strength": consensus_result.consensus_strength,
                "process_duration": (consensus_process.completed_at - consensus_process.started_at).total_seconds()
            }
        }
        
    except (ValueError, TypeError, AttributeError) as e:
        logger.error("Error in swarm consensus tool: %s", e)
        return {
            "status": "error",
            "error": f"Error: {str(e)}",
            "error_type": "consensus_error"
        }


@mcp_tool("consensus_statistics")
async def consensus_statistics_tool() -> Dict[str, Any]:
    """
    Get statistics about consensus processes and system performance.
    
    Returns:
        Statistics about consensus operations
    """
    try:
        # Get confidence scorer statistics
        confidence_scorer = DecisionConfidenceScorer()
        confidence_stats = confidence_scorer.get_confidence_statistics()
        
        # Get minority opinion statistics
        minority_preserver = MinorityOpinionPreserver()
        minority_stats = minority_preserver.get_minority_opinion_statistics()
        
        # Get audit trail statistics
        audit_generator = DecisionAuditTrailGenerator(audit_db)
        audit_stats = audit_generator.get_audit_statistics()
        
        # Get swarm coordinator metrics
        swarm_metrics = swarm_coordinator.coordination_metrics
        
        return {
            "status": "success",
            "confidence_statistics": confidence_stats,
            "minority_opinion_statistics": minority_stats,
            "audit_trail_statistics": audit_stats,
            "swarm_coordination_metrics": {
                "total_assignments": swarm_metrics.total_assignments,
                "successful_assignments": swarm_metrics.successful_assignments,
                "consensus_decisions": swarm_metrics.consensus_decisions,
                "unanimous_decisions": swarm_metrics.unanimous_decisions,
                "average_confidence": swarm_metrics.average_confidence,
                "coordination_efficiency": swarm_metrics.coordination_efficiency
            },
            "system_health": {
                "active_agents": len(await get_available_agents()),
                "available_algorithms": [algo.value for algo in ConsensusAlgorithm],
                "audit_records_count": len(audit_db.audit_records)
            }
        }
        
    except (ValueError, TypeError, AttributeError) as e:
        logger.error("Error getting consensus statistics: %s", e)
        return {
            "status": "error",
            "error": str(e),
            "error_type": "statistics_error"
        }


@mcp_tool("agent_consensus_history")
async def agent_consensus_history_tool(
    agent_id: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get consensus history for a specific agent or all agents.
    
    Args:
        agent_id: Specific agent ID to get history for (optional)
        limit: Maximum number of records to return
        
    Returns:
        Consensus history records
    """
    try:
        # Search audit trails
        search_criteria = {}
        if agent_id:
            search_criteria["agent_id"] = agent_id
            
        audit_trails = audit_db.search_audit_trails(search_criteria)
        
        # Limit results
        audit_trails = audit_trails[-limit:] if len(audit_trails) > limit else audit_trails
        
        # Extract relevant information
        history_records = []
        for trail in audit_trails:
            record = {
                "consensus_id": trail.consensus_id,
                "decision_topic": trail.decision_context.get("decision_topic", "unknown"),
                "algorithm_used": trail.voting_process.algorithm_used,
                "process_duration": trail.voting_process.process_duration,
                "consensus_strength": trail.consensus_evolution.convergence_metrics.get("consensus_strength_final", 0.0),
                "created_at": trail.created_at.isoformat(),
                "participants": len(trail.voting_process.participants)
            }
            
            if agent_id:
                # Add agent-specific information
                record["agent_participated"] = agent_id in trail.voting_process.participants
                # Could add agent-specific reasoning if available
                
            history_records.append(record)
        
        return {
            "status": "success",
            "agent_id": agent_id,
            "total_records": len(history_records),
            "history": history_records
        }
        
    except (ValueError, TypeError, AttributeError) as e:
        logger.error("Error getting agent consensus history: %s", e)
        return {
            "status": "error",
            "error": str(e),
            "error_type": "history_error"
        }


# Export the MCP tools for registration
MCP_TOOLS = [
    swarm_consensus_tool,
    consensus_statistics_tool,
    agent_consensus_history_tool
]