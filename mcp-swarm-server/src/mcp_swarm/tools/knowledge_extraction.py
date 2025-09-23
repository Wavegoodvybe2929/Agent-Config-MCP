"""
Knowledge Extraction Engine for Collective Knowledge Contribution

Implements automated knowledge extraction from interactions, task outcomes, decision patterns,
and failure analysis for the MCP Swarm Intelligence Server collective knowledge management.

According to orchestrator routing:
- Primary: hive_mind_specialist.md
- Secondary: code.md, memory_management_specialist.md
"""
import logging
import re
import time
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class ExtractionType(Enum):
    """Types of knowledge extraction patterns"""
    TASK_OUTCOME = "task_outcome"
    AGENT_INTERACTION = "agent_interaction"
    DECISION_PATTERN = "decision_pattern"
    FAILURE_ANALYSIS = "failure_analysis"
    SUCCESS_PATTERN = "success_pattern"
    COORDINATION_INSIGHT = "coordination_insight"

@dataclass
class ExtractionPattern:
    """Pattern for knowledge extraction"""
    pattern_type: ExtractionType
    regex_pattern: str
    confidence_threshold: float = 0.7
    metadata_extractors: List[str] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)

@dataclass
class KnowledgeCandidate:
    """Candidate knowledge extracted from content"""
    content: str
    source_type: str
    extraction_type: ExtractionType
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    source_hash: str = field(init=False)
    
    def __post_init__(self):
        """Generate source hash for deduplication"""
        content_normalized = re.sub(r'\s+', ' ', self.content.strip().lower())
        self.source_hash = hashlib.md5(content_normalized.encode()).hexdigest()

@dataclass
class AgentInteraction:
    """Represents an agent interaction for knowledge extraction"""
    agent_id: str
    task_id: str
    interaction_type: str
    content: str
    success: bool
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class CompletedTask:
    """Represents a completed task for knowledge extraction"""
    task_id: str
    agent_ids: List[str]
    task_type: str
    outcome: str
    success: bool
    duration: float
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class TaskOutcomeExtractor:
    """Extracts knowledge from completed tasks"""
    
    def __init__(self):
        self.patterns = [
            ExtractionPattern(
                pattern_type=ExtractionType.TASK_OUTCOME,
                regex_pattern=r"(?i)task\s+(\w+)\s+(?:completed|succeeded|failed).*?because\s+(.+?)(?:\.|$)",
                confidence_threshold=0.8,
                metadata_extractors=["task_type", "outcome_reason", "duration"],
                validation_rules=["has_task_id", "has_outcome", "has_reason"]
            ),
            ExtractionPattern(
                pattern_type=ExtractionType.SUCCESS_PATTERN,
                regex_pattern=r"(?i)(?:successful|effective|optimal)\s+(?:approach|strategy|method):\s+(.+?)(?:\.|$)",
                confidence_threshold=0.75,
                metadata_extractors=["success_factors", "context"],
                validation_rules=["has_strategy", "is_actionable"]
            )
        ]
    
    async def extract_from_task(self, task: CompletedTask) -> List[KnowledgeCandidate]:
        """Extract knowledge from a completed task"""
        candidates = []
        
        # Extract outcome patterns
        if task.success and task.quality_score and task.quality_score > 0.8:
            knowledge_content = f"Task {task.task_type} completed successfully with quality score {task.quality_score}. "
            knowledge_content += f"Duration: {task.duration:.2f}s. Agents involved: {', '.join(task.agent_ids)}. "
            knowledge_content += f"Outcome: {task.outcome}"
            
            candidate = KnowledgeCandidate(
                content=knowledge_content,
                source_type="task_completion",
                extraction_type=ExtractionType.TASK_OUTCOME,
                confidence=min(task.quality_score, 0.95),
                metadata={
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "agent_count": len(task.agent_ids),
                    "duration": task.duration,
                    "quality_score": task.quality_score,
                    "success": task.success
                }
            )
            candidates.append(candidate)
        
        # Extract failure patterns for learning
        elif not task.success:
            knowledge_content = f"Task {task.task_type} failed. "
            knowledge_content += f"Duration: {task.duration:.2f}s. Agents involved: {', '.join(task.agent_ids)}. "
            knowledge_content += f"Failure reason: {task.outcome}"
            
            candidate = KnowledgeCandidate(
                content=knowledge_content,
                source_type="task_failure",
                extraction_type=ExtractionType.FAILURE_ANALYSIS,
                confidence=0.85,
                metadata={
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "agent_count": len(task.agent_ids),
                    "duration": task.duration,
                    "failure_reason": task.outcome,
                    "success": task.success
                }
            )
            candidates.append(candidate)
        
        return candidates

class InteractionExtractor:
    """Extracts knowledge from agent interactions"""
    
    def __init__(self):
        self.patterns = [
            ExtractionPattern(
                pattern_type=ExtractionType.AGENT_INTERACTION,
                regex_pattern=r"(?i)agent\s+(\w+)\s+(?:discovered|learned|found)\s+(.+?)(?:\.|$)",
                confidence_threshold=0.75,
                metadata_extractors=["agent_id", "discovery_type"],
                validation_rules=["has_agent", "has_discovery", "is_valuable"]
            ),
            ExtractionPattern(
                pattern_type=ExtractionType.COORDINATION_INSIGHT,
                regex_pattern=r"(?i)coordination\s+(?:improved|optimized|enhanced)\s+by\s+(.+?)(?:\.|$)",
                confidence_threshold=0.8,
                metadata_extractors=["improvement_method", "coordination_type"],
                validation_rules=["has_method", "is_actionable"]
            )
        ]
    
    async def extract_from_interaction(self, interaction: AgentInteraction) -> List[KnowledgeCandidate]:
        """Extract knowledge from agent interaction"""
        candidates = []
        
        # Extract successful coordination patterns
        if interaction.success and interaction.duration < 10.0:  # Fast successful interactions
            knowledge_content = f"Agent {interaction.agent_id} successfully completed {interaction.interaction_type} "
            knowledge_content += f"in {interaction.duration:.2f}s. Content: {interaction.content[:200]}..."
            
            candidate = KnowledgeCandidate(
                content=knowledge_content,
                source_type="agent_interaction",
                extraction_type=ExtractionType.AGENT_INTERACTION,
                confidence=0.75,
                metadata={
                    "agent_id": interaction.agent_id,
                    "task_id": interaction.task_id,
                    "interaction_type": interaction.interaction_type,
                    "duration": interaction.duration,
                    "success": interaction.success
                }
            )
            candidates.append(candidate)
        
        return candidates

class DecisionPatternExtractor:
    """Extracts decision patterns and strategies"""
    
    def __init__(self):
        self.decision_keywords = {
            "strategy_selection", "consensus_building", "task_assignment", 
            "resource_allocation", "conflict_resolution", "optimization"
        }
    
    async def extract_decision_patterns(self, content: str, context: Dict[str, Any]) -> List[KnowledgeCandidate]:
        """Extract decision patterns from content"""
        candidates = []
        
        # Look for decision-making patterns
        decision_patterns = re.findall(r"(?i)decision\s+to\s+(.+?)\s+(?:resulted|led)\s+to\s+(.+?)(?:\.|$)", content)
        
        for decision, outcome in decision_patterns:
            if any(keyword in decision.lower() for keyword in self.decision_keywords):
                knowledge_content = f"Decision pattern: {decision.strip()} resulted in {outcome.strip()}"
                
                candidate = KnowledgeCandidate(
                    content=knowledge_content,
                    source_type="decision_analysis",
                    extraction_type=ExtractionType.DECISION_PATTERN,
                    confidence=0.8,
                    metadata={
                        "decision_type": decision.strip(),
                        "outcome": outcome.strip(),
                        "context": context
                    }
                )
                candidates.append(candidate)
        
        return candidates

class FailureAnalysisExtractor:
    """Extracts insights from failures for learning"""
    
    def __init__(self):
        self.failure_indicators = {
            "timeout", "error", "failed", "exception", "conflict", 
            "inconsistency", "deadlock", "bottleneck"
        }
    
    async def extract_failure_insights(self, content: str, context: Dict[str, Any]) -> List[KnowledgeCandidate]:
        """Extract failure insights for learning"""
        candidates = []
        
        # Extract failure patterns with causes
        failure_patterns = re.findall(
            r"(?i)(?:failure|error|issue)\s+(?:in|with|during)\s+(.+?)\s+(?:caused|due)\s+to\s+(.+?)(?:\.|$)", 
            content
        )
        
        for component, cause in failure_patterns:
            knowledge_content = f"Failure analysis: {component.strip()} failed due to {cause.strip()}"
            
            candidate = KnowledgeCandidate(
                content=knowledge_content,
                source_type="failure_analysis",
                extraction_type=ExtractionType.FAILURE_ANALYSIS,
                confidence=0.85,
                metadata={
                    "component": component.strip(),
                    "failure_cause": cause.strip(),
                    "context": context,
                    "prevention_potential": True
                }
            )
            candidates.append(candidate)
        
        return candidates

class KnowledgeExtractionEngine:
    """Main engine for extracting knowledge from various sources"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "src/data/memory.db"
        self.extractors = {
            "task_outcomes": TaskOutcomeExtractor(),
            "agent_interactions": InteractionExtractor(),
            "decision_patterns": DecisionPatternExtractor(),
            "failure_analysis": FailureAnalysisExtractor()
        }
        
    async def extract_knowledge_from_interaction(
        self, 
        interaction: AgentInteraction
    ) -> List[KnowledgeCandidate]:
        """Extract knowledge from agent interactions"""
        try:
            return await self.extractors["agent_interactions"].extract_from_interaction(interaction)
        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Error extracting knowledge from interaction: %s", str(e))
            return []
    
    async def extract_knowledge_from_task(
        self, 
        task: CompletedTask
    ) -> List[KnowledgeCandidate]:
        """Extract knowledge from completed tasks"""
        try:
            return await self.extractors["task_outcomes"].extract_from_task(task)
        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Error extracting knowledge from task: %s", str(e))
            return []
    
    async def extract_knowledge_from_content(
        self, 
        content: str, 
        source_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[KnowledgeCandidate]:
        """Extract knowledge from arbitrary content"""
        candidates = []
        context = context or {}
        
        try:
            # Extract decision patterns
            decision_candidates = await self.extractors["decision_patterns"].extract_decision_patterns(
                content, context
            )
            candidates.extend(decision_candidates)
            
            # Extract failure insights
            failure_candidates = await self.extractors["failure_analysis"].extract_failure_insights(
                content, context
            )
            candidates.extend(failure_candidates)
            
            # General pattern extraction
            general_candidates = self._extract_general_patterns(content, source_type, context)
            candidates.extend(general_candidates)
            
        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Error extracting knowledge from content: %s", str(e))
        
        return candidates
    
    def _identify_extractable_patterns(
        self, 
        content: str
    ) -> List[ExtractionPattern]:
        """Identify patterns suitable for knowledge extraction"""
        patterns = []
        content_lower = content.lower()
        
        # Identify pattern types based on content
        if any(word in content_lower for word in ["completed", "succeeded", "achieved"]):
            patterns.append(ExtractionPattern(
                pattern_type=ExtractionType.TASK_OUTCOME,
                regex_pattern=r"(?i)(?:completed|succeeded|achieved)\s+(.+?)(?:\.|$)",
                confidence_threshold=0.7
            ))
        
        if any(word in content_lower for word in ["failed", "error", "exception"]):
            patterns.append(ExtractionPattern(
                pattern_type=ExtractionType.FAILURE_ANALYSIS,
                regex_pattern=r"(?i)(?:failed|error|exception)\s+(.+?)(?:\.|$)",
                confidence_threshold=0.8
            ))
        
        if any(word in content_lower for word in ["decided", "chosen", "selected"]):
            patterns.append(ExtractionPattern(
                pattern_type=ExtractionType.DECISION_PATTERN,
                regex_pattern=r"(?i)(?:decided|chosen|selected)\s+(.+?)(?:\.|$)",
                confidence_threshold=0.75
            ))
        
        return patterns
    
    def _extract_general_patterns(
        self, 
        content: str, 
        source_type: str, 
        context: Dict[str, Any]
    ) -> List[KnowledgeCandidate]:
        """Extract general knowledge patterns from content"""
        candidates = []
        
        # Extract key insights and learnings
        insight_patterns = re.findall(
            r"(?i)(?:learned|discovered|found|realized)\s+that\s+(.+?)(?:\.|$)", 
            content
        )
        
        for insight in insight_patterns:
            if len(insight.strip()) > 10:  # Minimum meaningful content
                candidate = KnowledgeCandidate(
                    content=f"Insight: {insight.strip()}",
                    source_type=source_type,
                    extraction_type=ExtractionType.COORDINATION_INSIGHT,
                    confidence=0.7,
                    metadata={
                        "source_type": source_type,
                        "context": context,
                        "extraction_method": "general_pattern"
                    }
                )
                candidates.append(candidate)
        
        return candidates
    
    def _validate_knowledge_candidate(
        self, 
        candidate: KnowledgeCandidate
    ) -> bool:
        """Validate extracted knowledge before storing"""
        # Minimum content length
        if len(candidate.content.strip()) < 10:
            return False
        
        # Confidence threshold
        if candidate.confidence < 0.5:
            return False
        
        # Check for meaningful content (not just noise)
        if candidate.content.count(' ') < 2:  # At least 3 words
            return False
        
        # Avoid duplicate or near-duplicate content
        # This would be enhanced with actual duplicate detection
        
        return True