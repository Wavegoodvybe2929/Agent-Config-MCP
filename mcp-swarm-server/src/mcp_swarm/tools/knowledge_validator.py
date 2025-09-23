"""
Knowledge Validation System for Collective Knowledge Management

Implements consistency checking, conflict detection, and resolution strategies for the MCP 
Swarm Intelligence Server collective knowledge contribution system.

According to orchestrator routing:
- Primary: hive_mind_specialist.md
- Secondary: code.md, memory_management_specialist.md
"""
import logging
import re
import time
from typing import List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from .knowledge_classifier import KnowledgeEntry

logger = logging.getLogger(__name__)

class ConflictType(Enum):
    """Types of knowledge conflicts"""
    CONTRADICTORY = "contradictory"         # Direct contradiction
    REDUNDANT = "redundant"                 # Duplicate or near-duplicate
    INCONSISTENT = "inconsistent"           # Logically inconsistent
    OUTDATED = "outdated"                   # Superseded by newer knowledge
    INCOMPLETE = "incomplete"               # Missing key information
    CONTEXT_MISMATCH = "context_mismatch"   # Different contexts but similar content

class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    CRITICAL = "critical"    # Must be resolved before storage
    WARNING = "warning"      # Should be reviewed but can proceed
    INFO = "info"           # Informational, no action required

@dataclass
class ConsistencyIssue:
    """Represents a consistency issue in knowledge"""
    issue_type: str
    severity: ValidationSeverity
    description: str
    affected_content: str
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeConflict:
    """Represents a conflict between knowledge entries"""
    conflict_type: ConflictType
    severity: ValidationSeverity
    new_knowledge_id: str
    existing_knowledge_id: str
    similarity_score: float
    description: str
    conflict_details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConflictResolution:
    """Strategy for resolving knowledge conflicts"""
    resolution_type: str
    action: str                    # merge, replace, keep_both, reject_new
    confidence: float
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Result of knowledge validation"""
    is_valid: bool
    confidence: float
    issues: List[ConsistencyIssue] = field(default_factory=list)
    conflicts: List[KnowledgeConflict] = field(default_factory=list)
    resolutions: List[ConflictResolution] = field(default_factory=list)
    validation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary"""
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "issues": [
                {
                    "type": issue.issue_type,
                    "severity": issue.severity.value,
                    "description": issue.description,
                    "suggestions": issue.suggestions
                }
                for issue in self.issues
            ],
            "conflicts": [
                {
                    "type": conflict.conflict_type.value,
                    "severity": conflict.severity.value,
                    "description": conflict.description,
                    "similarity_score": conflict.similarity_score
                }
                for conflict in self.conflicts
            ],
            "resolutions": [
                {
                    "type": resolution.resolution_type,
                    "action": resolution.action,
                    "confidence": resolution.confidence,
                    "rationale": resolution.rationale
                }
                for resolution in self.resolutions
            ],
            "validation_metadata": self.validation_metadata
        }

class ConsistencyChecker:
    """Checks knowledge for logical consistency"""
    
    def __init__(self):
        self.consistency_patterns = {
            "contradictory_statements": [
                (r"\b(always|never|all|none)\b", r"\b(sometimes|maybe|some|few)\b"),
                (r"\b(increase|improve|enhance)\b", r"\b(decrease|worsen|degrade)\b"),
                (r"\b(success|successful|effective)\b", r"\b(fail|failure|ineffective)\b")
            ],
            "temporal_inconsistencies": [
                (r"\bbefore\b.*\bafter\b", r"\bafter\b.*\bbefore\b"),
                (r"\bfirst\b.*\blast\b", r"\blast\b.*\bfirst\b")
            ],
            "logical_contradictions": [
                (r"\bif\s+(\w+)\s+then\s+(\w+)", r"\bif\s+\1\s+then\s+not\s+\2"),
                (r"\b(\w+)\s+causes\s+(\w+)", r"\b\1\s+prevents\s+\2")
            ]
        }
    
    def check_logical_consistency(self, knowledge_content: str) -> List[ConsistencyIssue]:
        """Check for logical consistency issues"""
        issues = []
        content_lower = knowledge_content.lower()
        
        # Check for contradictory statements
        for pattern_type, pattern_pairs in self.consistency_patterns.items():
            for positive_pattern, negative_pattern in pattern_pairs:
                positive_matches = re.findall(positive_pattern, content_lower)
                negative_matches = re.findall(negative_pattern, content_lower)
                
                if positive_matches and negative_matches:
                    issue = ConsistencyIssue(
                        issue_type=pattern_type,
                        severity=ValidationSeverity.WARNING,
                        description=f"Found potentially contradictory statements: {positive_matches} vs {negative_matches}",
                        affected_content=knowledge_content[:200] + "..." if len(knowledge_content) > 200 else knowledge_content,
                        suggestions=[
                            "Review for logical consistency",
                            "Clarify context or conditions",
                            "Consider splitting into separate knowledge entries"
                        ]
                    )
                    issues.append(issue)
        
        # Check for absolute statements that might be too broad
        absolute_patterns = [r"\balways\b", r"\bnever\b", r"\ball\b", r"\bnone\b"]
        for pattern in absolute_patterns:
            if re.search(pattern, content_lower):
                issue = ConsistencyIssue(
                    issue_type="absolute_statement",
                    severity=ValidationSeverity.INFO,
                    description=f"Contains absolute statement: {pattern}",
                    affected_content=knowledge_content,
                    suggestions=[
                        "Consider if absolute language is appropriate",
                        "Add context or conditions",
                        "Use more qualified language if appropriate"
                    ]
                )
                issues.append(issue)
        
        return issues
    
    def validate_knowledge_structure(self, knowledge_content: str) -> List[ConsistencyIssue]:
        """Validate knowledge structure and completeness"""
        issues = []
        
        # Check minimum content requirements
        if len(knowledge_content.strip()) < 10:
            issues.append(ConsistencyIssue(
                issue_type="insufficient_content",
                severity=ValidationSeverity.CRITICAL,
                description="Knowledge content is too short to be meaningful",
                affected_content=knowledge_content,
                suggestions=["Provide more detailed information", "Add context and examples"]
            ))
        
        # Check for meaningful content (not just punctuation/numbers)
        meaningful_words = re.findall(r'\b[a-zA-Z]{3,}\b', knowledge_content)
        if len(meaningful_words) < 3:
            issues.append(ConsistencyIssue(
                issue_type="insufficient_meaningful_content",
                severity=ValidationSeverity.CRITICAL,
                description="Knowledge lacks sufficient meaningful content",
                affected_content=knowledge_content,
                suggestions=["Add more descriptive content", "Include actionable information"]
            ))
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', knowledge_content)
        complete_sentences = [s for s in sentences if len(s.strip()) > 5 and ' ' in s.strip()]
        if len(complete_sentences) == 0:
            issues.append(ConsistencyIssue(
                issue_type="poor_structure",
                severity=ValidationSeverity.WARNING,
                description="Knowledge lacks proper sentence structure",
                affected_content=knowledge_content,
                suggestions=["Use complete sentences", "Structure information clearly"]
            ))
        
        return issues

class ConflictDetector:
    """Detects conflicts between knowledge entries"""
    
    def __init__(self, db_path: str = "src/data/memory.db"):
        self.db_path = db_path
        self.similarity_threshold = 0.7
        self.high_similarity_threshold = 0.9
    
    def detect_conflicts(
        self, 
        new_knowledge: 'KnowledgeEntry',
        existing_knowledge: List['KnowledgeEntry']
    ) -> List[KnowledgeConflict]:
        """Detect conflicts with existing knowledge"""
        conflicts = []
        
        for existing in existing_knowledge:
            # Calculate content similarity
            similarity = self._calculate_content_similarity(
                new_knowledge.content, existing.content
            )
            
            # Check for redundancy
            if similarity > self.high_similarity_threshold:
                conflict = KnowledgeConflict(
                    conflict_type=ConflictType.REDUNDANT,
                    severity=ValidationSeverity.WARNING,
                    new_knowledge_id=getattr(new_knowledge, 'id', 'new'),
                    existing_knowledge_id=getattr(existing, 'id', 'existing'),
                    similarity_score=similarity,
                    description=f"High similarity ({similarity:.2f}) with existing knowledge",
                    conflict_details={
                        "similarity_type": "content",
                        "threshold_exceeded": self.high_similarity_threshold
                    }
                )
                conflicts.append(conflict)
            
            # Check for contradictions
            elif self._detect_contradiction(new_knowledge.content, existing.content):
                conflict = KnowledgeConflict(
                    conflict_type=ConflictType.CONTRADICTORY,
                    severity=ValidationSeverity.CRITICAL,
                    new_knowledge_id=getattr(new_knowledge, 'id', 'new'),
                    existing_knowledge_id=getattr(existing, 'id', 'existing'),
                    similarity_score=similarity,
                    description="Direct contradiction with existing knowledge",
                    conflict_details={
                        "contradiction_type": "logical",
                        "context_similarity": similarity
                    }
                )
                conflicts.append(conflict)
            
            # Check for context mismatches
            elif similarity > self.similarity_threshold and similarity <= self.high_similarity_threshold:
                if self._detect_context_mismatch(new_knowledge, existing):
                    conflict = KnowledgeConflict(
                        conflict_type=ConflictType.CONTEXT_MISMATCH,
                        severity=ValidationSeverity.INFO,
                        new_knowledge_id=getattr(new_knowledge, 'id', 'new'),
                        existing_knowledge_id=getattr(existing, 'id', 'existing'),
                        similarity_score=similarity,
                        description="Similar content but different contexts",
                        conflict_details={
                            "context_difference": True,
                            "requires_review": True
                        }
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings"""
        # Simple word-based similarity (could be enhanced with semantic similarity)
        words1 = set(re.findall(r'\b\w+\b', content1.lower()))
        words2 = set(re.findall(r'\b\w+\b', content2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _detect_contradiction(self, content1: str, content2: str) -> bool:
        """Detect if two content pieces contradict each other"""
        content1_lower = content1.lower()
        content2_lower = content2.lower()
        
        # Simple contradiction patterns
        contradiction_pairs = [
            ("success", "fail"),
            ("increase", "decrease"),
            ("improve", "worsen"),
            ("effective", "ineffective"),
            ("optimal", "suboptimal"),
            ("should", "should not"),
            ("always", "never"),
            ("all", "none")
        ]
        
        for pos, neg in contradiction_pairs:
            if (pos in content1_lower and neg in content2_lower) or \
               (neg in content1_lower and pos in content2_lower):
                # Check if they're talking about the same subject
                # This is a simplified check - could be enhanced
                common_words = self._get_common_significant_words(content1, content2)
                if len(common_words) >= 2:
                    return True
        
        return False
    
    def _detect_context_mismatch(self, knowledge1: 'KnowledgeEntry', knowledge2: 'KnowledgeEntry') -> bool:
        """Detect if knowledge entries have context mismatches"""
        # Check metadata differences
        meta1 = knowledge1.metadata
        meta2 = knowledge2.metadata
        
        # Different source types might indicate different contexts
        if meta1.get('source_type') != meta2.get('source_type'):
            return True
        
        # Different time periods might indicate context changes
        time_diff = abs(knowledge1.created_at - knowledge2.created_at)
        if time_diff > 7 * 24 * 3600:  # More than a week apart
            return True
        
        return False
    
    def _get_common_significant_words(self, content1: str, content2: str) -> List[str]:
        """Get common significant words between two content pieces"""
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        words1 = {w.lower() for w in re.findall(r'\b\w+\b', content1) if len(w) > 2}
        words2 = {w.lower() for w in re.findall(r'\b\w+\b', content2) if len(w) > 2}
        
        # Remove stop words
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        return list(words1.intersection(words2))

class KnowledgeValidationSystem:
    """Main validation system for knowledge entries"""
    
    def __init__(self, knowledge_db: Any):
        self.knowledge_db = knowledge_db
        self.consistency_checker = ConsistencyChecker()
        self.conflict_detector = ConflictDetector()
        
    async def validate_knowledge_consistency(
        self, 
        new_knowledge: 'KnowledgeEntry'
    ) -> ValidationResult:
        """Validate knowledge consistency with existing knowledge base"""
        
        # Check logical consistency
        consistency_issues = self.consistency_checker.check_logical_consistency(
            new_knowledge.content
        )
        
        # Check knowledge structure
        structure_issues = self.consistency_checker.validate_knowledge_structure(
            new_knowledge.content
        )
        
        all_issues = consistency_issues + structure_issues
        
        # Get related existing knowledge for conflict detection
        existing_knowledge = await self._get_related_knowledge(new_knowledge)
        
        # Detect conflicts
        conflicts = self.conflict_detector.detect_conflicts(new_knowledge, existing_knowledge)
        
        # Generate conflict resolutions
        resolutions = self._resolve_conflicts(conflicts)
        
        # Determine overall validity
        critical_issues = [issue for issue in all_issues if issue.severity == ValidationSeverity.CRITICAL]
        critical_conflicts = [conflict for conflict in conflicts if conflict.severity == ValidationSeverity.CRITICAL]
        
        is_valid = len(critical_issues) == 0 and len(critical_conflicts) == 0
        
        # Calculate validation confidence
        confidence = self._calculate_validation_confidence(all_issues, conflicts, resolutions)
        
        # Generate validation metadata
        validation_metadata = {
            "validation_timestamp": time.time(),
            "total_issues": len(all_issues),
            "critical_issues": len(critical_issues),
            "total_conflicts": len(conflicts),
            "critical_conflicts": len(critical_conflicts),
            "resolutions_generated": len(resolutions),
            "existing_knowledge_checked": len(existing_knowledge)
        }
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=all_issues,
            conflicts=conflicts,
            resolutions=resolutions,
            validation_metadata=validation_metadata
        )
    
    def _check_logical_consistency(
        self, 
        knowledge: 'KnowledgeEntry'
    ) -> List[ConsistencyIssue]:
        """Check for logical consistency issues"""
        return self.consistency_checker.check_logical_consistency(knowledge.content)
    
    def _detect_conflicts(
        self, 
        new_knowledge: 'KnowledgeEntry',
        existing_knowledge: List['KnowledgeEntry']
    ) -> List[KnowledgeConflict]:
        """Detect conflicts with existing knowledge"""
        return self.conflict_detector.detect_conflicts(new_knowledge, existing_knowledge)
    
    def _resolve_conflicts(
        self, 
        conflicts: List[KnowledgeConflict]
    ) -> List[ConflictResolution]:
        """Generate conflict resolution strategies"""
        resolutions = []
        
        for conflict in conflicts:
            if conflict.conflict_type == ConflictType.REDUNDANT:
                if conflict.similarity_score > 0.95:
                    resolution = ConflictResolution(
                        resolution_type="redundancy",
                        action="reject_new",
                        confidence=0.9,
                        rationale="Nearly identical content already exists"
                    )
                else:
                    resolution = ConflictResolution(
                        resolution_type="redundancy",
                        action="merge",
                        confidence=0.7,
                        rationale="Similar content could be consolidated"
                    )
            
            elif conflict.conflict_type == ConflictType.CONTRADICTORY:
                resolution = ConflictResolution(
                    resolution_type="contradiction",
                    action="keep_both",
                    confidence=0.6,
                    rationale="Manual review required for contradictory information"
                )
            
            elif conflict.conflict_type == ConflictType.CONTEXT_MISMATCH:
                resolution = ConflictResolution(
                    resolution_type="context",
                    action="keep_both",
                    confidence=0.8,
                    rationale="Different contexts justify separate entries"
                )
            
            else:
                resolution = ConflictResolution(
                    resolution_type="general",
                    action="review_required",
                    confidence=0.5,
                    rationale="Manual review recommended"
                )
            
            resolutions.append(resolution)
        
        return resolutions
    
    async def _get_related_knowledge(self, _new_knowledge: 'KnowledgeEntry') -> List['KnowledgeEntry']:
        """Get existing knowledge related to the new knowledge"""
        # This would query the knowledge database for related entries
        # For now, return empty list - would be implemented with actual database queries
        try:
            # Simulate database query for related knowledge
            # In real implementation, this would use semantic search or keyword matching
            return []
        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Error retrieving related knowledge: %s", str(e))
            return []
    
    def _calculate_validation_confidence(
        self,
        issues: List[ConsistencyIssue],
        conflicts: List[KnowledgeConflict],
        resolutions: List[ConflictResolution]
    ) -> float:
        """Calculate confidence in validation results"""
        base_confidence = 1.0
        
        # Reduce confidence based on issues
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                base_confidence -= 0.3
            elif issue.severity == ValidationSeverity.WARNING:
                base_confidence -= 0.1
            # INFO issues don't reduce confidence significantly
        
        # Reduce confidence based on conflicts
        for conflict in conflicts:
            if conflict.severity == ValidationSeverity.CRITICAL:
                base_confidence -= 0.4
            elif conflict.severity == ValidationSeverity.WARNING:
                base_confidence -= 0.15
        
        # Increase confidence if resolutions are high-confidence
        resolution_confidence = sum(r.confidence for r in resolutions) / len(resolutions) if resolutions else 1.0
        base_confidence = (base_confidence + resolution_confidence) / 2
        
        return max(0.0, min(1.0, base_confidence))