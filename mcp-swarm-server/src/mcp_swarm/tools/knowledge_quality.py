"""
Knowledge Quality Scorer for Collective Knowledge Management

Implements comprehensive quality assessment including accuracy, completeness, timeliness, 
source reliability, and verification level scoring for the MCP Swarm Intelligence Server.

According to orchestrator routing:
- Primary: hive_mind_specialist.md
- Secondary: code.md, memory_management_specialist.md
"""
import logging
import re
import time
from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class QualityMetric(Enum):
    """Quality assessment metrics"""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    TIMELINESS = "timeliness"
    SOURCE_RELIABILITY = "source_reliability"
    VERIFICATION_LEVEL = "verification_level"
    CLARITY = "clarity"
    ACTIONABILITY = "actionability"
    RELEVANCE = "relevance"

@dataclass
class QualityScore:
    """Comprehensive quality score for knowledge"""
    overall: float
    metrics: Dict[QualityMetric, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.8
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert quality score to dictionary"""
        return {
            "overall": self.overall,
            "metrics": {metric.value: score for metric, score in self.metrics.items()},
            "details": self.details,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "created_at": self.created_at
        }

@dataclass
class KnowledgeEntry:
    """Knowledge entry for quality scoring"""
    content: str
    source_type: str
    extraction_type: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    id: str = ""

class AccuracyAssessor:
    """Assesses knowledge accuracy based on verification and validation"""
    
    def __init__(self):
        self.accuracy_indicators = {
            "high_confidence": {
                "patterns": [r"verified", r"confirmed", r"validated", r"proven"],
                "score_bonus": 0.3
            },
            "medium_confidence": {
                "patterns": [r"tested", r"observed", r"measured", r"documented"],
                "score_bonus": 0.2
            },
            "low_confidence": {
                "patterns": [r"assumed", r"suspected", r"might", r"could"],
                "score_penalty": 0.2
            },
            "uncertainty": {
                "patterns": [r"unknown", r"unclear", r"uncertain", r"maybe"],
                "score_penalty": 0.3
            }
        }
    
    def assess_accuracy(self, knowledge: KnowledgeEntry) -> float:
        """Assess knowledge accuracy based on verification and validation"""
        base_accuracy = knowledge.confidence
        content_lower = knowledge.content.lower()
        
        # Apply pattern-based adjustments
        for _category, config in self.accuracy_indicators.items():
            pattern_matches = sum(1 for pattern in config["patterns"] 
                                if re.search(pattern, content_lower))
            
            if pattern_matches > 0:
                if "score_bonus" in config:
                    base_accuracy += config["score_bonus"] * min(pattern_matches / 3, 1.0)
                elif "score_penalty" in config:
                    base_accuracy -= config["score_penalty"] * min(pattern_matches / 3, 1.0)
        
        # Source type reliability
        source_reliability = self._assess_source_reliability(knowledge.source_type)
        base_accuracy = (base_accuracy + source_reliability) / 2
        
        # Extraction type reliability
        extraction_reliability = self._assess_extraction_reliability(knowledge.extraction_type)
        base_accuracy = (base_accuracy + extraction_reliability) / 2
        
        return max(0.0, min(1.0, base_accuracy))
    
    def _assess_source_reliability(self, source_type: str) -> float:
        """Assess reliability based on source type"""
        reliability_scores = {
            "task_completion": 0.9,
            "agent_interaction": 0.8,
            "manual": 0.7,
            "automated": 0.75,
            "decision_analysis": 0.85,
            "failure_analysis": 0.8,
            "observation": 0.7,
            "unknown": 0.5
        }
        return reliability_scores.get(source_type, 0.6)
    
    def _assess_extraction_reliability(self, extraction_type: str) -> float:
        """Assess reliability based on extraction type"""
        reliability_scores = {
            "task_outcome": 0.9,
            "success_pattern": 0.85,
            "failure_analysis": 0.8,
            "decision_pattern": 0.8,
            "agent_interaction": 0.75,
            "coordination_insight": 0.7,
            "general_pattern": 0.6,
            "unknown": 0.5
        }
        return reliability_scores.get(extraction_type, 0.6)

class CompletenessAssessor:
    """Assesses knowledge completeness and coverage"""
    
    def __init__(self):
        self.completeness_factors = {
            "context": ["when", "where", "why", "how", "what"],
            "details": ["specific", "exact", "precise", "detailed"],
            "examples": ["example", "instance", "case", "sample"],
            "outcomes": ["result", "outcome", "effect", "consequence"],
            "conditions": ["if", "when", "unless", "provided", "given"]
        }
    
    def assess_completeness(self, knowledge: KnowledgeEntry) -> float:
        """Assess knowledge completeness and coverage"""
        content_lower = knowledge.content.lower()
        completeness_score = 0.0
        
        # Length-based completeness
        content_length = len(knowledge.content)
        length_score = min(1.0, content_length / 200.0)  # Optimal around 200 chars
        completeness_score += length_score * 0.3
        
        # Word count completeness
        word_count = len(knowledge.content.split())
        word_score = min(1.0, word_count / 30.0)  # Optimal around 30 words
        completeness_score += word_score * 0.2
        
        # Factor-based completeness
        for _factor, keywords in self.completeness_factors.items():
            factor_score = 0.0
            for keyword in keywords:
                if keyword in content_lower:
                    factor_score += 1.0
            
            # Normalize by number of keywords
            factor_score = min(1.0, factor_score / len(keywords))
            completeness_score += factor_score * 0.1  # Each factor contributes 10%
        
        # Sentence structure completeness
        sentences = re.split(r'[.!?]+', knowledge.content)
        complete_sentences = [s for s in sentences if len(s.strip()) > 10 and ' ' in s.strip()]
        structure_score = min(1.0, len(complete_sentences) / 3.0)  # Optimal around 3 sentences
        completeness_score += structure_score * 0.2
        
        return max(0.0, min(1.0, completeness_score))

class TimelinessAssessor:
    """Assesses knowledge timeliness and relevance over time"""
    
    def __init__(self):
        self.timeliness_decay = {
            "immediate": 1.0,      # < 1 hour
            "recent": 0.9,         # 1 hour - 1 day
            "current": 0.8,        # 1 day - 1 week
            "moderate": 0.6,       # 1 week - 1 month
            "old": 0.4,           # 1 month - 6 months
            "outdated": 0.2        # > 6 months
        }
    
    def assess_timeliness(self, knowledge: KnowledgeEntry) -> float:
        """Assess knowledge timeliness"""
        age_seconds = time.time() - knowledge.created_at
        age_hours = age_seconds / 3600
        age_days = age_hours / 24
        age_weeks = age_days / 7
        age_months = age_days / 30
        
        # Determine timeliness category
        if age_hours < 1:
            timeliness_score = self.timeliness_decay["immediate"]
        elif age_hours < 24:
            timeliness_score = self.timeliness_decay["recent"]
        elif age_days < 7:
            timeliness_score = self.timeliness_decay["current"]
        elif age_weeks < 4:
            timeliness_score = self.timeliness_decay["moderate"]
        elif age_months < 6:
            timeliness_score = self.timeliness_decay["old"]
        else:
            timeliness_score = self.timeliness_decay["outdated"]
        
        # Adjust based on content type
        content_lower = knowledge.content.lower()
        
        # Time-sensitive content decays faster
        if any(word in content_lower for word in ["current", "now", "today", "recent"]):
            timeliness_score *= 0.8  # Faster decay for time-sensitive content
        
        # Timeless principles decay slower
        if any(word in content_lower for word in ["principle", "rule", "always", "fundamental"]):
            timeliness_score = min(1.0, timeliness_score * 1.2)  # Slower decay for principles
        
        return max(0.0, min(1.0, timeliness_score))

class SourceReliabilityAssessor:
    """Assesses source reliability and credibility"""
    
    def __init__(self):
        self.source_reliability_matrix = {
            "automated_system": {
                "task_outcome": 0.95,
                "success_pattern": 0.9,
                "failure_analysis": 0.9,
                "default": 0.85
            },
            "agent_interaction": {
                "coordination_insight": 0.85,
                "decision_pattern": 0.8,
                "general_pattern": 0.75,
                "default": 0.7
            },
            "manual": {
                "expert_input": 0.9,
                "observation": 0.7,
                "assumption": 0.5,
                "default": 0.6
            },
            "derived": {
                "analysis": 0.8,
                "synthesis": 0.75,
                "inference": 0.6,
                "default": 0.65
            }
        }
    
    def assess_source_reliability(self, knowledge: KnowledgeEntry) -> float:
        """Assess source reliability and credibility"""
        source_type = knowledge.source_type
        extraction_type = knowledge.extraction_type
        
        # Get reliability matrix for source type
        if source_type in self.source_reliability_matrix:
            reliability_matrix = self.source_reliability_matrix[source_type]
            base_reliability = reliability_matrix.get(extraction_type, reliability_matrix["default"])
        else:
            base_reliability = 0.6  # Default for unknown source types
        
        # Adjust based on metadata
        metadata = knowledge.metadata
        
        # Confidence from original source
        if "source_confidence" in metadata:
            source_confidence = metadata["source_confidence"]
            base_reliability = (base_reliability + source_confidence) / 2
        
        # Success rate of source agent
        if "agent_success_rate" in metadata:
            agent_success_rate = metadata["agent_success_rate"]
            base_reliability = (base_reliability + agent_success_rate) / 2
        
        # Quality score from extraction
        if "quality_score" in metadata:
            quality_score = metadata["quality_score"]
            base_reliability = (base_reliability + quality_score) / 2
        
        return max(0.0, min(1.0, base_reliability))

class VerificationLevelAssessor:
    """Assesses verification level and validation status"""
    
    def __init__(self):
        self.verification_indicators = {
            "verified": 1.0,
            "tested": 0.9,
            "validated": 0.9,
            "confirmed": 0.85,
            "observed": 0.8,
            "documented": 0.75,
            "reported": 0.7,
            "claimed": 0.5,
            "assumed": 0.3,
            "suspected": 0.2
        }
    
    def assess_verification_level(self, knowledge: KnowledgeEntry) -> float:
        """Assess verification level and validation status"""
        content_lower = knowledge.content.lower()
        verification_scores = []
        
        # Look for verification indicators
        for indicator, score in self.verification_indicators.items():
            if indicator in content_lower:
                verification_scores.append(score)
        
        # Use highest verification level found
        if verification_scores:
            base_verification = max(verification_scores)
        else:
            base_verification = 0.5  # Default medium verification
        
        # Adjust based on source type
        source_type = knowledge.source_type
        if source_type in ["task_completion", "automated_system"]:
            base_verification = min(1.0, base_verification * 1.2)  # Higher verification for automated sources
        elif source_type in ["manual", "observation"]:
            base_verification *= 0.8  # Lower verification for manual sources
        
        # Consider extraction confidence
        base_verification = (base_verification + knowledge.confidence) / 2
        
        return max(0.0, min(1.0, base_verification))

class KnowledgeQualityScorer:
    """Main scorer for knowledge quality assessment"""
    
    def __init__(self):
        self.quality_metrics = {
            QualityMetric.ACCURACY: AccuracyAssessor(),
            QualityMetric.COMPLETENESS: CompletenessAssessor(),
            QualityMetric.TIMELINESS: TimelinessAssessor(),
            QualityMetric.SOURCE_RELIABILITY: SourceReliabilityAssessor(),
            QualityMetric.VERIFICATION_LEVEL: VerificationLevelAssessor()
        }
        
        # Weights for overall score calculation
        self.metric_weights = {
            QualityMetric.ACCURACY: 0.25,
            QualityMetric.COMPLETENESS: 0.20,
            QualityMetric.TIMELINESS: 0.15,
            QualityMetric.SOURCE_RELIABILITY: 0.20,
            QualityMetric.VERIFICATION_LEVEL: 0.20
        }
        
    async def score_knowledge_quality(
        self, 
        knowledge: KnowledgeEntry
    ) -> QualityScore:
        """Calculate comprehensive quality score for knowledge"""
        
        metric_scores = {}
        details = {}
        recommendations = []
        
        # Calculate each metric score
        for metric, assessor in self.quality_metrics.items():
            try:
                if metric == QualityMetric.ACCURACY:
                    score = assessor.assess_accuracy(knowledge)
                elif metric == QualityMetric.COMPLETENESS:
                    score = assessor.assess_completeness(knowledge)
                elif metric == QualityMetric.TIMELINESS:
                    score = assessor.assess_timeliness(knowledge)
                elif metric == QualityMetric.SOURCE_RELIABILITY:
                    score = assessor.assess_source_reliability(knowledge)
                elif metric == QualityMetric.VERIFICATION_LEVEL:
                    score = assessor.assess_verification_level(knowledge)
                else:
                    score = 0.5  # Default score for unknown metrics
                
                metric_scores[metric] = score
                
                # Generate recommendations for low scores
                if score < 0.6:
                    recommendations.extend(self._generate_improvement_recommendations(metric, score))
                
            except (ValueError, KeyError, AttributeError) as e:
                logger.error("Error calculating %s: %s", metric.value, str(e))
                metric_scores[metric] = 0.5  # Default score on error
        
        # Calculate overall score
        overall_score = self._calculate_overall_quality(metric_scores)
        
        # Generate quality details
        details = {
            "content_length": len(knowledge.content),
            "word_count": len(knowledge.content.split()),
            "source_type": knowledge.source_type,
            "extraction_type": knowledge.extraction_type,
            "age_hours": (time.time() - knowledge.created_at) / 3600,
            "original_confidence": knowledge.confidence
        }
        
        # Calculate confidence in quality assessment
        assessment_confidence = self._calculate_assessment_confidence(metric_scores, knowledge)
        
        return QualityScore(
            overall=overall_score,
            metrics=metric_scores,
            details=details,
            recommendations=recommendations,
            confidence=assessment_confidence
        )
    
    def _assess_accuracy(self, knowledge: KnowledgeEntry) -> float:
        """Assess knowledge accuracy based on verification and validation"""
        return self.quality_metrics[QualityMetric.ACCURACY].assess_accuracy(knowledge)
    
    def _assess_completeness(self, knowledge: KnowledgeEntry) -> float:
        """Assess knowledge completeness and coverage"""
        return self.quality_metrics[QualityMetric.COMPLETENESS].assess_completeness(knowledge)
    
    def _assess_timeliness(self, knowledge: KnowledgeEntry) -> float:
        """Assess knowledge timeliness"""
        return self.quality_metrics[QualityMetric.TIMELINESS].assess_timeliness(knowledge)
    
    def _assess_source_reliability(self, knowledge: KnowledgeEntry) -> float:
        """Assess source reliability"""
        return self.quality_metrics[QualityMetric.SOURCE_RELIABILITY].assess_source_reliability(knowledge)
    
    def _assess_verification(self, knowledge: KnowledgeEntry) -> float:
        """Assess verification level"""
        return self.quality_metrics[QualityMetric.VERIFICATION_LEVEL].assess_verification_level(knowledge)
    
    def _calculate_overall_quality(self, metrics: Dict[QualityMetric, float]) -> float:
        """Calculate weighted overall quality score"""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric, score in metrics.items():
            weight = self.metric_weights.get(metric, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _generate_improvement_recommendations(self, metric: QualityMetric, score: float) -> List[str]:
        """Generate recommendations for improving quality scores"""
        recommendations = []
        
        if metric == QualityMetric.ACCURACY:
            if score < 0.5:
                recommendations.extend([
                    "Add verification or validation information",
                    "Include sources or references",
                    "Provide evidence for claims"
                ])
            else:
                recommendations.append("Consider adding more verification details")
        
        elif metric == QualityMetric.COMPLETENESS:
            if score < 0.5:
                recommendations.extend([
                    "Add more detailed explanation",
                    "Include context and conditions",
                    "Provide examples or use cases"
                ])
            else:
                recommendations.append("Consider adding more context")
        
        elif metric == QualityMetric.TIMELINESS:
            if score < 0.5:
                recommendations.extend([
                    "Update with current information",
                    "Mark as historical if still relevant",
                    "Consider refreshing the knowledge"
                ])
            else:
                recommendations.append("Consider updating if needed")
        
        elif metric == QualityMetric.SOURCE_RELIABILITY:
            if score < 0.5:
                recommendations.extend([
                    "Verify information from additional sources",
                    "Add source credibility information",
                    "Cross-reference with trusted sources"
                ])
            else:
                recommendations.append("Consider source verification")
        
        elif metric == QualityMetric.VERIFICATION_LEVEL:
            if score < 0.5:
                recommendations.extend([
                    "Add verification or testing information",
                    "Include validation results",
                    "Provide confirmation details"
                ])
            else:
                recommendations.append("Consider additional verification")
        
        return recommendations
    
    def _calculate_assessment_confidence(
        self, 
        metric_scores: Dict[QualityMetric, float], 
        knowledge: KnowledgeEntry
    ) -> float:
        """Calculate confidence in the quality assessment"""
        confidence_factors = []
        
        # Content length confidence
        content_length = len(knowledge.content)
        length_confidence = min(1.0, content_length / 100.0)
        confidence_factors.append(length_confidence)
        
        # Metric consistency confidence
        if metric_scores:
            scores = list(metric_scores.values())
            score_variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
            consistency_confidence = max(0.0, 1.0 - score_variance)
            confidence_factors.append(consistency_confidence)
        
        # Original confidence
        confidence_factors.append(knowledge.confidence)
        
        # Source type confidence
        source_confidence = self._get_source_type_confidence(knowledge.source_type)
        confidence_factors.append(source_confidence)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _get_source_type_confidence(self, source_type: str) -> float:
        """Get confidence based on source type"""
        source_confidences = {
            "task_completion": 0.9,
            "automated_system": 0.85,
            "agent_interaction": 0.8,
            "decision_analysis": 0.8,
            "failure_analysis": 0.75,
            "manual": 0.7,
            "observation": 0.65,
            "unknown": 0.5
        }
        return source_confidences.get(source_type, 0.6)