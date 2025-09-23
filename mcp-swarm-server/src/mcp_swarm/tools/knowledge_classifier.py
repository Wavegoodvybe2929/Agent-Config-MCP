"""
Knowledge Classification System for Collective Knowledge Management

Implements domain classification, complexity analysis, and relevance scoring for the MCP 
Swarm Intelligence Server collective knowledge contribution system.

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

class DomainCategory(Enum):
    """Knowledge domain categories"""
    AGENT_COORDINATION = "agent_coordination"
    TASK_MANAGEMENT = "task_management"
    DECISION_MAKING = "decision_making"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    ERROR_HANDLING = "error_handling"
    RESOURCE_ALLOCATION = "resource_allocation"
    COMMUNICATION = "communication"
    LEARNING_PATTERNS = "learning_patterns"
    SYSTEM_ARCHITECTURE = "system_architecture"
    USER_INTERACTION = "user_interaction"
    GENERAL = "general"

class ComplexityLevel(Enum):
    """Knowledge complexity levels"""
    SIMPLE = "simple"           # Basic facts or single-step procedures
    MODERATE = "moderate"       # Multi-step processes with some complexity
    COMPLEX = "complex"         # Multi-faceted with dependencies and nuances
    EXPERT = "expert"           # Requires deep domain knowledge and experience

@dataclass
class KnowledgeClassification:
    """Classification result for knowledge entry"""
    domains: List[DomainCategory]
    primary_domain: DomainCategory
    complexity_level: ComplexityLevel
    relevance_score: float
    confidence: float
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert classification to dictionary"""
        return {
            "domains": [d.value for d in self.domains],
            "primary_domain": self.primary_domain.value,
            "complexity_level": self.complexity_level.value,
            "relevance_score": self.relevance_score,
            "confidence": self.confidence,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at
        }

@dataclass
class KnowledgeEntry:
    """Knowledge entry for classification"""
    content: str
    source_type: str
    extraction_type: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

class DomainClassifier:
    """Classifies knowledge into domain categories"""
    
    def __init__(self):
        self.domain_keywords = {
            DomainCategory.AGENT_COORDINATION: [
                "agent", "coordination", "swarm", "collective", "collaboration", 
                "synchronization", "handoff", "assignment", "delegation"
            ],
            DomainCategory.TASK_MANAGEMENT: [
                "task", "workflow", "execution", "scheduling", "priority", 
                "deadline", "completion", "management", "tracking"
            ],
            DomainCategory.DECISION_MAKING: [
                "decision", "choice", "selection", "consensus", "voting", 
                "judgment", "evaluation", "criteria", "option"
            ],
            DomainCategory.PERFORMANCE_OPTIMIZATION: [
                "performance", "optimization", "efficiency", "speed", "throughput",
                "latency", "bottleneck", "scalability", "resource usage"
            ],
            DomainCategory.ERROR_HANDLING: [
                "error", "exception", "failure", "fault", "recovery", "retry", 
                "resilience", "robustness", "tolerance"
            ],
            DomainCategory.RESOURCE_ALLOCATION: [
                "resource", "allocation", "capacity", "load", "distribution", 
                "availability", "utilization", "sharing", "contention"
            ],
            DomainCategory.COMMUNICATION: [
                "communication", "message", "protocol", "interface", "API", 
                "exchange", "transmission", "channel", "network"
            ],
            DomainCategory.LEARNING_PATTERNS: [
                "learning", "pattern", "adaptation", "improvement", "evolution", 
                "training", "knowledge", "experience", "insight"
            ],
            DomainCategory.SYSTEM_ARCHITECTURE: [
                "architecture", "design", "structure", "component", "module", 
                "framework", "infrastructure", "topology", "hierarchy"
            ],
            DomainCategory.USER_INTERACTION: [
                "user", "interface", "interaction", "usability", "experience", 
                "feedback", "request", "response", "presentation"
            ]
        }
    
    def classify_domains(self, knowledge: KnowledgeEntry) -> List[DomainCategory]:
        """Classify knowledge into domain categories"""
        content_lower = knowledge.content.lower()
        domain_scores = {}
        
        # Calculate domain scores based on keyword matches
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                # Exact word matches
                if re.search(rf'\b{re.escape(keyword)}\b', content_lower):
                    score += 2
                # Partial matches
                elif keyword in content_lower:
                    score += 1
            
            if score > 0:
                domain_scores[domain] = score
        
        # Sort domains by score and return top matches
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return domains with significant scores
        threshold = max(2, sorted_domains[0][1] * 0.5) if sorted_domains else 2
        relevant_domains = [domain for domain, score in sorted_domains if score >= threshold]
        
        # Always include at least one domain
        if not relevant_domains and sorted_domains:
            relevant_domains = [sorted_domains[0][0]]
        elif not relevant_domains:
            relevant_domains = [DomainCategory.GENERAL]
        
        return relevant_domains[:3]  # Limit to top 3 domains
    
    def get_primary_domain(self, domains: List[DomainCategory], knowledge: KnowledgeEntry) -> DomainCategory:
        """Get the primary domain for the knowledge"""
        if not domains:
            return DomainCategory.GENERAL
        
        # Use source type to inform primary domain selection
        source_type = knowledge.source_type.lower()
        extraction_type = knowledge.extraction_type.lower()
        
        # Task-related sources favor task management
        if "task" in source_type or "task" in extraction_type:
            if DomainCategory.TASK_MANAGEMENT in domains:
                return DomainCategory.TASK_MANAGEMENT
        
        # Agent interaction sources favor coordination
        if "agent" in source_type or "interaction" in source_type:
            if DomainCategory.AGENT_COORDINATION in domains:
                return DomainCategory.AGENT_COORDINATION
        
        # Failure sources favor error handling
        if "failure" in extraction_type or "error" in extraction_type:
            if DomainCategory.ERROR_HANDLING in domains:
                return DomainCategory.ERROR_HANDLING
        
        # Default to first domain
        return domains[0]

class ComplexityAnalyzer:
    """Analyzes knowledge complexity"""
    
    def __init__(self):
        self.complexity_indicators = {
            ComplexityLevel.SIMPLE: {
                "max_sentences": 2,
                "max_clauses": 3,
                "simple_patterns": [r"is", r"are", r"was", r"were", r"do", r"does"],
                "score_range": (0.0, 0.3)
            },
            ComplexityLevel.MODERATE: {
                "max_sentences": 5,
                "max_clauses": 8,
                "moderate_patterns": [r"because", r"therefore", r"however", r"although"],
                "score_range": (0.3, 0.6)
            },
            ComplexityLevel.COMPLEX: {
                "max_sentences": 10,
                "max_clauses": 15,
                "complex_patterns": [r"consequently", r"furthermore", r"nevertheless", r"meanwhile"],
                "score_range": (0.6, 0.85)
            },
            ComplexityLevel.EXPERT: {
                "max_sentences": float('inf'),
                "max_clauses": float('inf'),
                "expert_patterns": [r"paradigm", r"methodology", r"algorithm", r"optimization"],
                "score_range": (0.85, 1.0)
            }
        }
    
    def analyze_complexity(self, knowledge: KnowledgeEntry) -> ComplexityLevel:
        """Analyze knowledge complexity level"""
        content = knowledge.content
        
        # Count sentences and clauses
        sentences = len(re.findall(r'[.!?]+', content))
        clauses = len(re.findall(r'[,;]+', content)) + sentences
        
        # Count technical terms
        technical_terms = len(re.findall(r'\b(?:algorithm|optimization|protocol|architecture|framework|methodology)\b', content.lower()))
        
        # Calculate complexity score
        complexity_score = 0.0
        
        # Length-based scoring
        complexity_score += min(sentences / 10.0, 0.3)
        complexity_score += min(clauses / 20.0, 0.3)
        
        # Technical term scoring
        complexity_score += min(technical_terms / 5.0, 0.2)
        
        # Pattern-based scoring
        for _level, indicators in self.complexity_indicators.items():
            pattern_matches = 0
            if "simple_patterns" in indicators:
                patterns = indicators["simple_patterns"]
            elif "moderate_patterns" in indicators:
                patterns = indicators["moderate_patterns"]
            elif "complex_patterns" in indicators:
                patterns = indicators["complex_patterns"]
            elif "expert_patterns" in indicators:
                patterns = indicators["expert_patterns"]
            else:
                continue
            
            for pattern in patterns:
                if re.search(pattern, content.lower()):
                    pattern_matches += 1
            
            if pattern_matches > 0:
                min_score, max_score = indicators["score_range"]
                complexity_score += (max_score - min_score) * min(pattern_matches / len(patterns), 1.0)
        
        # Determine complexity level
        if complexity_score <= 0.3:
            return ComplexityLevel.SIMPLE
        elif complexity_score <= 0.6:
            return ComplexityLevel.MODERATE
        elif complexity_score <= 0.85:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.EXPERT

class RelevanceScorer:
    """Scores knowledge relevance"""
    
    def __init__(self):
        self.relevance_factors = {
            "recency": 0.25,      # How recent the knowledge is
            "specificity": 0.30,  # How specific vs general
            "actionability": 0.25, # How actionable the knowledge is
            "uniqueness": 0.20    # How unique/novel the knowledge is
        }
        
        self.actionable_keywords = [
            "should", "must", "need", "implement", "use", "apply", "configure",
            "avoid", "prevent", "optimize", "improve", "ensure", "verify"
        ]
    
    def calculate_relevance_score(self, knowledge: KnowledgeEntry) -> float:
        """Calculate relevance score for knowledge"""
        scores = {}
        
        # Recency score (newer knowledge is more relevant)
        age_hours = (time.time() - knowledge.created_at) / 3600
        scores["recency"] = max(0.0, 1.0 - (age_hours / (24 * 7)))  # Decay over a week
        
        # Specificity score (more specific is more relevant)
        content_length = len(knowledge.content)
        specific_terms = len(re.findall(r'\b(?:specific|exactly|precisely|particular)\b', knowledge.content.lower()))
        scores["specificity"] = min(1.0, (content_length / 200.0) + (specific_terms * 0.2))
        
        # Actionability score (actionable knowledge is more relevant)
        actionable_count = sum(1 for keyword in self.actionable_keywords 
                             if keyword in knowledge.content.lower())
        scores["actionability"] = min(1.0, actionable_count / 3.0)
        
        # Uniqueness score (based on source confidence and extraction type)
        uniqueness_base = knowledge.confidence
        if knowledge.extraction_type in ["failure_analysis", "decision_pattern"]:
            uniqueness_base += 0.2  # These types tend to be more unique
        scores["uniqueness"] = min(1.0, uniqueness_base)
        
        # Calculate weighted score
        relevance_score = sum(
            scores[factor] * weight 
            for factor, weight in self.relevance_factors.items()
        )
        
        return min(1.0, max(0.0, relevance_score))

class KnowledgeClassifier:
    """Main classifier for knowledge entries"""
    
    def __init__(self):
        self.domain_classifier = DomainClassifier()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.relevance_scorer = RelevanceScorer()
        
    async def classify_knowledge(
        self, 
        knowledge: KnowledgeEntry
    ) -> KnowledgeClassification:
        """Classify knowledge by domain, complexity, and relevance"""
        
        # Classify domains
        domains = self.domain_classifier.classify_domains(knowledge)
        primary_domain = self.domain_classifier.get_primary_domain(domains, knowledge)
        
        # Analyze complexity
        complexity_level = self.complexity_analyzer.analyze_complexity(knowledge)
        
        # Calculate relevance score
        relevance_score = self.relevance_scorer.calculate_relevance_score(knowledge)
        
        # Generate tags
        tags = self._generate_tags(knowledge, domains, complexity_level)
        
        # Calculate classification confidence
        confidence = self._calculate_classification_confidence(
            knowledge, domains, complexity_level, relevance_score
        )
        
        # Generate metadata
        metadata = self._generate_classification_metadata(
            knowledge, domains, complexity_level, relevance_score
        )
        
        return KnowledgeClassification(
            domains=domains,
            primary_domain=primary_domain,
            complexity_level=complexity_level,
            relevance_score=relevance_score,
            confidence=confidence,
            tags=tags,
            metadata=metadata
        )
    
    def _classify_domain(self, knowledge: KnowledgeEntry) -> List[str]:
        """Classify knowledge into domain categories"""
        domains = self.domain_classifier.classify_domains(knowledge)
        return [domain.value for domain in domains]
    
    def _assess_complexity(self, knowledge: KnowledgeEntry) -> ComplexityLevel:
        """Assess knowledge complexity level"""
        return self.complexity_analyzer.analyze_complexity(knowledge)
    
    def _calculate_relevance_score(self, knowledge: KnowledgeEntry) -> float:
        """Calculate relevance score for knowledge"""
        return self.relevance_scorer.calculate_relevance_score(knowledge)
    
    def _generate_classification_metadata(
        self, 
        knowledge: KnowledgeEntry,
        domains: List[DomainCategory],
        complexity_level: ComplexityLevel,
        relevance_score: float
    ) -> Dict[str, Any]:
        """Generate metadata for classification"""
        return {
            "content_length": len(knowledge.content),
            "word_count": len(knowledge.content.split()),
            "sentence_count": len(re.findall(r'[.!?]+', knowledge.content)),
            "source_type": knowledge.source_type,
            "extraction_type": knowledge.extraction_type,
            "domain_count": len(domains),
            "complexity_score": self._complexity_to_score(complexity_level),
            "relevance_score": relevance_score,
            "classification_timestamp": time.time()
        }
    
    def _generate_tags(
        self, 
        knowledge: KnowledgeEntry,
        domains: List[DomainCategory],
        complexity_level: ComplexityLevel
    ) -> List[str]:
        """Generate tags for knowledge entry"""
        tags = []
        
        # Add domain-based tags
        tags.extend([domain.value for domain in domains])
        
        # Add complexity tag
        tags.append(f"complexity_{complexity_level.value}")
        
        # Add source type tag
        tags.append(f"source_{knowledge.source_type}")
        
        # Add extraction type tag
        tags.append(f"extraction_{knowledge.extraction_type}")
        
        # Add content-based tags
        content_lower = knowledge.content.lower()
        if any(word in content_lower for word in ["success", "achieve", "complete"]):
            tags.append("success_related")
        if any(word in content_lower for word in ["fail", "error", "problem"]):
            tags.append("failure_related")
        if any(word in content_lower for word in ["optimize", "improve", "enhance"]):
            tags.append("improvement_related")
        
        return list(set(tags))  # Remove duplicates
    
    def _calculate_classification_confidence(
        self,
        knowledge: KnowledgeEntry,
        domains: List[DomainCategory],
        complexity_level: ComplexityLevel,
        relevance_score: float
    ) -> float:
        """Calculate confidence in the classification"""
        confidence_factors = []
        
        # Domain classification confidence
        if len(domains) >= 1:
            confidence_factors.append(0.8)  # Good domain classification
        else:
            confidence_factors.append(0.4)  # Uncertain domain classification
        
        # Content quality confidence
        content_quality = min(1.0, len(knowledge.content) / 100.0)
        confidence_factors.append(content_quality)
        
        # Source confidence
        confidence_factors.append(knowledge.confidence)
        
        # Relevance confidence
        confidence_factors.append(relevance_score)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _complexity_to_score(self, complexity_level: ComplexityLevel) -> float:
        """Convert complexity level to numeric score"""
        mapping = {
            ComplexityLevel.SIMPLE: 0.25,
            ComplexityLevel.MODERATE: 0.5,
            ComplexityLevel.COMPLEX: 0.75,
            ComplexityLevel.EXPERT: 1.0
        }
        return mapping.get(complexity_level, 0.5)