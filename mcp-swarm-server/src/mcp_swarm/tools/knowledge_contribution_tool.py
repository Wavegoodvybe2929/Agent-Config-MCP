"""
Knowledge Contribution Tool for MCP Swarm Intelligence Server

Implements the complete MCP tool interface for collective knowledge contribution including 
extraction, classification, validation, quality scoring, and storage with proper error handling.

According to orchestrator routing:
- Primary: hive_mind_specialist.md
- Secondary: code.md, memory_management_specialist.md
"""
import logging
import time
import sqlite3
from typing import Dict, Any, Optional, List

# Import our components
from .knowledge_extraction import (
    KnowledgeExtractionEngine, 
    KnowledgeCandidate, 
    AgentInteraction, 
    CompletedTask
)
from .knowledge_classifier import KnowledgeClassifier, KnowledgeEntry
from .knowledge_validator import KnowledgeValidationSystem
from .knowledge_quality import KnowledgeQualityScorer

logger = logging.getLogger(__name__)

class KnowledgeDatabase:
    """Simple knowledge database interface for storing knowledge entries"""
    
    def __init__(self, db_path: str = "src/data/memory.db"):
        self.db_path = db_path
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure the database and knowledge table exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_entries (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        source TEXT,
                        confidence REAL DEFAULT 0.5,
                        embedding BLOB,
                        tags TEXT,
                        category TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """)
                conn.commit()
        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Error creating knowledge database: %s", str(e))
    
    async def store_knowledge_entry(
        self, 
        knowledge_candidate: KnowledgeCandidate,
        classification: Any,
        quality_score: Any
    ) -> 'StoredKnowledgeEntry':
        """Store a knowledge entry in the database"""
        try:
            knowledge_id = f"k_{int(time.time() * 1000)}_{hash(knowledge_candidate.content) % 10000}"
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO knowledge_entries 
                    (id, content, source, confidence, tags, category, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    knowledge_id,
                    knowledge_candidate.content,
                    knowledge_candidate.source_type,
                    quality_score.overall,
                    ",".join(classification.tags),
                    classification.primary_domain.value,
                    knowledge_candidate.created_at
                ))
                conn.commit()
            
            return StoredKnowledgeEntry(
                entry_id=knowledge_id,
                content=knowledge_candidate.content,
                source=knowledge_candidate.source_type,
                confidence=quality_score.overall,
                tags=classification.tags,
                category=classification.primary_domain.value,
                created_at=knowledge_candidate.created_at
            )
            
        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Error storing knowledge entry: %s", str(e))
            raise

class StoredKnowledgeEntry:
    """Represents a stored knowledge entry"""
    
    def __init__(self, entry_id: str, content: str, source: str, confidence: float, 
                 tags: List[str], category: str, created_at: float):
        self.id = entry_id
        self.content = content
        self.source = source
        self.confidence = confidence
        self.tags = tags
        self.category = category
        self.created_at = created_at

def generate_improvement_suggestions(validation_result, quality_score) -> List[str]:
    """Generate improvement suggestions based on validation and quality results"""
    suggestions = []
    
    # Add validation-based suggestions
    if validation_result.issues:
        for issue in validation_result.issues:
            suggestions.extend(issue.suggestions)
    
    # Add quality-based suggestions
    suggestions.extend(quality_score.recommendations)
    
    # Add conflict-based suggestions
    if validation_result.conflicts:
        suggestions.append("Review for conflicts with existing knowledge")
        suggestions.append("Consider merging or updating existing knowledge")
    
    # Remove duplicates and limit
    unique_suggestions = list(set(suggestions))
    return unique_suggestions[:5]  # Limit to top 5 suggestions

async def store_knowledge_entry(
    knowledge_candidate: KnowledgeCandidate,
    classification: Any,
    quality_score: Any
) -> StoredKnowledgeEntry:
    """Store a knowledge entry in the database"""
    knowledge_db = KnowledgeDatabase()
    return await knowledge_db.store_knowledge_entry(knowledge_candidate, classification, quality_score)

# MCP Tool Implementation
async def knowledge_contribution_tool(
    source_type: str,
    content: str,
    domain: Optional[str] = None,
    confidence: float = 0.8,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    MCP tool for contributing knowledge to the hive mind.
    
    Args:
        source_type: Type of knowledge source (task/interaction/manual)
        content: Knowledge content to contribute
        domain: Domain classification (optional)
        confidence: Contributor confidence level
        metadata: Additional metadata for the knowledge
        
    Returns:
        Knowledge contribution result with quality assessment
    """
    try:
        # Initialize components
        extraction_engine = KnowledgeExtractionEngine()
        classifier = KnowledgeClassifier()
        knowledge_db = KnowledgeDatabase()
        validator = KnowledgeValidationSystem(knowledge_db)
        quality_scorer = KnowledgeQualityScorer()
        
        # Extract knowledge candidates from content
        if source_type == "manual":
            from .knowledge_extraction import ExtractionType
            knowledge_candidate = KnowledgeCandidate(
                content=content,
                source_type=source_type,
                extraction_type=ExtractionType.COORDINATION_INSIGHT,  # Use valid enum value
                confidence=confidence,
                metadata=metadata or {}
            )
        elif source_type == "task":
            # Create a mock completed task for extraction
            mock_task = CompletedTask(
                task_id=metadata.get("task_id", "unknown") if metadata else "unknown",
                agent_ids=metadata.get("agent_ids", []) if metadata else [],
                task_type=metadata.get("task_type", "unknown") if metadata else "unknown",
                outcome=content,
                success=metadata.get("success", True) if metadata else True,
                duration=metadata.get("duration", 0.0) if metadata else 0.0,
                quality_score=confidence,
                metadata=metadata or {}
            )
            candidates = await extraction_engine.extract_knowledge_from_task(mock_task)
            knowledge_candidate = candidates[0] if candidates else None
        elif source_type == "interaction":
            # Create a mock agent interaction for extraction
            mock_interaction = AgentInteraction(
                agent_id=metadata.get("agent_id", "unknown") if metadata else "unknown",
                task_id=metadata.get("task_id", "unknown") if metadata else "unknown",
                interaction_type=metadata.get("interaction_type", "unknown") if metadata else "unknown",
                content=content,
                success=metadata.get("success", True) if metadata else True,
                duration=metadata.get("duration", 0.0) if metadata else 0.0,
                metadata=metadata or {}
            )
            candidates = await extraction_engine.extract_knowledge_from_interaction(mock_interaction)
            knowledge_candidate = candidates[0] if candidates else None
        else:
            # Generic content extraction
            candidates = await extraction_engine.extract_knowledge_from_content(
                content, source_type, metadata
            )
            knowledge_candidate = candidates[0] if candidates else None
        
        if not knowledge_candidate:
            return {
                "status": "failed", 
                "reason": "No extractable knowledge found",
                "suggestions": [
                    "Provide more detailed content",
                    "Include specific outcomes or insights",
                    "Add context about what was learned"
                ]
            }
        
        # Convert to KnowledgeEntry for classification
        knowledge_entry = KnowledgeEntry(
            content=knowledge_candidate.content,
            source_type=knowledge_candidate.source_type,
            extraction_type=knowledge_candidate.extraction_type.value,  # Convert enum to string
            confidence=knowledge_candidate.confidence,
            metadata=knowledge_candidate.metadata,
            created_at=knowledge_candidate.created_at
        )
        
        # Classify the knowledge
        classification = await classifier.classify_knowledge(knowledge_entry)
        
        # Override domain if specified
        if domain:
            # Find matching domain category
            from .knowledge_classifier import DomainCategory
            for category in DomainCategory:
                if category.value == domain:
                    classification.primary_domain = category
                    if category not in classification.domains:
                        classification.domains.insert(0, category)
                    break
        
        # Validate consistency
        validation_result = await validator.validate_knowledge_consistency(knowledge_entry)
        
        # Score quality - create compatible entry for quality scorer
        from .knowledge_quality import KnowledgeEntry as QualityKnowledgeEntry
        quality_entry = QualityKnowledgeEntry(
            content=knowledge_entry.content,
            source_type=knowledge_entry.source_type,
            extraction_type=knowledge_entry.extraction_type,
            confidence=knowledge_entry.confidence,
            metadata=knowledge_entry.metadata,
            created_at=knowledge_entry.created_at,
            id=getattr(knowledge_entry, 'id', '')
        )
        quality_score = await quality_scorer.score_knowledge_quality(quality_entry)
        
        # Check if knowledge meets quality threshold
        quality_threshold = 0.6
        if validation_result.is_valid and quality_score.overall > quality_threshold:
            # Store the knowledge
            stored_entry = await store_knowledge_entry(
                knowledge_candidate, classification, quality_score
            )
            
            return {
                "status": "success",
                "knowledge_id": stored_entry.id,
                "classification": classification.to_dict(),
                "quality_score": quality_score.to_dict(),
                "validation_result": validation_result.to_dict(),
                "conflicts_detected": len(validation_result.conflicts),
                "stored_at": stored_entry.created_at,
                "summary": {
                    "content_length": len(knowledge_candidate.content),
                    "primary_domain": classification.primary_domain.value,
                    "complexity": classification.complexity_level.value,
                    "overall_quality": quality_score.overall,
                    "validation_confidence": validation_result.confidence
                }
            }
        else:
            # Knowledge rejected due to quality or validation issues
            improvement_suggestions = generate_improvement_suggestions(
                validation_result, quality_score
            )
            
            return {
                "status": "rejected",
                "reason": "Quality or validation threshold not met",
                "quality_score": quality_score.overall,
                "quality_threshold": quality_threshold,
                "validation_issues": [
                    {
                        "type": issue.issue_type,
                        "severity": issue.severity.value,
                        "description": issue.description
                    }
                    for issue in validation_result.issues
                ],
                "conflicts": [
                    {
                        "type": conflict.conflict_type.value,
                        "severity": conflict.severity.value,
                        "description": conflict.description
                    }
                    for conflict in validation_result.conflicts
                ],
                "improvement_suggestions": improvement_suggestions,
                "details": {
                    "classification": classification.to_dict(),
                    "validation_confidence": validation_result.confidence,
                    "quality_breakdown": quality_score.to_dict()
                }
            }
            
    except (ValueError, KeyError, AttributeError) as e:
        logger.error("Error in knowledge contribution tool: %s", str(e))
        return {
            "status": "error",
            "reason": f"Internal error during knowledge contribution: {str(e)}",
            "suggestions": [
                "Try again with different content",
                "Check content format and encoding",
                "Contact system administrator if error persists"
            ]
        }

# Additional MCP tools for knowledge management

async def knowledge_search_tool(
    query: str,
    domain: Optional[str] = None,
    max_results: int = 10,
    min_quality: float = 0.5
) -> Dict[str, Any]:
    """
    Search for knowledge in the hive mind database.
    
    Args:
        query: Search query
        domain: Optional domain filter
        max_results: Maximum number of results
        min_quality: Minimum quality threshold
        
    Returns:
        Search results with knowledge entries
    """
    try:
        knowledge_db = KnowledgeDatabase()
        
        with sqlite3.connect(knowledge_db.db_path) as conn:
            # Simple text search (could be enhanced with semantic search)
            sql = """
                SELECT id, content, source, confidence, tags, category, created_at
                FROM knowledge_entries 
                WHERE content LIKE ? AND confidence >= ?
            """
            params = [f"%{query}%", min_quality]
            
            if domain:
                sql += " AND category = ?"
                params.append(domain)
            
            sql += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
            params.append(max_results)
            
            cursor = conn.execute(sql, params)
            results = cursor.fetchall()
        
        knowledge_results = []
        for row in results:
            knowledge_results.append({
                "id": row[0],
                "content": row[1],
                "source": row[2],
                "confidence": row[3],
                "tags": row[4].split(",") if row[4] else [],
                "category": row[5],
                "created_at": row[6],
                "relevance_score": 1.0  # Would be calculated with semantic search
            })
        
        return {
            "status": "success",
            "query": query,
            "results_count": len(knowledge_results),
            "results": knowledge_results,
            "search_metadata": {
                "domain_filter": domain,
                "quality_threshold": min_quality,
                "max_results": max_results
            }
        }
        
    except (ValueError, KeyError, AttributeError, sqlite3.Error) as e:
        logger.error("Error in knowledge search tool: %s", str(e))
        return {
            "status": "error",
            "reason": f"Search error: {str(e)}",
            "results": []
        }

async def knowledge_stats_tool() -> Dict[str, Any]:
    """
    Get statistics about the knowledge base.
    
    Returns:
        Knowledge base statistics
    """
    try:
        knowledge_db = KnowledgeDatabase()
        
        with sqlite3.connect(knowledge_db.db_path) as conn:
            # Total knowledge entries
            total_count = conn.execute("SELECT COUNT(*) FROM knowledge_entries").fetchone()[0]
            
            # Average quality
            avg_quality = conn.execute("SELECT AVG(confidence) FROM knowledge_entries").fetchone()[0] or 0.0
            
            # Knowledge by category
            category_stats = conn.execute("""
                SELECT category, COUNT(*), AVG(confidence) 
                FROM knowledge_entries 
                GROUP BY category 
                ORDER BY COUNT(*) DESC
            """).fetchall()
            
            # Recent knowledge (last 24 hours)
            recent_count = conn.execute("""
                SELECT COUNT(*) FROM knowledge_entries 
                WHERE created_at > datetime('now', '-1 day')
            """).fetchone()[0]
            
            # Quality distribution
            quality_distribution = conn.execute("""
                SELECT 
                    CASE 
                        WHEN confidence >= 0.8 THEN 'high'
                        WHEN confidence >= 0.6 THEN 'medium'
                        ELSE 'low'
                    END as quality_level,
                    COUNT(*)
                FROM knowledge_entries
                GROUP BY quality_level
            """).fetchall()
        
        return {
            "status": "success",
            "statistics": {
                "total_knowledge_entries": total_count,
                "average_quality": round(avg_quality, 3),
                "recent_additions_24h": recent_count,
                "category_breakdown": [
                    {
                        "category": cat[0],
                        "count": cat[1],
                        "average_quality": round(cat[2], 3)
                    }
                    for cat in category_stats
                ],
                "quality_distribution": {
                    level[0]: level[1] for level in quality_distribution
                }
            },
            "generated_at": time.time()
        }
        
    except (ValueError, KeyError, AttributeError, sqlite3.Error) as e:
        logger.error("Error in knowledge stats tool: %s", str(e))
        return {
            "status": "error",
            "reason": f"Stats error: {str(e)}"
        }

# MCP Tool Registration
def register_knowledge_tools():
    """Register all knowledge management tools with MCP"""
    return {
        "knowledge_contribution": {
            "description": "Contribute knowledge to the collective hive mind",
            "parameters": {
                "source_type": {"type": "string", "required": True},
                "content": {"type": "string", "required": True},
                "domain": {"type": "string", "required": False},
                "confidence": {"type": "number", "required": False, "default": 0.8},
                "metadata": {"type": "object", "required": False}
            },
            "handler": knowledge_contribution_tool
        },
        "knowledge_search": {
            "description": "Search the collective knowledge base",
            "parameters": {
                "query": {"type": "string", "required": True},
                "domain": {"type": "string", "required": False},
                "max_results": {"type": "integer", "required": False, "default": 10},
                "min_quality": {"type": "number", "required": False, "default": 0.5}
            },
            "handler": knowledge_search_tool
        },
        "knowledge_stats": {
            "description": "Get statistics about the knowledge base",
            "parameters": {},
            "handler": knowledge_stats_tool
        }
    }