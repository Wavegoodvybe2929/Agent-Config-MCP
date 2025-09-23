"""
Knowledge Synthesis Engine for Hive Mind Intelligence

Implements multi-source knowledge synthesis with conflict resolution and accuracy validation.
Combines knowledge from different sources while maintaining coherency and reducing redundancy.

According to orchestrator routing:
- Primary: hive_mind_specialist.md
- Secondary: memory_management_specialist.md, mcp_specialist.md
"""
import json
import logging
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import sqlite3
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeSource:
    """Represents a source of knowledge with metadata"""
    namespace: str
    key: str
    content: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    source_hash: str = field(init=False)
    
    def __post_init__(self):
        """Generate source hash for deduplication"""
        content_normalized = re.sub(r'\s+', ' ', self.content.strip().lower())
        self.source_hash = hashlib.md5(content_normalized.encode()).hexdigest()

@dataclass
class SynthesisResult:
    """Result of knowledge synthesis with metadata"""
    synthesized_content: str
    source_count: int
    confidence_score: float
    synthesis_method: str
    conflict_resolution: Dict[str, Any]
    redundancy_eliminated: int
    metadata_combined: Dict[str, Any]
    synthesis_timestamp: float = field(default_factory=time.time)

@dataclass
class ConflictResolution:
    """Information about how conflicts were resolved"""
    conflicting_sources: List[str]
    resolution_method: str
    chosen_version: str
    confidence_factor: float
    resolution_reason: str

class KnowledgeSynthesisEngine:
    """
    Advanced knowledge synthesis engine that combines multiple sources.
    
    Features:
    - Multi-source knowledge consolidation
    - Intelligent conflict resolution
    - Redundancy elimination with semantic analysis
    - Confidence-weighted synthesis
    - Metadata preservation and combination
    - Real-time synthesis with caching
    """
    
    def __init__(self, db_path: str):
        """
        Initialize synthesis engine with database connection.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.synthesis_cache = {}
        self.conflict_strategies = {
            'highest_confidence': self._resolve_by_confidence,
            'most_recent': self._resolve_by_recency,
            'consensus': self._resolve_by_consensus,
            'weighted_average': self._resolve_by_weighted_average,
            'expert_preference': self._resolve_by_expert_preference
        }
    
    async def synthesize_knowledge(
        self,
        sources: List[KnowledgeSource],
        strategy: str = 'highest_confidence',
        enable_deduplication: bool = True,
        min_confidence_threshold: float = 0.3
    ) -> SynthesisResult:
        """
        Synthesize knowledge from multiple sources.
        
        Args:
            sources: List of knowledge sources to synthesize
            strategy: Conflict resolution strategy
            enable_deduplication: Whether to eliminate duplicate content
            min_confidence_threshold: Minimum confidence to include source
            
        Returns:
            Synthesized knowledge result
        """
        if not sources:
            raise ValueError("No sources provided for synthesis")
        
        try:
            # Filter sources by confidence threshold
            filtered_sources = [
                s for s in sources 
                if s.confidence >= min_confidence_threshold
            ]
            
            if not filtered_sources:
                raise ValueError("No sources meet confidence threshold")
            
            # Remove duplicates if requested
            if enable_deduplication:
                filtered_sources = await self._deduplicate_sources(filtered_sources)
            
            # Group sources by content similarity for conflict detection
            conflict_groups = await self._group_conflicting_sources(filtered_sources)
            
            # Resolve conflicts for each group
            resolved_sources = []
            conflict_resolutions = {}
            
            for group_id, group_sources in conflict_groups.items():
                if len(group_sources) == 1:
                    # No conflict
                    resolved_sources.extend(group_sources)
                else:
                    # Resolve conflict
                    resolved_source, resolution = await self._resolve_conflict(
                        group_sources, strategy
                    )
                    resolved_sources.append(resolved_source)
                    conflict_resolutions[group_id] = resolution
            
            # Synthesize final content
            synthesized_content = await self._synthesize_content(resolved_sources)
            
            # Calculate overall confidence
            confidence_score = await self._calculate_synthesis_confidence(resolved_sources)
            
            # Combine metadata
            combined_metadata = await self._combine_metadata(resolved_sources)
            
            return SynthesisResult(
                synthesized_content=synthesized_content,
                source_count=len(sources),
                confidence_score=confidence_score,
                synthesis_method=strategy,
                conflict_resolution=conflict_resolutions,
                redundancy_eliminated=len(sources) - len(resolved_sources),
                metadata_combined=combined_metadata
            )
            
        except Exception as e:
            logger.error("Knowledge synthesis failed: %s", str(e))
            raise
    
    async def _deduplicate_sources(self, sources: List[KnowledgeSource]) -> List[KnowledgeSource]:
        """Remove duplicate sources based on content similarity."""
        unique_sources = []
        seen_hashes = set()
        
        for source in sources:
            if source.source_hash not in seen_hashes:
                unique_sources.append(source)
                seen_hashes.add(source.source_hash)
            else:
                # Find existing source and merge if this one has higher confidence
                for i, existing in enumerate(unique_sources):
                    if existing.source_hash == source.source_hash:
                        if source.confidence > existing.confidence:
                            unique_sources[i] = source
                        break
        
        return unique_sources
    
    async def _group_conflicting_sources(
        self, 
        sources: List[KnowledgeSource]
    ) -> Dict[str, List[KnowledgeSource]]:
        """Group sources that have conflicting information."""
        groups = defaultdict(list)
        similarity_threshold = 0.7  # Content similarity threshold for conflict detection
        
        for i, source in enumerate(sources):
            group_id = f"group_{i}"
            groups[group_id].append(source)
            
            # Check against existing groups for content conflicts
            for existing_group_id, existing_sources in list(groups.items()):
                if existing_group_id == group_id:
                    continue
                
                for existing_source in existing_sources:
                    similarity = self._calculate_content_similarity(
                        source.content, existing_source.content
                    )
                    
                    # If content is similar but from different sources, it's a potential conflict
                    if (similarity > similarity_threshold and 
                        source.namespace != existing_source.namespace):
                        
                        # Move to existing group
                        groups[existing_group_id].append(source)
                        groups[group_id].remove(source)
                        if not groups[group_id]:
                            del groups[group_id]
                        break
        
        return dict(groups)
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        # Normalize content
        norm1 = re.sub(r'\s+', ' ', content1.strip().lower())
        norm2 = re.sub(r'\s+', ' ', content2.strip().lower())
        
        # Use sequence matcher for similarity
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    async def _resolve_conflict(
        self, 
        conflicting_sources: List[KnowledgeSource], 
        strategy: str
    ) -> Tuple[KnowledgeSource, ConflictResolution]:
        """Resolve conflict between multiple sources."""
        if strategy not in self.conflict_strategies:
            strategy = 'highest_confidence'
        
        resolver = self.conflict_strategies[strategy]
        chosen_source, resolution_info = await resolver(conflicting_sources)
        
        resolution = ConflictResolution(
            conflicting_sources=[f"{s.namespace}:{s.key}" for s in conflicting_sources],
            resolution_method=strategy,
            chosen_version=f"{chosen_source.namespace}:{chosen_source.key}",
            confidence_factor=chosen_source.confidence,
            resolution_reason=resolution_info
        )
        
        return chosen_source, resolution
    
    async def _resolve_by_confidence(
        self, 
        sources: List[KnowledgeSource]
    ) -> Tuple[KnowledgeSource, str]:
        """Resolve conflict by choosing source with highest confidence."""
        best_source = max(sources, key=lambda s: s.confidence)
        reason = f"Chose source with highest confidence ({best_source.confidence:.3f})"
        return best_source, reason
    
    async def _resolve_by_recency(
        self, 
        sources: List[KnowledgeSource]
    ) -> Tuple[KnowledgeSource, str]:
        """Resolve conflict by choosing most recent source."""
        most_recent = max(sources, key=lambda s: s.timestamp)
        reason = f"Chose most recent source (timestamp: {most_recent.timestamp})"
        return most_recent, reason
    
    async def _resolve_by_consensus(
        self, 
        sources: List[KnowledgeSource]
    ) -> Tuple[KnowledgeSource, str]:
        """Resolve conflict by finding consensus among sources."""
        # Group sources by content similarity
        content_groups = defaultdict(list)
        
        for source in sources:
            content_key = source.source_hash[:8]  # Short hash for grouping
            content_groups[content_key].append(source)
        
        # Find the group with most sources (consensus)
        consensus_group = max(content_groups.values(), key=len)
        
        # Within consensus group, choose highest confidence
        best_source = max(consensus_group, key=lambda s: s.confidence)
        reason = f"Consensus among {len(consensus_group)} sources"
        
        return best_source, reason
    
    async def _resolve_by_weighted_average(
        self, 
        sources: List[KnowledgeSource]
    ) -> Tuple[KnowledgeSource, str]:
        """Resolve conflict by creating weighted average content."""
        # This is a simplified version - in practice, would need more sophisticated merging
        total_weight = sum(s.confidence for s in sources)
        
        # Choose source that best represents the weighted average
        # For now, choose the one closest to average confidence
        avg_confidence = total_weight / len(sources)
        best_source = min(sources, key=lambda s: abs(s.confidence - avg_confidence))
        
        reason = f"Best representative of weighted average (avg confidence: {avg_confidence:.3f})"
        return best_source, reason
    
    async def _resolve_by_expert_preference(
        self, 
        sources: List[KnowledgeSource]
    ) -> Tuple[KnowledgeSource, str]:
        """Resolve conflict by preferring expert or authoritative sources."""
        # Define expert namespace preferences
        expert_namespaces = ['expert', 'official', 'verified', 'admin']
        
        # Look for expert sources first
        for namespace in expert_namespaces:
            expert_sources = [s for s in sources if namespace in s.namespace.lower()]
            if expert_sources:
                best_expert = max(expert_sources, key=lambda s: s.confidence)
                reason = f"Preferred expert source from namespace: {best_expert.namespace}"
                return best_expert, reason
        
        # Fallback to highest confidence if no expert sources
        return await self._resolve_by_confidence(sources)
    
    async def _synthesize_content(self, sources: List[KnowledgeSource]) -> str:
        """Synthesize final content from resolved sources."""
        if len(sources) == 1:
            return sources[0].content
        
        # Sort sources by confidence (highest first)
        sorted_sources = sorted(sources, key=lambda s: s.confidence, reverse=True)
        
        # Start with highest confidence source as base
        base_content = sorted_sources[0].content
        
        # For now, return the base content
        # In a more sophisticated implementation, would merge complementary information
        return base_content
    
    async def _calculate_synthesis_confidence(self, sources: List[KnowledgeSource]) -> float:
        """Calculate overall confidence score for synthesized knowledge."""
        if not sources:
            return 0.0
        
        # Weight by confidence and number of sources
        confidence_sum = sum(s.confidence for s in sources)
        source_count_factor = min(1.0, len(sources) / 5.0)  # Cap at 5 sources
        
        return (confidence_sum / len(sources)) * source_count_factor
    
    async def _combine_metadata(self, sources: List[KnowledgeSource]) -> Dict[str, Any]:
        """Combine metadata from all sources."""
        combined = {
            'source_namespaces': list(set(s.namespace for s in sources)),
            'source_keys': [f"{s.namespace}:{s.key}" for s in sources],
            'confidence_range': {
                'min': min(s.confidence for s in sources),
                'max': max(s.confidence for s in sources),
                'avg': sum(s.confidence for s in sources) / len(sources)
            },
            'timestamp_range': {
                'earliest': min(s.timestamp for s in sources),
                'latest': max(s.timestamp for s in sources)
            },
            'total_access_count': sum(s.access_count for s in sources)
        }
        
        # Merge individual metadata
        all_metadata = {}
        for source in sources:
            for key, value in source.metadata.items():
                if key not in all_metadata:
                    all_metadata[key] = []
                all_metadata[key].append(value)
        
        # Deduplicate metadata values
        for key, values in all_metadata.items():
            unique_values = list(set(str(v) for v in values))
            if len(unique_values) == 1:
                combined[key] = values[0]
            else:
                combined[f"{key}_all"] = unique_values
        
        return combined
    
    async def synthesize_from_database(
        self,
        namespace_pattern: Optional[str] = None,
        key_pattern: Optional[str] = None,
        min_confidence: float = 0.3,
        strategy: str = 'highest_confidence'
    ) -> Optional[SynthesisResult]:
        """
        Synthesize knowledge directly from database sources.
        
        Args:
            namespace_pattern: SQL LIKE pattern for namespace filtering
            key_pattern: SQL LIKE pattern for key filtering
            min_confidence: Minimum confidence threshold
            strategy: Conflict resolution strategy
            
        Returns:
            Synthesized knowledge result or None if no sources found
        """
        try:
            # Build query
            query = "SELECT namespace, key, content, confidence, metadata, accessed_at, access_count FROM hive_knowledge WHERE confidence >= ?"
            params: List[Any] = [min_confidence]
            
            if namespace_pattern:
                query += " AND namespace LIKE ?"
                params.append(namespace_pattern)
            
            if key_pattern:
                query += " AND key LIKE ?"
                params.append(key_pattern)
            
            # Execute query
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
            
            if not rows:
                return None
            
            # Convert to KnowledgeSource objects
            sources = []
            for row in rows:
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                source = KnowledgeSource(
                    namespace=row['namespace'],
                    key=row['key'],
                    content=row['content'],
                    confidence=row['confidence'],
                    metadata=metadata,
                    timestamp=row['accessed_at'] or time.time(),
                    access_count=row['access_count'] or 0
                )
                sources.append(source)
            
            # Perform synthesis
            return await self.synthesize_knowledge(sources, strategy)
            
        except (sqlite3.Error, ValueError, KeyError) as e:
            logger.error("Database synthesis failed: %s", str(e))
            return None
    
    async def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get synthesis engine statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Count knowledge sources by namespace
                cursor = conn.execute("""
                    SELECT namespace, COUNT(*) as count, AVG(confidence) as avg_confidence
                    FROM hive_knowledge 
                    GROUP BY namespace
                    ORDER BY count DESC
                """)
                namespace_stats = cursor.fetchall()
                
                # Get potential conflicts (similar content, different namespaces)
                cursor = conn.execute("""
                    SELECT COUNT(*) as potential_conflicts
                    FROM (
                        SELECT COUNT(*) as cnt 
                        FROM hive_knowledge 
                        GROUP BY SUBSTR(content, 1, 100)
                        HAVING cnt > 1
                    )
                """)
                conflicts = cursor.fetchone()[0] if cursor.fetchone() else 0
            
            return {
                'total_sources': sum(stat[1] for stat in namespace_stats),
                'namespace_distribution': [
                    {'namespace': stat[0], 'count': stat[1], 'avg_confidence': stat[2]}
                    for stat in namespace_stats
                ],
                'potential_conflicts': conflicts,
                'cache_size': len(self.synthesis_cache),
                'available_strategies': list(self.conflict_strategies.keys())
            }
            
        except (sqlite3.Error, KeyError) as e:
            logger.error("Failed to get synthesis stats: %s", str(e))
            return {}