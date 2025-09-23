"""
Semantic Search Engine for Hive Mind Knowledge Management

Implements vector-based semantic search with embeddings for the MCP Swarm Intelligence Server.
Provides similarity-based knowledge retrieval with confidence scoring and caching.

According to orchestrator routing:
- Primary: hive_mind_specialist.md
- Secondary: memory_management_specialist.md, mcp_specialist.md
"""
import asyncio
import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None
try:
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except ImportError:
    cosine_similarity = None
import hashlib
import time

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a semantic search result with metadata"""
    content: str
    namespace: str
    key: str
    confidence: float
    similarity_score: float
    metadata: Dict[str, Any]
    access_count: int
    last_accessed: float

@dataclass
class SearchConfig:
    """Configuration for semantic search parameters"""
    model_name: str = "all-MiniLM-L6-v2"
    max_results: int = 10
    min_similarity_threshold: float = 0.3
    cache_timeout: int = 3600  # 1 hour
    enable_caching: bool = True
    rerank_results: bool = True

class SemanticSearchEngine:
    """
    High-performance semantic search engine with vector embeddings.
    
    Features:
    - Sentence transformer embeddings for semantic similarity
    - SQLite-based vector storage with efficient retrieval
    - Intelligent caching for frequently accessed queries
    - Cosine similarity matching with configurable thresholds
    - Real-time search result ranking and confidence scoring
    """
    
    def __init__(self, db_path: str, config: Optional[SearchConfig] = None):
        """
        Initialize semantic search engine with database and model.
        
        Args:
            db_path: Path to SQLite database
            config: Search configuration parameters
        """
        self.db_path = db_path
        self.config = config or SearchConfig()
        self.model = None
        self.embedding_cache = {}
        self.search_cache = {}
        self._initialize_model()
        self._initialize_database()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model for embeddings."""
        try:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers not available")
            self.model = SentenceTransformer(self.config.model_name)
            logger.info(f"Loaded semantic model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            raise
    
    def _initialize_database(self):
        """Initialize database schema for semantic search if needed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                # Check if hive_knowledge table exists and has embedding column
                cursor = conn.execute("""
                    SELECT sql FROM sqlite_master 
                    WHERE type='table' AND name='hive_knowledge'
                """)
                result = cursor.fetchone()
                if not result or 'embedding' not in result[0]:
                    logger.warning("hive_knowledge table missing embedding column")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate vector embedding for text input.
        
        Args:
            text: Input text to embed
            
        Returns:
            Vector embedding as numpy array
        """
        if self.model is None:
            raise RuntimeError("Semantic model not initialized")
            
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        try:
            # Generate embedding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                lambda: self.model.encode([text])[0]  # type: ignore
            )
            
            # Cache the embedding
            if self.config.enable_caching:
                self.embedding_cache[text_hash] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def search_knowledge(
        self, 
        query: str, 
        namespace: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search across knowledge base.
        
        Args:
            query: Search query text
            namespace: Optional namespace filter
            max_results: Maximum number of results to return
            
        Returns:
            List of search results ranked by similarity
        """
        max_results = max_results or self.config.max_results
        
        # Check search cache
        cache_key = f"{query}:{namespace}:{max_results}"
        if self.config.enable_caching and cache_key in self.search_cache:
            cached_result, timestamp = self.search_cache[cache_key]
            if time.time() - timestamp < self.config.cache_timeout:
                return cached_result
        
        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            
            # Retrieve all knowledge entries with embeddings
            knowledge_entries = await self._get_knowledge_entries(namespace)
            
            # Calculate similarities
            results = []
            for entry in knowledge_entries:
                if entry['embedding'] is None:
                    continue
                
                # Deserialize embedding
                stored_embedding = np.frombuffer(entry['embedding'], dtype=np.float32)
                
                # Calculate cosine similarity
                if cosine_similarity is None:
                    raise RuntimeError("scikit-learn not available for similarity calculation")
                    
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    stored_embedding.reshape(1, -1)
                )[0][0]
                
                # Filter by threshold
                if similarity >= self.config.min_similarity_threshold:
                    result = SearchResult(
                        content=entry['content'],
                        namespace=entry['namespace'],
                        key=entry['key'],
                        confidence=entry['confidence'],
                        similarity_score=float(similarity),
                        metadata=json.loads(entry['metadata']) if entry['metadata'] else {},
                        access_count=entry['access_count'],
                        last_accessed=entry['accessed_at'] or 0
                    )
                    results.append(result)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Limit results
            results = results[:max_results]
            
            # Rerank if enabled
            if self.config.rerank_results:
                results = await self._rerank_results(query, results)
            
            # Update access statistics
            await self._update_access_stats([r.key for r in results])
            
            # Cache results
            if self.config.enable_caching:
                self.search_cache[cache_key] = (results, time.time())
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    async def _get_knowledge_entries(self, namespace: Optional[str] = None) -> List[Dict]:
        """Retrieve knowledge entries from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if namespace:
                    cursor = conn.execute("""
                        SELECT id, namespace, key, content, metadata, embedding, 
                               confidence, accessed_at, access_count
                        FROM hive_knowledge 
                        WHERE namespace = ?
                    """, (namespace,))
                else:
                    cursor = conn.execute("""
                        SELECT id, namespace, key, content, metadata, embedding, 
                               confidence, accessed_at, access_count
                        FROM hive_knowledge
                    """)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge entries: {e}")
            raise
    
    async def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Rerank search results using additional factors beyond similarity.
        
        Factors:
        - Confidence score of knowledge entry
        - Access frequency (popularity)
        - Recency of access
        - Content length relevance
        """
        try:
            for result in results:
                # Base score is similarity
                base_score = result.similarity_score
                
                # Confidence boost (0-1)
                confidence_boost = result.confidence * 0.1
                
                # Popularity boost based on access count (normalized)
                max_access = max((r.access_count for r in results), default=1)
                popularity_boost = (result.access_count / max_access) * 0.05
                
                # Recency boost for recently accessed items
                current_time = time.time()
                if result.last_accessed > 0:
                    hours_since_access = (current_time - result.last_accessed) / 3600
                    recency_boost = max(0, (24 - hours_since_access) / 24) * 0.05
                else:
                    recency_boost = 0
                
                # Content length relevance (prefer moderate length)
                content_len = len(result.content)
                if 100 <= content_len <= 1000:
                    length_boost = 0.02
                elif content_len < 100:
                    length_boost = -0.01
                else:
                    length_boost = 0
                
                # Calculate final score
                final_score = base_score + confidence_boost + popularity_boost + recency_boost + length_boost
                result.similarity_score = min(1.0, final_score)  # Cap at 1.0
            
            # Re-sort by updated scores
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results
            
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return results
    
    async def _update_access_stats(self, keys: List[str]):
        """Update access statistics for retrieved knowledge entries."""
        if not keys:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                current_time = time.time()
                placeholders = ','.join(['?' for _ in keys])
                conn.execute(f"""
                    UPDATE hive_knowledge 
                    SET access_count = access_count + 1,
                        accessed_at = ?
                    WHERE key IN ({placeholders})
                """, [current_time] + keys)
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Failed to update access stats: {e}")
    
    async def add_knowledge_with_embedding(
        self, 
        namespace: str, 
        key: str, 
        content: str,
        metadata: Optional[Dict] = None,
        confidence: float = 1.0
    ) -> bool:
        """
        Add knowledge entry with pre-computed embedding.
        
        Args:
            namespace: Knowledge namespace
            key: Unique key within namespace
            content: Knowledge content
            metadata: Optional metadata dictionary
            confidence: Confidence score (0-1)
            
        Returns:
            True if successfully added
        """
        try:
            # Generate embedding for content
            embedding = await self.generate_embedding(content)
            embedding_blob = embedding.astype(np.float32).tobytes()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO hive_knowledge 
                    (namespace, key, content, metadata, embedding, confidence, 
                     created_at, updated_at, accessed_at, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    namespace, key, content, 
                    json.dumps(metadata) if metadata else None,
                    embedding_blob, confidence,
                    time.time(), time.time(), 0, 0
                ))
                conn.commit()
            
            logger.info(f"Added knowledge: {namespace}:{key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add knowledge with embedding: {e}")
            return False
    
    async def update_embeddings_batch(self, batch_size: int = 100) -> int:
        """
        Update embeddings for knowledge entries that don't have them.
        
        Args:
            batch_size: Number of entries to process at once
            
        Returns:
            Number of embeddings updated
        """
        updated_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get entries without embeddings
                cursor = conn.execute("""
                    SELECT id, content FROM hive_knowledge 
                    WHERE embedding IS NULL 
                    LIMIT ?
                """, (batch_size,))
                
                entries = cursor.fetchall()
                
                for entry in entries:
                    try:
                        # Generate embedding
                        embedding = await self.generate_embedding(entry['content'])
                        embedding_blob = embedding.astype(np.float32).tobytes()
                        
                        # Update database
                        conn.execute("""
                            UPDATE hive_knowledge 
                            SET embedding = ?, updated_at = ?
                            WHERE id = ?
                        """, (embedding_blob, time.time(), entry['id']))
                        
                        updated_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to update embedding for entry {entry['id']}: {e}")
                
                conn.commit()
                
            logger.info(f"Updated {updated_count} embeddings")
            return updated_count
            
        except Exception as e:
            logger.error(f"Batch embedding update failed: {e}")
            return updated_count
    
    def clear_cache(self):
        """Clear all caches to free memory."""
        self.embedding_cache.clear()
        self.search_cache.clear()
        logger.info("Cleared semantic search caches")
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics and performance metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Count total knowledge entries
                cursor = conn.execute("SELECT COUNT(*) FROM hive_knowledge")
                total_entries = cursor.fetchone()[0]
                
                # Count entries with embeddings
                cursor = conn.execute("SELECT COUNT(*) FROM hive_knowledge WHERE embedding IS NOT NULL")
                embedded_entries = cursor.fetchone()[0]
                
                # Get most accessed entries
                cursor = conn.execute("""
                    SELECT namespace, key, access_count 
                    FROM hive_knowledge 
                    ORDER BY access_count DESC 
                    LIMIT 5
                """)
                popular_entries = cursor.fetchall()
            
            return {
                "total_entries": total_entries,
                "embedded_entries": embedded_entries,
                "embedding_coverage": embedded_entries / max(total_entries, 1),
                "cache_size": {
                    "embeddings": len(self.embedding_cache),
                    "searches": len(self.search_cache)
                },
                "popular_entries": popular_entries,
                "model_name": self.config.model_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            return {}