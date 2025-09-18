"""
Pheromone Trail Management for Swarm Intelligence

This module implements pheromone trail management systems for coordinating
swarm intelligence behaviors in the MCP Swarm Intelligence Server. It provides
persistent pheromone storage, decay mechanisms, and reinforcement patterns
for optimizing multi-agent coordination.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any, Set
from dataclasses import dataclass, field
import sqlite3
import asyncio
import logging
from datetime import datetime, timedelta
import json
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class PheromoneEntry:
    """Represents a single pheromone trail entry."""
    source_id: str
    target_id: str
    trail_type: str
    intensity: float
    last_updated: datetime
    decay_rate: float = 0.1
    success_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrailPattern:
    """Represents a discovered pattern in pheromone trails."""
    pattern_id: str
    source_pattern: str
    target_pattern: str
    confidence: float
    frequency: int
    last_seen: datetime
    success_rate: float


@dataclass
class TrailStatistics:
    """Statistics about pheromone trail usage."""
    total_trails: int
    active_trails: int
    average_intensity: float
    strongest_trail: Optional[PheromoneEntry]
    most_used_path: Optional[Tuple[str, str]]
    decay_efficiency: float
    pattern_count: int


class PheromoneTrail:
    """
    Manage pheromone trails for swarm coordination.
    
    This class provides comprehensive pheromone trail management including:
    - Persistent storage in SQLite database
    - Automatic decay and evaporation
    - Success-based reinforcement
    - Pattern recognition and learning
    - Cross-session persistence
    - Real-time trail intensity calculations
    """
    
    def __init__(
        self,
        database_path: str = "data/memory.db",
        default_decay_rate: float = 0.1,
        min_intensity: float = 0.01,
        max_intensity: float = 10.0,
        decay_interval: float = 300.0,  # 5 minutes
        pattern_detection_threshold: int = 5,
        auto_cleanup: bool = True
    ):
        """
        Initialize pheromone trail manager.
        
        Args:
            database_path: Path to SQLite database for persistence
            default_decay_rate: Default decay rate for trails (per decay_interval)
            min_intensity: Minimum trail intensity before removal
            max_intensity: Maximum trail intensity (clamping)
            decay_interval: Time interval for decay processing (seconds)
            pattern_detection_threshold: Minimum frequency for pattern detection
            auto_cleanup: Whether to automatically clean up expired trails
        """
        self.database_path = database_path
        self.default_decay_rate = default_decay_rate
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.decay_interval = decay_interval
        self.pattern_detection_threshold = pattern_detection_threshold
        self.auto_cleanup = auto_cleanup
        
        # In-memory cache for performance
        self._trail_cache: Dict[Tuple[str, str, str], PheromoneEntry] = {}
        self._cache_lock = threading.RLock()
        self._last_decay_time = datetime.now()
        
        # Pattern recognition
        self._detected_patterns: Dict[str, TrailPattern] = {}
        self._pattern_lock = threading.RLock()
        
        # Background tasks
        self._decay_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Initialize database
        self._initialize_database()
    
    async def start(self) -> None:
        """Start the pheromone trail manager."""
        if self._running:
            return
        
        self._running = True
        
        # Load existing trails from database
        await self._load_trails_from_database()
        
        # Start background decay task
        self._decay_task = asyncio.create_task(self._decay_loop())
        
        logger.info("Pheromone trail manager started")
    
    async def stop(self) -> None:
        """Stop the pheromone trail manager."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop background task
        if self._decay_task:
            self._decay_task.cancel()
            try:
                await self._decay_task
            except asyncio.CancelledError:
                pass
        
        # Save all trails to database
        await self._save_trails_to_database()
        
        logger.info("Pheromone trail manager stopped")
    
    def _initialize_database(self) -> None:
        """Initialize the SQLite database schema."""
        with self._get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pheromone_trails (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    trail_type TEXT NOT NULL,
                    intensity REAL NOT NULL,
                    last_updated TIMESTAMP NOT NULL,
                    decay_rate REAL NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    UNIQUE(source_id, target_id, trail_type)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trail_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT UNIQUE NOT NULL,
                    source_pattern TEXT NOT NULL,
                    target_pattern TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    frequency INTEGER NOT NULL,
                    last_seen TIMESTAMP NOT NULL,
                    success_rate REAL NOT NULL
                )
            """)
            
            # Create indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trails_source 
                ON pheromone_trails(source_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trails_target 
                ON pheromone_trails(target_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trails_type 
                ON pheromone_trails(trail_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trails_intensity 
                ON pheromone_trails(intensity)
            """)
    
    @contextmanager
    def _get_db_connection(self):
        """Get a database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.database_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error("Database error: %s", e)
            raise
        finally:
            if conn:
                conn.close()
    
    async def deposit_pheromone(
        self,
        source_id: str,
        target_id: str,
        trail_type: str,
        intensity: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Deposit pheromone on a trail between source and target.
        
        Args:
            source_id: Source identifier (agent, task, etc.)
            target_id: Target identifier
            trail_type: Type of trail (task_assignment, consensus, etc.)
            intensity: Pheromone intensity to deposit
            success: Whether this deposit represents a successful interaction
            metadata: Additional metadata for the trail
        """
        if intensity <= 0:
            return
        
        trail_key = (source_id, target_id, trail_type)
        
        with self._cache_lock:
            existing_trail = self._trail_cache.get(trail_key)
            
            if existing_trail:
                # Update existing trail
                existing_trail.intensity = min(
                    self.max_intensity,
                    existing_trail.intensity + intensity
                )
                existing_trail.last_updated = datetime.now()
                
                if success:
                    existing_trail.success_count += 1
                else:
                    existing_trail.failure_count += 1
                
                # Update metadata
                if metadata:
                    existing_trail.metadata.update(metadata)
            else:
                # Create new trail
                new_trail = PheromoneEntry(
                    source_id=source_id,
                    target_id=target_id,
                    trail_type=trail_type,
                    intensity=min(self.max_intensity, intensity),
                    last_updated=datetime.now(),
                    decay_rate=self.default_decay_rate,
                    success_count=1 if success else 0,
                    failure_count=0 if success else 1,
                    metadata=metadata or {}
                )
                self._trail_cache[trail_key] = new_trail
        
        # Update pattern detection
        await self._update_patterns(source_id, target_id, trail_type, success)
        
        logger.debug("Deposited pheromone: %s -> %s (%s) intensity=%.3f", 
                    source_id, target_id, trail_type, intensity)
    
    async def get_trail_intensity(
        self,
        source_id: str,
        target_id: str,
        trail_type: str,
        apply_decay: bool = True
    ) -> float:
        """
        Get current pheromone intensity for a trail.
        
        Args:
            source_id: Source identifier
            target_id: Target identifier
            trail_type: Type of trail
            apply_decay: Whether to apply time-based decay
            
        Returns:
            Current trail intensity
        """
        trail_key = (source_id, target_id, trail_type)
        
        with self._cache_lock:
            trail = self._trail_cache.get(trail_key)
            
            if not trail:
                return 0.0
            
            intensity = trail.intensity
            
            if apply_decay:
                # Calculate time-based decay
                time_since_update = (datetime.now() - trail.last_updated).total_seconds()
                decay_periods = time_since_update / self.decay_interval
                decay_factor = (1.0 - trail.decay_rate) ** decay_periods
                intensity *= decay_factor
            
            return max(0.0, intensity)
    
    async def get_strongest_trails(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        trail_type: Optional[str] = None,
        limit: int = 10,
        min_intensity: Optional[float] = None
    ) -> List[PheromoneEntry]:
        """
        Get the strongest pheromone trails matching criteria.
        
        Args:
            source_id: Filter by source (optional)
            target_id: Filter by target (optional)
            trail_type: Filter by trail type (optional)
            limit: Maximum number of trails to return
            min_intensity: Minimum intensity threshold
            
        Returns:
            List of strongest trails matching criteria
        """
        matching_trails = []
        min_threshold = min_intensity or self.min_intensity
        
        with self._cache_lock:
            for trail in self._trail_cache.values():
                # Apply filters
                if source_id and trail.source_id != source_id:
                    continue
                if target_id and trail.target_id != target_id:
                    continue
                if trail_type and trail.trail_type != trail_type:
                    continue
                
                # Calculate current intensity with decay
                current_intensity = await self.get_trail_intensity(
                    trail.source_id, trail.target_id, trail.trail_type
                )
                
                if current_intensity >= min_threshold:
                    # Create a copy with current intensity
                    trail_copy = PheromoneEntry(
                        source_id=trail.source_id,
                        target_id=trail.target_id,
                        trail_type=trail.trail_type,
                        intensity=current_intensity,
                        last_updated=trail.last_updated,
                        decay_rate=trail.decay_rate,
                        success_count=trail.success_count,
                        failure_count=trail.failure_count,
                        metadata=trail.metadata.copy()
                    )
                    matching_trails.append(trail_copy)
        
        # Sort by intensity and limit results
        matching_trails.sort(key=lambda t: t.intensity, reverse=True)
        return matching_trails[:limit]
    
    async def get_trail_suggestions(
        self,
        source_id: str,
        trail_type: str,
        exclude_targets: Optional[Set[str]] = None,
        limit: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get suggested targets based on pheromone trails.
        
        Args:
            source_id: Source identifier
            trail_type: Type of trail to consider
            exclude_targets: Targets to exclude from suggestions
            limit: Maximum number of suggestions
            
        Returns:
            List of (target_id, intensity) suggestions
        """
        suggestions = []
        exclude_set = exclude_targets or set()
        
        trails = await self.get_strongest_trails(
            source_id=source_id,
            trail_type=trail_type,
            limit=limit * 2  # Get more to account for exclusions
        )
        
        for trail in trails:
            if trail.target_id not in exclude_set:
                suggestions.append((trail.target_id, trail.intensity))
                if len(suggestions) >= limit:
                    break
        
        return suggestions
    
    async def evaporate_trails(
        self,
        trail_type: Optional[str] = None,
        evaporation_rate: float = 0.1
    ) -> int:
        """
        Apply evaporation to pheromone trails.
        
        Args:
            trail_type: Specific trail type to evaporate (optional)
            evaporation_rate: Rate of evaporation (0.0 to 1.0)
            
        Returns:
            Number of trails affected
        """
        evaporated_count = 0
        trails_to_remove = []
        
        with self._cache_lock:
            for trail_key, trail in self._trail_cache.items():
                if trail_type and trail.trail_type != trail_type:
                    continue
                
                # Apply evaporation
                trail.intensity *= (1.0 - evaporation_rate)
                evaporated_count += 1
                
                # Mark for removal if below threshold
                if trail.intensity < self.min_intensity:
                    trails_to_remove.append(trail_key)
            
            # Remove trails below threshold
            for trail_key in trails_to_remove:
                del self._trail_cache[trail_key]
        
        logger.debug("Evaporated %d trails, removed %d weak trails", 
                    evaporated_count, len(trails_to_remove))
        
        return evaporated_count
    
    async def _decay_loop(self) -> None:
        """Background task for periodic decay processing."""
        while self._running:
            try:
                await asyncio.sleep(self.decay_interval)
                
                if not self._running:
                    break
                
                # Apply natural decay
                await self._apply_natural_decay()
                
                # Clean up expired trails
                if self.auto_cleanup:
                    await self._cleanup_expired_trails()
                
                # Detect new patterns
                await self._detect_patterns()
                
                # Periodic database save
                await self._save_trails_to_database()
                
            except asyncio.CancelledError:
                break
            except (sqlite3.Error, OSError) as e:
                logger.error("Error in decay loop: %s", e)
    
    async def _apply_natural_decay(self) -> None:
        """Apply natural time-based decay to all trails."""
        current_time = datetime.now()
        time_delta = (current_time - self._last_decay_time).total_seconds()
        
        if time_delta < self.decay_interval:
            return
        
        decay_periods = time_delta / self.decay_interval
        trails_to_remove = []
        
        with self._cache_lock:
            for trail_key, trail in self._trail_cache.items():
                # Calculate decay
                decay_factor = (1.0 - trail.decay_rate) ** decay_periods
                trail.intensity *= decay_factor
                
                # Mark weak trails for removal
                if trail.intensity < self.min_intensity:
                    trails_to_remove.append(trail_key)
            
            # Remove weak trails
            for trail_key in trails_to_remove:
                del self._trail_cache[trail_key]
        
        self._last_decay_time = current_time
        
        if trails_to_remove:
            logger.debug("Natural decay removed %d weak trails", len(trails_to_remove))
    
    async def _cleanup_expired_trails(self) -> None:
        """Clean up trails that haven't been updated in a long time."""
        cutoff_time = datetime.now() - timedelta(hours=24)  # 24 hour cutoff
        trails_to_remove = []
        
        with self._cache_lock:
            for trail_key, trail in self._trail_cache.items():
                if trail.last_updated < cutoff_time and trail.intensity < 0.1:
                    trails_to_remove.append(trail_key)
            
            for trail_key in trails_to_remove:
                del self._trail_cache[trail_key]
        
        if trails_to_remove:
            logger.debug("Cleaned up %d expired trails", len(trails_to_remove))
    
    async def _update_patterns(
        self,
        source_id: str,
        target_id: str,
        trail_type: str,
        success: bool
    ) -> None:
        """Update pattern detection with new trail information."""
        # Simple pattern: source_type -> target_type
        source_pattern = source_id.split('_')[0] if '_' in source_id else source_id
        target_pattern = target_id.split('_')[0] if '_' in target_id else target_id
        
        pattern_key = f"{source_pattern}->{target_pattern}:{trail_type}"
        
        with self._pattern_lock:
            pattern = self._detected_patterns.get(pattern_key)
            
            if pattern:
                pattern.frequency += 1
                pattern.last_seen = datetime.now()
                
                # Update success rate
                if success:
                    pattern.success_rate = (
                        pattern.success_rate * (pattern.frequency - 1) + 1.0
                    ) / pattern.frequency
                else:
                    pattern.success_rate = (
                        pattern.success_rate * (pattern.frequency - 1)
                    ) / pattern.frequency
                
                # Update confidence based on frequency and success rate
                pattern.confidence = min(1.0, 
                    (pattern.frequency / self.pattern_detection_threshold) * 
                    pattern.success_rate
                )
            else:
                # Create new pattern
                self._detected_patterns[pattern_key] = TrailPattern(
                    pattern_id=pattern_key,
                    source_pattern=source_pattern,
                    target_pattern=target_pattern,
                    confidence=0.1,
                    frequency=1,
                    last_seen=datetime.now(),
                    success_rate=1.0 if success else 0.0
                )
    
    async def _detect_patterns(self) -> None:
        """Detect and update trail patterns."""
        # Pattern detection is updated in real-time via _update_patterns
        # This method could be extended for more complex pattern analysis
        # Currently no additional processing needed
        return
    
    async def _load_trails_from_database(self) -> None:
        """Load existing trails from database into cache."""
        with self._get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT source_id, target_id, trail_type, intensity, last_updated,
                       decay_rate, success_count, failure_count, metadata
                FROM pheromone_trails
                WHERE intensity > ?
            """, (self.min_intensity,))
            
            loaded_count = 0
            
            with self._cache_lock:
                for row in cursor:
                    trail_key = (row['source_id'], row['target_id'], row['trail_type'])
                    
                    metadata = {}
                    if row['metadata']:
                        try:
                            metadata = json.loads(row['metadata'])
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse metadata for trail %s", trail_key)
                    
                    trail = PheromoneEntry(
                        source_id=row['source_id'],
                        target_id=row['target_id'],
                        trail_type=row['trail_type'],
                        intensity=row['intensity'],
                        last_updated=datetime.fromisoformat(row['last_updated']),
                        decay_rate=row['decay_rate'],
                        success_count=row['success_count'],
                        failure_count=row['failure_count'],
                        metadata=metadata
                    )
                    
                    self._trail_cache[trail_key] = trail
                    loaded_count += 1
            
            logger.info("Loaded %d pheromone trails from database", loaded_count)
    
    async def _save_trails_to_database(self) -> None:
        """Save current trails to database."""
        if not self._trail_cache:
            return
        
        with self._get_db_connection() as conn:
            # Clear existing trails
            conn.execute("DELETE FROM pheromone_trails")
            
            # Insert current trails
            with self._cache_lock:
                for trail in self._trail_cache.values():
                    metadata_json = json.dumps(trail.metadata) if trail.metadata else None
                    
                    conn.execute("""
                        INSERT INTO pheromone_trails 
                        (source_id, target_id, trail_type, intensity, last_updated,
                         decay_rate, success_count, failure_count, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trail.source_id, trail.target_id, trail.trail_type,
                        trail.intensity, trail.last_updated.isoformat(),
                        trail.decay_rate, trail.success_count, trail.failure_count,
                        metadata_json
                    ))
            
            logger.debug("Saved %d trails to database", len(self._trail_cache))
    
    async def get_statistics(self) -> TrailStatistics:
        """Get comprehensive statistics about pheromone trails."""
        with self._cache_lock:
            if not self._trail_cache:
                return TrailStatistics(
                    total_trails=0,
                    active_trails=0,
                    average_intensity=0.0,
                    strongest_trail=None,
                    most_used_path=None,
                    decay_efficiency=0.0,
                    pattern_count=0
                )
            
            total_trails = len(self._trail_cache)
            intensities = [trail.intensity for trail in self._trail_cache.values()]
            active_trails = sum(1 for i in intensities if i >= self.min_intensity)
            average_intensity = float(np.mean(intensities))
            
            # Find strongest trail
            strongest_trail = max(self._trail_cache.values(), key=lambda t: t.intensity)
            
            # Find most used path (highest success count)
            most_used_trail = max(
                self._trail_cache.values(), 
                key=lambda t: t.success_count + t.failure_count
            )
            most_used_path = (most_used_trail.source_id, most_used_trail.target_id)
            
            # Calculate decay efficiency (how well decay is working)
            weak_trails = sum(1 for i in intensities if i < self.min_intensity * 2)
            decay_efficiency = 1.0 - (weak_trails / total_trails) if total_trails > 0 else 1.0
            
            with self._pattern_lock:
                pattern_count = len(self._detected_patterns)
            
            return TrailStatistics(
                total_trails=total_trails,
                active_trails=active_trails,
                average_intensity=average_intensity,
                strongest_trail=strongest_trail,
                most_used_path=most_used_path,
                decay_efficiency=decay_efficiency,
                pattern_count=pattern_count
            )
    
    def get_detected_patterns(self) -> List[TrailPattern]:
        """Get all detected trail patterns."""
        with self._pattern_lock:
            return list(self._detected_patterns.values())
    
    async def clear_trails(
        self,
        trail_type: Optional[str] = None,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None
    ) -> int:
        """
        Clear pheromone trails matching criteria.
        
        Args:
            trail_type: Clear only this trail type (optional)
            source_id: Clear only trails from this source (optional)
            target_id: Clear only trails to this target (optional)
            
        Returns:
            Number of trails cleared
        """
        trails_to_remove = []
        
        with self._cache_lock:
            for trail_key, trail in self._trail_cache.items():
                if trail_type and trail.trail_type != trail_type:
                    continue
                if source_id and trail.source_id != source_id:
                    continue
                if target_id and trail.target_id != target_id:
                    continue
                
                trails_to_remove.append(trail_key)
            
            for trail_key in trails_to_remove:
                del self._trail_cache[trail_key]
        
        cleared_count = len(trails_to_remove)
        logger.info("Cleared %d pheromone trails", cleared_count)
        
        return cleared_count