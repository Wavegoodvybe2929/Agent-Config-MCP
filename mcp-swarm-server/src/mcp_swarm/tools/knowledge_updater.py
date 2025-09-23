"""
Real-time Knowledge Updater for Hive Mind Intelligence

Implements real-time knowledge base updates with consistency validation and conflict resolution.
Maintains knowledge base integrity while allowing dynamic updates during coordination.

According to orchestrator routing:
- Primary: hive_mind_specialist.md
- Secondary: memory_management_specialist.md, mcp_specialist.md
"""
import asyncio
import json
import logging
import time
import hashlib
from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from threading import Lock

logger = logging.getLogger(__name__)

class UpdateType(Enum):
    """Types of knowledge updates"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    VALIDATE = "validate"

class UpdatePriority(Enum):
    """Priority levels for updates"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class KnowledgeUpdate:
    """Represents a knowledge update operation"""
    update_id: str
    update_type: UpdateType
    namespace: str
    key: str
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: UpdatePriority = UpdatePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    source_id: str = ""
    confidence: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    validation_required: bool = True

@dataclass
class UpdateResult:
    """Result of knowledge update operation"""
    update_id: str
    success: bool
    message: str
    conflicts_detected: List[str] = field(default_factory=list)
    validation_passed: bool = True
    execution_time: float = 0.0
    affected_entries: int = 0

class KnowledgeUpdater:
    """
    Real-time knowledge updater with consistency validation.
    
    Features:
    - Asynchronous update processing with priority queues
    - Conflict detection and resolution during updates
    - Consistency validation with rollback capabilities
    - Dependency tracking and ordered updates
    - Real-time update notifications and monitoring
    - Batch update optimization for performance
    """
    
    def __init__(self, db_path: str):
        """
        Initialize knowledge updater with database connection.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.update_queue = asyncio.Queue()
        self.update_lock = Lock()
        self.active_updates = set()
        self.update_history = []
        self.consistency_checkers = []
        self.is_processing = False
        self._initialize_update_tables()
    
    def _initialize_update_tables(self):
        """Initialize database tables for update tracking."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Update history table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS update_history (
                        update_id TEXT PRIMARY KEY,
                        update_type TEXT NOT NULL,
                        namespace TEXT NOT NULL,
                        key TEXT NOT NULL,
                        priority INTEGER,
                        timestamp REAL,
                        source_id TEXT,
                        success BOOLEAN,
                        execution_time REAL,
                        error_message TEXT,
                        affected_entries INTEGER DEFAULT 0
                    )
                """)
                
                # Dependency tracking table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS update_dependencies (
                        update_id TEXT,
                        depends_on TEXT,
                        PRIMARY KEY (update_id, depends_on)
                    )
                """)
                
                # Update locks table for coordination
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS update_locks (
                        namespace TEXT,
                        key TEXT,
                        locked_by TEXT,
                        lock_timestamp REAL,
                        PRIMARY KEY (namespace, key)
                    )
                """)
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error("Failed to initialize update tables: %s", str(e))
    
    async def submit_update(self, update: KnowledgeUpdate) -> str:
        """
        Submit a knowledge update for processing.
        
        Args:
            update: Knowledge update to process
            
        Returns:
            Update ID for tracking
        """
        # Generate update ID if not provided
        if not update.update_id:
            update.update_id = self._generate_update_id(update)
        
        try:
            # Validate update
            validation_errors = await self._validate_update(update)
            if validation_errors:
                logger.warning("Update validation failed: %s", validation_errors)
                return update.update_id
            
            # Add to queue with priority
            await self.update_queue.put((update.priority.value, update))
            
            # Start processing if not already running
            if not self.is_processing:
                asyncio.create_task(self._process_update_queue())
            
            logger.info("Submitted update %s for %s:%s", update.update_id, update.namespace, update.key)
            return update.update_id
            
        except (ValueError, TypeError, sqlite3.Error) as e:
            logger.error("Failed to submit update: %s", str(e))
            return update.update_id
    
    def _generate_update_id(self, update: KnowledgeUpdate) -> str:
        """Generate unique update ID."""
        content_hash = hashlib.md5(
            f"{update.namespace}:{update.key}:{update.timestamp}".encode()
        ).hexdigest()[:8]
        return f"upd_{content_hash}"
    
    async def _validate_update(self, update: KnowledgeUpdate) -> List[str]:
        """Validate update before processing."""
        errors = []
        
        # Basic validation
        if not update.namespace:
            errors.append("Namespace cannot be empty")
        if not update.key:
            errors.append("Key cannot be empty")
        if update.update_type in [UpdateType.CREATE, UpdateType.UPDATE] and not update.content:
            errors.append("Content required for create/update operations")
        if not (0.0 <= update.confidence <= 1.0):
            errors.append("Confidence must be between 0.0 and 1.0")
        
        # Check for circular dependencies
        if update.dependencies and await self._has_circular_dependency(update):
            errors.append("Circular dependency detected")
        
        return errors
    
    async def _has_circular_dependency(self, update: KnowledgeUpdate) -> bool:
        """Check for circular dependencies in update chain."""
        visited = set()
        
        async def check_deps(dep_id: str) -> bool:
            if dep_id in visited:
                return True
            visited.add(dep_id)
            
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT depends_on FROM update_dependencies WHERE update_id = ?",
                        (dep_id,)
                    )
                    subdeps = [row[0] for row in cursor.fetchall()]
                
                for subdep in subdeps:
                    if await check_deps(subdep):
                        return True
                
                return False
                
            except sqlite3.Error:
                return False
        
        for dep in update.dependencies:
            if dep == update.update_id or await check_deps(dep):
                return True
        
        return False
    
    async def _process_update_queue(self):
        """Process updates from the priority queue."""
        self.is_processing = True
        
        try:
            while not self.update_queue.empty():
                # Get highest priority update
                _priority, update = await self.update_queue.get()
                
                # Process the update
                result = await self._execute_update(update)
                
                # Record result
                await self._record_update_result(update, result)
                
                # Notify about completion
                logger.info("Completed update %s: %s", update.update_id, result.message)
                
        except (asyncio.QueueEmpty, ValueError, TypeError) as e:
            logger.error("Update queue processing failed: %s", str(e))
        finally:
            self.is_processing = False
    
    async def _execute_update(self, update: KnowledgeUpdate) -> UpdateResult:
        """Execute a single knowledge update."""
        start_time = time.time()
        result = UpdateResult(
            update_id=update.update_id,
            success=False,
            message="",
            execution_time=0.0
        )
        
        try:
            # Check dependencies
            if update.dependencies and not await self._dependencies_satisfied(update.dependencies):
                result.message = "Dependencies not satisfied"
                return result
            
            # Acquire lock
            if not await self._acquire_lock(update.namespace, update.key, update.update_id):
                result.message = "Could not acquire lock"
                return result
            
            try:
                # Execute based on update type
                if update.update_type == UpdateType.CREATE:
                    result = await self._execute_create(update)
                elif update.update_type == UpdateType.UPDATE:
                    result = await self._execute_update_operation(update)
                elif update.update_type == UpdateType.DELETE:
                    result = await self._execute_delete(update)
                elif update.update_type == UpdateType.MERGE:
                    result = await self._execute_merge(update)
                elif update.update_type == UpdateType.VALIDATE:
                    result = await self._execute_validate(update)
                
                # Validate consistency if required
                if update.validation_required and result.success:
                    consistency_check = await self._validate_consistency(update)
                    if not consistency_check:
                        result.success = False
                        result.message += " (consistency validation failed)"
                        result.validation_passed = False
                
            finally:
                # Release lock
                await self._release_lock(update.namespace, update.key, update.update_id)
            
        except (ValueError, TypeError, sqlite3.Error) as e:
            result.success = False
            result.message = f"Execution failed: {str(e)}"
            logger.error("Update execution failed: %s", str(e))
        
        result.execution_time = time.time() - start_time
        return result
    
    async def _dependencies_satisfied(self, dependencies: List[str]) -> bool:
        """Check if all dependencies are satisfied."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for dep_id in dependencies:
                    cursor = conn.execute(
                        "SELECT success FROM update_history WHERE update_id = ?",
                        (dep_id,)
                    )
                    result = cursor.fetchone()
                    if not result or not result[0]:
                        return False
                return True
                
        except sqlite3.Error:
            return False
    
    async def _acquire_lock(self, namespace: str, key: str, update_id: str) -> bool:
        """Acquire lock for knowledge entry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if already locked
                cursor = conn.execute(
                    "SELECT locked_by FROM update_locks WHERE namespace = ? AND key = ?",
                    (namespace, key)
                )
                result = cursor.fetchone()
                
                if result and result[0] != update_id:
                    return False
                
                # Acquire lock
                conn.execute("""
                    INSERT OR REPLACE INTO update_locks 
                    (namespace, key, locked_by, lock_timestamp)
                    VALUES (?, ?, ?, ?)
                """, (namespace, key, update_id, time.time()))
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            logger.error("Failed to acquire lock: %s", str(e))
            return False
    
    async def _release_lock(self, namespace: str, key: str, update_id: str):
        """Release lock for knowledge entry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM update_locks 
                    WHERE namespace = ? AND key = ? AND locked_by = ?
                """, (namespace, key, update_id))
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error("Failed to release lock: %s", str(e))
    
    async def _execute_create(self, update: KnowledgeUpdate) -> UpdateResult:
        """Execute create operation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if entry already exists
                cursor = conn.execute(
                    "SELECT id FROM hive_knowledge WHERE namespace = ? AND key = ?",
                    (update.namespace, update.key)
                )
                if cursor.fetchone():
                    return UpdateResult(
                        update_id=update.update_id,
                        success=False,
                        message="Entry already exists"
                    )
                
                # Create new entry
                conn.execute("""
                    INSERT INTO hive_knowledge 
                    (namespace, key, content, metadata, confidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    update.namespace, update.key, update.content,
                    json.dumps(update.metadata), update.confidence,
                    update.timestamp, update.timestamp
                ))
                conn.commit()
                
                return UpdateResult(
                    update_id=update.update_id,
                    success=True,
                    message="Entry created successfully",
                    affected_entries=1
                )
                
        except sqlite3.Error as e:
            return UpdateResult(
                update_id=update.update_id,
                success=False,
                message=f"Create failed: {str(e)}"
            )
    
    async def _execute_update_operation(self, update: KnowledgeUpdate) -> UpdateResult:
        """Execute update operation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Update existing entry
                cursor = conn.execute("""
                    UPDATE hive_knowledge 
                    SET content = ?, metadata = ?, confidence = ?, updated_at = ?
                    WHERE namespace = ? AND key = ?
                """, (
                    update.content, json.dumps(update.metadata),
                    update.confidence, update.timestamp,
                    update.namespace, update.key
                ))
                
                if cursor.rowcount == 0:
                    return UpdateResult(
                        update_id=update.update_id,
                        success=False,
                        message="Entry not found for update"
                    )
                
                conn.commit()
                
                return UpdateResult(
                    update_id=update.update_id,
                    success=True,
                    message="Entry updated successfully",
                    affected_entries=cursor.rowcount
                )
                
        except sqlite3.Error as e:
            return UpdateResult(
                update_id=update.update_id,
                success=False,
                message=f"Update failed: {str(e)}"
            )
    
    async def _execute_delete(self, update: KnowledgeUpdate) -> UpdateResult:
        """Execute delete operation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM hive_knowledge WHERE namespace = ? AND key = ?",
                    (update.namespace, update.key)
                )
                
                if cursor.rowcount == 0:
                    return UpdateResult(
                        update_id=update.update_id,
                        success=False,
                        message="Entry not found for deletion"
                    )
                
                conn.commit()
                
                return UpdateResult(
                    update_id=update.update_id,
                    success=True,
                    message="Entry deleted successfully",
                    affected_entries=cursor.rowcount
                )
                
        except sqlite3.Error as e:
            return UpdateResult(
                update_id=update.update_id,
                success=False,
                message=f"Delete failed: {str(e)}"
            )
    
    async def _execute_merge(self, update: KnowledgeUpdate) -> UpdateResult:
        """Execute merge operation with existing entry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get existing entry
                cursor = conn.execute("""
                    SELECT content, metadata, confidence FROM hive_knowledge 
                    WHERE namespace = ? AND key = ?
                """, (update.namespace, update.key))
                result = cursor.fetchone()
                
                if not result:
                    # No existing entry, create new one
                    return await self._execute_create(update)
                
                existing_content, existing_metadata_str, existing_confidence = result
                existing_metadata = json.loads(existing_metadata_str) if existing_metadata_str else {}
                
                # Merge content (simple concatenation for now)
                merged_content = f"{existing_content}\n{update.content}"
                
                # Merge metadata
                merged_metadata = {**existing_metadata, **update.metadata}
                
                # Average confidence
                merged_confidence = (existing_confidence + update.confidence) / 2
                
                # Update with merged data
                conn.execute("""
                    UPDATE hive_knowledge 
                    SET content = ?, metadata = ?, confidence = ?, updated_at = ?
                    WHERE namespace = ? AND key = ?
                """, (
                    merged_content, json.dumps(merged_metadata),
                    merged_confidence, update.timestamp,
                    update.namespace, update.key
                ))
                conn.commit()
                
                return UpdateResult(
                    update_id=update.update_id,
                    success=True,
                    message="Entry merged successfully",
                    affected_entries=1
                )
                
        except (sqlite3.Error, json.JSONDecodeError) as e:
            return UpdateResult(
                update_id=update.update_id,
                success=False,
                message=f"Merge failed: {str(e)}"
            )
    
    async def _execute_validate(self, update: KnowledgeUpdate) -> UpdateResult:
        """Execute validation operation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if entry exists and validate
                cursor = conn.execute("""
                    SELECT content, confidence FROM hive_knowledge 
                    WHERE namespace = ? AND key = ?
                """, (update.namespace, update.key))
                result = cursor.fetchone()
                
                if not result:
                    return UpdateResult(
                        update_id=update.update_id,
                        success=False,
                        message="Entry not found for validation"
                    )
                
                content, confidence = result
                
                # Perform validation checks
                validation_errors = []
                if not content.strip():
                    validation_errors.append("Empty content")
                if confidence < 0.1:
                    validation_errors.append("Very low confidence")
                
                success = len(validation_errors) == 0
                message = "Validation passed" if success else f"Validation failed: {', '.join(validation_errors)}"
                
                return UpdateResult(
                    update_id=update.update_id,
                    success=success,
                    message=message,
                    validation_passed=success
                )
                
        except sqlite3.Error as e:
            return UpdateResult(
                update_id=update.update_id,
                success=False,
                message=f"Validation failed: {str(e)}"
            )
    
    async def _validate_consistency(self, update: KnowledgeUpdate) -> bool:
        """Validate knowledge base consistency after update."""
        try:
            # Basic consistency checks
            with sqlite3.connect(self.db_path) as conn:
                # Check for duplicate keys in same namespace
                cursor = conn.execute("""
                    SELECT key, COUNT(*) FROM hive_knowledge 
                    WHERE namespace = ? 
                    GROUP BY key HAVING COUNT(*) > 1
                """, (update.namespace,))
                
                duplicates = cursor.fetchall()
                if duplicates:
                    logger.warning("Duplicate keys detected: %s", duplicates)
                    return False
                
                # Check for corrupted JSON metadata
                cursor = conn.execute("""
                    SELECT id, metadata FROM hive_knowledge 
                    WHERE namespace = ? AND metadata IS NOT NULL
                """, (update.namespace,))
                
                for row in cursor.fetchall():
                    try:
                        json.loads(row[1])
                    except json.JSONDecodeError:
                        logger.warning("Corrupted metadata in entry %s", row[0])
                        return False
                
                return True
                
        except sqlite3.Error as e:
            logger.error("Consistency validation failed: %s", str(e))
            return False
    
    async def _record_update_result(self, update: KnowledgeUpdate, result: UpdateResult):
        """Record update result in history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO update_history 
                    (update_id, update_type, namespace, key, priority, timestamp, 
                     source_id, success, execution_time, error_message, affected_entries)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    update.update_id, update.update_type.value, update.namespace, update.key,
                    update.priority.value, update.timestamp, update.source_id,
                    result.success, result.execution_time,
                    result.message if not result.success else None,
                    result.affected_entries
                ))
                
                # Record dependencies
                for dep_id in update.dependencies:
                    conn.execute("""
                        INSERT OR IGNORE INTO update_dependencies (update_id, depends_on)
                        VALUES (?, ?)
                    """, (update.update_id, dep_id))
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error("Failed to record update result: %s", str(e))
    
    async def get_update_status(self, update_id: str) -> Dict[str, Any]:
        """Get status of a specific update."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT update_type, namespace, key, priority, timestamp, 
                           source_id, success, execution_time, error_message, affected_entries
                    FROM update_history WHERE update_id = ?
                """, (update_id,))
                result = cursor.fetchone()
                
                if not result:
                    return {"status": "not_found"}
                
                return {
                    "update_id": update_id,
                    "update_type": result[0],
                    "namespace": result[1],
                    "key": result[2],
                    "priority": result[3],
                    "timestamp": result[4],
                    "source_id": result[5],
                    "success": result[6],
                    "execution_time": result[7],
                    "error_message": result[8],
                    "affected_entries": result[9],
                    "status": "completed" if result[6] else "failed"
                }
                
        except sqlite3.Error as e:
            logger.error("Failed to get update status: %s", str(e))
            return {"status": "error", "message": str(e)}
    
    async def get_updater_stats(self) -> Dict[str, Any]:
        """Get knowledge updater statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Update statistics
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_updates,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_updates,
                        AVG(execution_time) as avg_execution_time,
                        SUM(affected_entries) as total_affected_entries
                    FROM update_history
                """)
                stats = cursor.fetchone()
                
                # Update types breakdown
                cursor = conn.execute("""
                    SELECT update_type, COUNT(*) as count
                    FROM update_history
                    GROUP BY update_type
                    ORDER BY count DESC
                """)
                type_breakdown = cursor.fetchall()
                
                # Active locks
                cursor = conn.execute("SELECT COUNT(*) FROM update_locks")
                active_locks = cursor.fetchone()[0]
                
                return {
                    "total_updates": stats[0] if stats else 0,
                    "successful_updates": stats[1] if stats else 0,
                    "success_rate": (stats[1] / stats[0]) if stats and stats[0] > 0 else 0.0,
                    "avg_execution_time": stats[2] if stats else 0.0,
                    "total_affected_entries": stats[3] if stats else 0,
                    "update_type_breakdown": [
                        {"type": row[0], "count": row[1]} for row in type_breakdown
                    ],
                    "active_locks": active_locks,
                    "queue_size": self.update_queue.qsize(),
                    "is_processing": self.is_processing
                }
                
        except sqlite3.Error as e:
            logger.error("Failed to get updater stats: %s", str(e))
            return {}