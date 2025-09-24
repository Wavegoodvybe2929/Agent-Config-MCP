"""
Adaptive Learning Engine for MCP Swarm Intelligence Server

This module implements machine learning-based adaptive system evolution that learns
from experience and improves swarm coordination patterns over time.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sqlite3

# Machine learning imports for adaptive algorithms
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, mean_squared_error
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = None
    logging.warning("Scikit-learn not available. Using simple heuristics for adaptive learning.")

logger = logging.getLogger(__name__)

class LearningMode(Enum):
    """Learning mode configurations"""
    CONTINUOUS = "continuous"
    BATCH = "batch"
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"

class PatternType(Enum):
    """Types of patterns that can be learned"""
    TASK_ASSIGNMENT = "task_assignment"
    COORDINATION = "coordination"
    RESOURCE_ALLOCATION = "resource_allocation"
    CONFLICT_RESOLUTION = "conflict_resolution"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"

@dataclass
class LearningPattern:
    """Represents a learned pattern from swarm coordination"""
    id: str
    pattern_type: PatternType
    features: Dict[str, Any]
    outcome: Dict[str, Any]
    success_score: float
    frequency: int
    last_used: datetime
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class AdaptationEvent:
    """Records an adaptation event for learning"""
    timestamp: datetime
    event_type: str
    context: Dict[str, Any]
    decision_made: Dict[str, Any]
    outcome: Dict[str, Any]
    success_metrics: Dict[str, float]

class AdaptiveLearningEngine:
    """
    Machine learning-based adaptive system evolution that learns from experience
    and continuously improves swarm coordination patterns.
    """
    
    def __init__(
        self,
        database_path: str = "data/memory.db",
        learning_mode: LearningMode = LearningMode.CONTINUOUS,
        pattern_threshold: float = 0.7,
        adaptation_rate: float = 0.1
    ):
        self.database_path = Path(database_path)
        self.learning_mode = learning_mode
        self.pattern_threshold = pattern_threshold
        self.adaptation_rate = adaptation_rate
        
        # Learning state
        self.patterns: Dict[str, LearningPattern] = {}
        self.adaptation_history: List[AdaptationEvent] = []
        self.feature_scaler = StandardScaler() if ML_AVAILABLE else None
        
        # ML models
        self.success_classifier = None
        self.performance_regressor = None
        self.pattern_clusterer = None
        
        # Performance metrics
        self.learning_metrics = {
            'patterns_learned': 0,
            'adaptations_made': 0,
            'prediction_accuracy': 0.0,
            'improvement_rate': 0.0
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize machine learning models"""
        if ML_AVAILABLE:
            # Classification model for predicting success
            self.success_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Regression model for performance prediction
            self.performance_regressor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
            
            # Clustering for pattern discovery
            self.pattern_clusterer = DBSCAN(
                eps=0.3,
                min_samples=5
            )
        else:
            logger.info("Using simple heuristic models for adaptive learning")
    
    def _get_db_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    async def initialize_database(self):
        """Initialize database tables for adaptive learning"""
        conn = self._get_db_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    features TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    success_score REAL NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_used TIMESTAMP NOT NULL,
                    confidence REAL NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS adaptation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    event_type TEXT NOT NULL,
                    context TEXT NOT NULL,
                    decision_made TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    success_metrics TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_type_score 
                ON learning_patterns(pattern_type, success_score)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    async def learn_from_experience(
        self,
        context: Dict[str, Any],
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
        success_metrics: Dict[str, float]
    ) -> bool:
        """
        Learn from a coordination experience and update patterns
        
        Args:
            context: The situation context when decision was made
            decision: The decision that was made
            outcome: The actual outcome
            success_metrics: Metrics measuring success/failure
            
        Returns:
            True if learning resulted in pattern updates
        """
        try:
            # Create adaptation event
            event = AdaptationEvent(
                timestamp=datetime.now(),
                event_type=context.get('event_type', 'coordination'),
                context=context,
                decision_made=decision,
                outcome=outcome,
                success_metrics=success_metrics
            )
            
            # Extract pattern features
            features = self._extract_features(context, decision)
            pattern_type = self._classify_pattern_type(context)
            
            # Calculate overall success score
            success_score = self._calculate_success_score(success_metrics)
            
            # Generate pattern ID
            pattern_id = self._generate_pattern_id(features, pattern_type)
            
            # Update or create pattern
            updated = await self._update_or_create_pattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                features=features,
                outcome=outcome,
                success_score=success_score
            )
            
            # Store adaptation event
            await self._store_adaptation_event(event)
            
            # Update learning metrics
            self.learning_metrics['adaptations_made'] += 1
            if updated:
                self.learning_metrics['patterns_learned'] += 1
            
            # Trigger model retraining if needed
            if self.learning_mode == LearningMode.CONTINUOUS:
                await self._incremental_model_update(features, success_score, outcome)
            
            logger.info(f"Learned from experience: {pattern_type.value}, success: {success_score:.3f}")
            return updated
            
        except Exception as e:
            logger.error(f"Error learning from experience: {e}")
            return False
    
    async def predict_success(
        self,
        context: Dict[str, Any],
        proposed_decision: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Predict success probability for a proposed decision
        
        Args:
            context: Current situation context
            proposed_decision: Decision being considered
            
        Returns:
            Tuple of (success_probability, confidence)
        """
        try:
            features = self._extract_features(context, proposed_decision)
            pattern_type = self._classify_pattern_type(context)
            
            # Find similar patterns
            similar_patterns = await self._find_similar_patterns(
                features, pattern_type, limit=10
            )
            
            if ML_AVAILABLE and self.success_classifier and hasattr(self.success_classifier, 'predict_proba'):
                # Use ML model for prediction if trained
                feature_vector = self._features_to_vector(features)
                if feature_vector is not None:
                    try:
                        prediction = self.success_classifier.predict_proba([feature_vector])
                        success_prob = prediction[0][1] if len(prediction[0]) > 1 else prediction[0][0]
                        confidence = max(prediction[0])
                        return float(success_prob), float(confidence)
                    except Exception as ml_error:
                        logger.warning(f"ML prediction failed: {ml_error}, falling back to pattern-based")
            
            # Fallback to pattern-based prediction
            if similar_patterns:
                weighted_success = sum(
                    pattern.success_score * pattern.frequency * pattern.confidence
                    for pattern in similar_patterns
                )
                total_weight = sum(
                    pattern.frequency * pattern.confidence
                    for pattern in similar_patterns
                )
                
                if total_weight > 0:
                    success_prob = weighted_success / total_weight
                    confidence = min(1.0, total_weight / 10.0)  # Normalize confidence
                    return success_prob, confidence
            
            # Default prediction for unknown patterns
            return 0.5, 0.1
            
        except Exception as e:
            logger.error(f"Error predicting success: {e}")
            return 0.5, 0.1
    
    async def recommend_adaptations(
        self,
        current_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Recommend adaptations based on learned patterns
        
        Args:
            current_context: Current system context
            
        Returns:
            List of recommended adaptations with confidence scores
        """
        try:
            recommendations = []
            
            # Analyze current context
            context_features = self._extract_context_features(current_context)
            
            # Find patterns with high success rates
            high_success_patterns = await self._get_high_success_patterns(
                min_success_score=0.8,
                min_frequency=3
            )
            
            # Generate recommendations based on successful patterns
            for pattern in high_success_patterns:
                similarity = self._calculate_feature_similarity(
                    context_features, pattern.features
                )
                
                if similarity > self.pattern_threshold:
                    recommendation = {
                        'type': 'pattern_application',
                        'pattern_id': pattern.id,
                        'pattern_type': pattern.pattern_type.value,
                        'recommended_action': pattern.outcome,
                        'expected_success': pattern.success_score,
                        'confidence': pattern.confidence * similarity,
                        'rationale': f"Similar context achieved {pattern.success_score:.1%} success rate"
                    }
                    recommendations.append(recommendation)
            
            # Sort by confidence and expected success
            recommendations.sort(
                key=lambda x: x['confidence'] * x['expected_success'],
                reverse=True
            )
            
            return recommendations[:5]  # Top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    async def adapt_parameters(
        self,
        system_parameters: Dict[str, Any],
        performance_targets: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Adapt system parameters based on learned performance patterns
        
        Args:
            system_parameters: Current system parameters
            performance_targets: Target performance metrics
            
        Returns:
            Adapted parameters
        """
        try:
            adapted_params = system_parameters.copy()
            
            # Analyze parameter performance history
            param_patterns = await self._analyze_parameter_performance()
            
            for param_name, current_value in system_parameters.items():
                if param_name in param_patterns:
                    pattern_data = param_patterns[param_name]
                    
                    # Find optimal value based on historical performance
                    optimal_value = self._find_optimal_parameter_value(
                        pattern_data, performance_targets
                    )
                    
                    if optimal_value is not None:
                        # Gradual adaptation to prevent instability
                        new_value = current_value + (optimal_value - current_value) * self.adaptation_rate
                        adapted_params[param_name] = new_value
                        
                        logger.info(
                            f"Adapted parameter {param_name}: {current_value} → {new_value}"
                        )
            
            return adapted_params
            
        except Exception as e:
            logger.error(f"Error adapting parameters: {e}")
            return system_parameters
    
    def _extract_features(self, context: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from context and decision for pattern recognition"""
        features = {}
        
        # Context features
        features.update({
            f"context_{k}": v for k, v in context.items()
            if isinstance(v, (int, float, str, bool))
        })
        
        # Decision features
        features.update({
            f"decision_{k}": v for k, v in decision.items()
            if isinstance(v, (int, float, str, bool))
        })
        
        # Derived features
        features['context_complexity'] = len(context)
        features['decision_complexity'] = len(decision)
        features['timestamp'] = datetime.now().timestamp()
        
        return features
    
    def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from context only"""
        return {
            k: v for k, v in context.items()
            if isinstance(v, (int, float, str, bool))
        }
    
    def _classify_pattern_type(self, context: Dict[str, Any]) -> PatternType:
        """Classify the type of pattern based on context"""
        context_str = str(context).lower()
        if 'task' in context_str and 'assign' in context_str:
            return PatternType.TASK_ASSIGNMENT
        elif 'coordinat' in context_str:
            return PatternType.COORDINATION
        elif 'resource' in context_str:
            return PatternType.RESOURCE_ALLOCATION
        elif 'conflict' in context_str:
            return PatternType.CONFLICT_RESOLUTION
        else:
            return PatternType.PERFORMANCE_OPTIMIZATION
    
    def _calculate_success_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall success score from metrics"""
        if not metrics:
            return 0.5
        
        # Weight different metrics
        weights = {
            'completion_rate': 0.3,
            'quality_score': 0.3,
            'efficiency': 0.2,
            'user_satisfaction': 0.2
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            weight = weights.get(metric, 0.1)  # Default weight for unknown metrics
            weighted_sum += value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _generate_pattern_id(self, features: Dict[str, Any], pattern_type: PatternType) -> str:
        """Generate unique pattern ID based on features and type"""
        # Create a stable hash from features (excluding timestamp)
        stable_features = {k: v for k, v in features.items() if k != 'timestamp'}
        features_str = json.dumps(stable_features, sort_keys=True)
        feature_hash = hashlib.md5(features_str.encode()).hexdigest()[:8]
        return f"{pattern_type.value}_{feature_hash}"
    
    async def _update_or_create_pattern(
        self,
        pattern_id: str,
        pattern_type: PatternType,
        features: Dict[str, Any],
        outcome: Dict[str, Any],
        success_score: float
    ) -> bool:
        """Update existing pattern or create new one"""
        try:
            conn = self._get_db_connection()
            try:
                # Check if pattern exists
                cursor = conn.execute(
                    "SELECT * FROM learning_patterns WHERE id = ?",
                    (pattern_id,)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing pattern
                    new_frequency = existing['frequency'] + 1
                    # Weighted average of success scores
                    new_success = (
                        existing['success_score'] * existing['frequency'] + success_score
                    ) / new_frequency
                    
                    # Update confidence based on frequency and consistency
                    confidence = min(1.0, new_frequency / 10.0)
                    
                    conn.execute("""
                        UPDATE learning_patterns 
                        SET frequency = ?, success_score = ?, confidence = ?, last_used = ?
                        WHERE id = ?
                    """, (new_frequency, new_success, confidence, datetime.now().isoformat(), pattern_id))
                    
                    conn.commit()
                    return False  # Updated existing
                    
                else:
                    # Create new pattern
                    confidence = 0.1  # Low initial confidence
                    
                    conn.execute("""
                        INSERT INTO learning_patterns 
                        (id, pattern_type, features, outcome, success_score, frequency, 
                         last_used, confidence, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pattern_id,
                        pattern_type.value,
                        json.dumps(features),
                        json.dumps(outcome),
                        success_score,
                        1,
                        datetime.now().isoformat(),
                        confidence,
                        json.dumps({})
                    ))
                    
                    conn.commit()
                    return True  # Created new pattern
            finally:
                conn.close()
                    
        except Exception as e:
            logger.error(f"Error updating/creating pattern: {e}")
            return False
    
    async def _store_adaptation_event(self, event: AdaptationEvent):
        """Store adaptation event in database"""
        try:
            conn = self._get_db_connection()
            try:
                conn.execute("""
                    INSERT INTO adaptation_events 
                    (timestamp, event_type, context, decision_made, outcome, success_metrics)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    event.timestamp.isoformat(),
                    event.event_type,
                    json.dumps(event.context),
                    json.dumps(event.decision_made),
                    json.dumps(event.outcome),
                    json.dumps(event.success_metrics)
                ))
                conn.commit()
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Error storing adaptation event: {e}")
    
    def _features_to_vector(self, features: Dict[str, Any]) -> Optional[List[float]]:
        """Convert features to numeric vector for ML models"""
        if not ML_AVAILABLE:
            return None
        
        try:
            # Convert features to numeric values
            numeric_features = []
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    numeric_features.append(float(value))
                elif isinstance(value, bool):
                    numeric_features.append(1.0 if value else 0.0)
                elif isinstance(value, str):
                    # Simple string hashing
                    numeric_features.append(float(abs(hash(value)) % 1000))
            
            return numeric_features if numeric_features else None
            
        except Exception as e:
            logger.error(f"Error converting features to vector: {e}")
            return None
    
    async def _find_similar_patterns(
        self,
        features: Dict[str, Any],
        pattern_type: PatternType,
        limit: int = 10
    ) -> List[LearningPattern]:
        """Find similar patterns from database"""
        try:
            conn = self._get_db_connection()
            try:
                cursor = conn.execute("""
                    SELECT * FROM learning_patterns 
                    WHERE pattern_type = ? 
                    ORDER BY success_score DESC, confidence DESC
                    LIMIT ?
                """, (pattern_type.value, limit))
                
                rows = cursor.fetchall()
                patterns = []
                
                for row in rows:
                    pattern_features = json.loads(row['features'])
                    similarity = self._calculate_feature_similarity(features, pattern_features)
                    
                    if similarity > 0.3:  # Minimum similarity threshold
                        pattern = LearningPattern(
                            id=row['id'],
                            pattern_type=PatternType(row['pattern_type']),
                            features=pattern_features,
                            outcome=json.loads(row['outcome']),
                            success_score=row['success_score'],
                            frequency=row['frequency'],
                            last_used=datetime.fromisoformat(row['last_used']),
                            confidence=row['confidence'],
                            metadata=json.loads(row['metadata'] or '{}')
                        )
                        patterns.append(pattern)
                
                return sorted(patterns, key=lambda p: p.success_score * p.confidence, reverse=True)
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []
    
    def _calculate_feature_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between two feature sets"""
        try:
            common_keys = set(features1.keys()) & set(features2.keys())
            if not common_keys:
                return 0.0
            
            similarities = []
            for key in common_keys:
                val1, val2 = features1[key], features2[key]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Numeric similarity
                    max_val = max(abs(val1), abs(val2))
                    if max_val == 0:
                        similarities.append(1.0)
                    else:
                        similarity = 1.0 - abs(val1 - val2) / max_val
                        similarities.append(max(0.0, similarity))
                        
                elif isinstance(val1, str) and isinstance(val2, str):
                    # String similarity (exact match for now)
                    similarities.append(1.0 if val1 == val2 else 0.0)
                    
                elif isinstance(val1, bool) and isinstance(val2, bool):
                    # Boolean similarity
                    similarities.append(1.0 if val1 == val2 else 0.0)
            
            return sum(similarities) / len(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating feature similarity: {e}")
            return 0.0
    
    async def _get_high_success_patterns(
        self, 
        min_success_score: float = 0.8,
        min_frequency: int = 3
    ) -> List[LearningPattern]:
        """Get patterns with high success scores"""
        try:
            conn = self._get_db_connection()
            try:
                cursor = conn.execute("""
                    SELECT * FROM learning_patterns 
                    WHERE success_score >= ? AND frequency >= ?
                    ORDER BY success_score DESC, confidence DESC
                    LIMIT 20
                """, (min_success_score, min_frequency))
                
                rows = cursor.fetchall()
                patterns = []
                
                for row in rows:
                    pattern = LearningPattern(
                        id=row['id'],
                        pattern_type=PatternType(row['pattern_type']),
                        features=json.loads(row['features']),
                        outcome=json.loads(row['outcome']),
                        success_score=row['success_score'],
                        frequency=row['frequency'],
                        last_used=datetime.fromisoformat(row['last_used']),
                        confidence=row['confidence'],
                        metadata=json.loads(row['metadata'] or '{}')
                    )
                    patterns.append(pattern)
                
                return patterns
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Error getting high success patterns: {e}")
            return []
    
    async def _analyze_parameter_performance(self) -> Dict[str, Dict[str, Any]]:
        """Analyze historical parameter performance"""
        # Simplified implementation - would analyze parameter changes vs performance
        return {}
    
    def _find_optimal_parameter_value(
        self, 
        pattern_data: Dict[str, Any], 
        targets: Dict[str, float]
    ) -> Optional[float]:
        """Find optimal parameter value based on historical data"""
        # Simplified implementation - would use regression or optimization
        return None
    
    async def _incremental_model_update(
        self, 
        features: Dict[str, Any], 
        success_score: float,
        outcome: Dict[str, Any]
    ):
        """Incrementally update ML models with new data"""
        if not ML_AVAILABLE:
            return
        
        # In a production system, would implement online learning
        # For now, just log the update
        logger.debug("Incremental model update triggered")
    
    async def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status"""
        try:
            conn = self._get_db_connection()
            try:
                # Count patterns by type
                cursor = conn.execute("""
                    SELECT pattern_type, COUNT(*) as count, 
                           AVG(success_score) as avg_success,
                           AVG(confidence) as avg_confidence
                    FROM learning_patterns 
                    GROUP BY pattern_type
                """)
                pattern_stats = cursor.fetchall()
                
                # Recent adaptation events
                cursor = conn.execute("""
                    SELECT COUNT(*) as recent_adaptations
                    FROM adaptation_events
                    WHERE timestamp > datetime('now', '-24 hours')
                """)
                recent_adaptations_row = cursor.fetchone()
                recent_adaptations = recent_adaptations_row['recent_adaptations'] if recent_adaptations_row else 0
                
                return {
                    'learning_mode': self.learning_mode.value,
                    'pattern_stats': [dict(row) for row in pattern_stats],
                    'recent_adaptations': recent_adaptations,
                    'learning_metrics': self.learning_metrics,
                    'models_available': ML_AVAILABLE,
                    'total_patterns': len(self.patterns),
                    'pattern_threshold': self.pattern_threshold,
                    'adaptation_rate': self.adaptation_rate
                }
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Error getting learning status: {e}")
            return {
                'error': str(e),
                'learning_mode': self.learning_mode.value,
                'models_available': ML_AVAILABLE
            }