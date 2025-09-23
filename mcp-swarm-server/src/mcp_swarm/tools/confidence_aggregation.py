"""
Confidence Aggregation System for Hive Mind Intelligence

Implements uncertainty quantification and confidence scoring for knowledge and decisions.
Provides reliable confidence estimates for multi-source information and collective decisions.

According to orchestrator routing:
- Primary: hive_mind_specialist.md
- Secondary: memory_management_specialist.md, mcp_specialist.md
"""
import logging
import math
import statistics
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import sqlite3

logger = logging.getLogger(__name__)

class ConfidenceMethod(Enum):
    """Available confidence aggregation methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    BAYESIAN_UPDATE = "bayesian_update"
    ENTROPY_BASED = "entropy_based"
    CONSENSUS_BASED = "consensus_based"
    BETA_DISTRIBUTION = "beta_distribution"
    TRUST_PROPAGATION = "trust_propagation"

@dataclass
class ConfidenceSource:
    """Represents a source with confidence information"""
    source_id: str
    confidence: float
    weight: float = 1.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = None  # type: ignore
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class AggregatedConfidence:
    """Result of confidence aggregation"""
    final_confidence: float
    uncertainty: float
    method_used: str
    source_count: int
    weight_distribution: Dict[str, float]
    meta_confidence: float  # Confidence in the confidence estimate
    aggregation_timestamp: float = 0.0
    
    def __post_init__(self):
        if self.aggregation_timestamp == 0.0:
            self.aggregation_timestamp = time.time()

class ConfidenceAggregationSystem:
    """
    Advanced confidence aggregation system for uncertainty quantification.
    
    Features:
    - Multiple aggregation methods (weighted average, Bayesian, entropy-based)
    - Source reliability tracking and trust propagation
    - Uncertainty quantification with meta-confidence
    - Temporal confidence decay modeling
    - Historical confidence validation and learning
    """
    
    def __init__(self, db_path: str):
        """
        Initialize confidence aggregation system.
        
        Args:
            db_path: Path to SQLite database for confidence tracking
        """
        self.db_path = db_path
        self.source_reliability = {}  # Track source reliability over time
        self.confidence_history = {}  # Historical confidence tracking
        self._initialize_confidence_tables()
    
    def _initialize_confidence_tables(self):
        """Initialize database tables for confidence tracking."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Table for source reliability tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS source_reliability (
                        source_id TEXT PRIMARY KEY,
                        accuracy_score REAL DEFAULT 0.5,
                        prediction_count INTEGER DEFAULT 0,
                        correct_predictions INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        reliability_metadata TEXT
                    )
                """)
                
                # Table for confidence history
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS confidence_history (
                        id INTEGER PRIMARY KEY,
                        aggregation_id TEXT,
                        method_used TEXT,
                        sources_count INTEGER,
                        final_confidence REAL,
                        actual_outcome REAL,
                        accuracy_error REAL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error("Failed to initialize confidence tables: %s", str(e))
    
    async def aggregate_confidence(
        self,
        sources: List[ConfidenceSource],
        method: ConfidenceMethod = ConfidenceMethod.WEIGHTED_AVERAGE,
        include_temporal_decay: bool = True,
        uncertainty_threshold: float = 0.1
    ) -> AggregatedConfidence:
        """
        Aggregate confidence from multiple sources.
        
        Args:
            sources: List of confidence sources
            method: Aggregation method to use
            include_temporal_decay: Whether to apply temporal decay
            uncertainty_threshold: Threshold for high uncertainty warning
            
        Returns:
            Aggregated confidence result
        """
        if not sources:
            return AggregatedConfidence(
                final_confidence=0.0,
                uncertainty=1.0,
                method_used=method.value,
                source_count=0,
                weight_distribution={},
                meta_confidence=0.0
            )
        
        try:
            # Apply temporal decay if requested
            if include_temporal_decay:
                sources = await self._apply_temporal_decay(sources)
            
            # Update source weights based on reliability
            sources = await self._update_source_weights(sources)
            
            # Choose aggregation method
            if method == ConfidenceMethod.WEIGHTED_AVERAGE:
                result = await self._weighted_average_aggregation(sources)
            elif method == ConfidenceMethod.BAYESIAN_UPDATE:
                result = await self._bayesian_aggregation(sources)
            elif method == ConfidenceMethod.ENTROPY_BASED:
                result = await self._entropy_aggregation(sources)
            elif method == ConfidenceMethod.CONSENSUS_BASED:
                result = await self._consensus_aggregation(sources)
            elif method == ConfidenceMethod.BETA_DISTRIBUTION:
                result = await self._beta_distribution_aggregation(sources)
            elif method == ConfidenceMethod.TRUST_PROPAGATION:
                result = await self._trust_propagation_aggregation(sources)
            else:
                # Fallback to weighted average
                result = await self._weighted_average_aggregation(sources)
            
            # Calculate meta-confidence
            result.meta_confidence = await self._calculate_meta_confidence(sources, result)
            
            # Log high uncertainty cases
            if result.uncertainty > uncertainty_threshold:
                logger.warning(
                    "High uncertainty detected: %.3f for %d sources",
                    result.uncertainty, len(sources)
                )
            
            return result
            
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.error("Confidence aggregation failed: %s", str(e))
            # Return default low confidence result
            return AggregatedConfidence(
                final_confidence=0.1,
                uncertainty=0.9,
                method_used=method.value,
                source_count=len(sources),
                weight_distribution={},
                meta_confidence=0.1
            )
    
    async def _apply_temporal_decay(self, sources: List[ConfidenceSource]) -> List[ConfidenceSource]:
        """Apply temporal decay to source confidences."""
        current_time = time.time()
        decay_rate = 0.1  # Confidence decay rate per day
        
        decayed_sources = []
        for source in sources:
            # Calculate time difference in days
            time_diff_days = (current_time - source.timestamp) / (24 * 3600)
            
            # Apply exponential decay
            decay_factor = math.exp(-decay_rate * time_diff_days)
            decayed_confidence = source.confidence * decay_factor
            
            # Create new source with decayed confidence
            decayed_source = ConfidenceSource(
                source_id=source.source_id,
                confidence=decayed_confidence,
                weight=source.weight,
                timestamp=source.timestamp,
                metadata={**source.metadata, 'temporal_decay_applied': decay_factor}
            )
            decayed_sources.append(decayed_source)
        
        return decayed_sources
    
    async def _update_source_weights(self, sources: List[ConfidenceSource]) -> List[ConfidenceSource]:
        """Update source weights based on historical reliability."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                updated_sources = []
                
                for source in sources:
                    # Get source reliability
                    cursor = conn.execute(
                        "SELECT accuracy_score FROM source_reliability WHERE source_id = ?",
                        (source.source_id,)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        reliability = result[0]
                        # Adjust weight based on reliability
                        adjusted_weight = source.weight * reliability
                    else:
                        # New source, use default weight
                        adjusted_weight = source.weight * 0.5
                    
                    updated_source = ConfidenceSource(
                        source_id=source.source_id,
                        confidence=source.confidence,
                        weight=adjusted_weight,
                        timestamp=source.timestamp,
                        metadata={**source.metadata, 'reliability_adjusted': True}
                    )
                    updated_sources.append(updated_source)
                
                return updated_sources
                
        except sqlite3.Error as e:
            logger.warning("Failed to update source weights: %s", str(e))
            return sources
    
    async def _weighted_average_aggregation(self, sources: List[ConfidenceSource]) -> AggregatedConfidence:
        """Aggregate using weighted average method."""
        total_weight = sum(s.weight for s in sources)
        if total_weight == 0:
            total_weight = len(sources)
        
        # Calculate weighted average
        weighted_sum = sum(s.confidence * s.weight for s in sources)
        final_confidence = weighted_sum / total_weight
        
        # Calculate uncertainty (weighted standard deviation)
        variance = sum(
            s.weight * ((s.confidence - final_confidence) ** 2) 
            for s in sources
        ) / total_weight
        uncertainty = math.sqrt(variance)
        
        # Weight distribution
        weight_dist = {
            s.source_id: s.weight / total_weight 
            for s in sources
        }
        
        return AggregatedConfidence(
            final_confidence=final_confidence,
            uncertainty=uncertainty,
            method_used=ConfidenceMethod.WEIGHTED_AVERAGE.value,
            source_count=len(sources),
            weight_distribution=weight_dist,
            meta_confidence=0.0  # Will be calculated later
        )
    
    async def _bayesian_aggregation(self, sources: List[ConfidenceSource]) -> AggregatedConfidence:
        """Aggregate using Bayesian updating method."""
        # Start with uniform prior
        prior_alpha = 1.0
        prior_beta = 1.0
        
        # Update with each source
        posterior_alpha = prior_alpha
        posterior_beta = prior_beta
        
        for source in sources:
            # Convert confidence to pseudo-observations
            observations = max(1, int(source.weight * 10))  # Scale weight to observation count
            successes = int(observations * source.confidence)
            failures = observations - successes
            
            # Bayesian update
            posterior_alpha += successes
            posterior_beta += failures
        
        # Calculate final confidence (mean of beta distribution)
        final_confidence = posterior_alpha / (posterior_alpha + posterior_beta)
        
        # Calculate uncertainty (variance of beta distribution)
        total = posterior_alpha + posterior_beta
        variance = (posterior_alpha * posterior_beta) / ((total ** 2) * (total + 1))
        uncertainty = math.sqrt(variance)
        
        return AggregatedConfidence(
            final_confidence=final_confidence,
            uncertainty=uncertainty,
            method_used=ConfidenceMethod.BAYESIAN_UPDATE.value,
            source_count=len(sources),
            weight_distribution={s.source_id: s.weight / sum(s.weight for s in sources) for s in sources},
            meta_confidence=0.0
        )
    
    async def _entropy_aggregation(self, sources: List[ConfidenceSource]) -> AggregatedConfidence:
        """Aggregate using entropy-based method."""
        if not sources:
            return AggregatedConfidence(0.0, 1.0, ConfidenceMethod.ENTROPY_BASED.value, 0, {}, 0.0)
        
        # Calculate entropy for each source
        entropies = []
        for source in sources:
            p = source.confidence
            if p == 0 or p == 1:
                entropy = 0
            else:
                entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
            entropies.append(entropy)
        
        # Weight sources inversely by entropy (lower entropy = higher weight)
        max_entropy = 1.0  # Maximum possible entropy for binary case
        entropy_weights = [(max_entropy - e) for e in entropies]
        total_entropy_weight = sum(entropy_weights)
        
        if total_entropy_weight == 0:
            # All sources have maximum entropy, use equal weights
            final_confidence = statistics.mean(s.confidence for s in sources)
            uncertainty = statistics.stdev(s.confidence for s in sources) if len(sources) > 1 else 0.5
        else:
            # Weighted average by inverse entropy
            final_confidence = sum(
                s.confidence * w for s, w in zip(sources, entropy_weights)
            ) / total_entropy_weight
            
            # Calculate uncertainty based on entropy distribution
            avg_entropy = statistics.mean(entropies)
            uncertainty = avg_entropy / max_entropy  # Normalize to [0, 1]
        
        return AggregatedConfidence(
            final_confidence=final_confidence,
            uncertainty=uncertainty,
            method_used=ConfidenceMethod.ENTROPY_BASED.value,
            source_count=len(sources),
            weight_distribution={
                s.source_id: w / total_entropy_weight if total_entropy_weight > 0 else 1.0 / len(sources)
                for s, w in zip(sources, entropy_weights)
            },
            meta_confidence=0.0
        )
    
    async def _consensus_aggregation(self, sources: List[ConfidenceSource]) -> AggregatedConfidence:
        """Aggregate using consensus-based method."""
        confidences = [s.confidence for s in sources]
        
        # Find consensus by clustering around modes
        # Simple implementation: use median as consensus center
        median_confidence = statistics.median(confidences)
        
        # Calculate agreement (how close sources are to consensus)
        deviations = [abs(c - median_confidence) for c in confidences]
        avg_deviation = statistics.mean(deviations)
        
        # Final confidence weighted by agreement
        agreement_weights = [1.0 / (1.0 + dev) for dev in deviations]
        total_agreement_weight = sum(agreement_weights)
        
        final_confidence = sum(
            s.confidence * w for s, w in zip(sources, agreement_weights)
        ) / total_agreement_weight
        
        # Uncertainty based on disagreement
        uncertainty = min(1.0, avg_deviation * 2)  # Scale and cap at 1.0
        
        return AggregatedConfidence(
            final_confidence=final_confidence,
            uncertainty=uncertainty,
            method_used=ConfidenceMethod.CONSENSUS_BASED.value,
            source_count=len(sources),
            weight_distribution={
                s.source_id: w / total_agreement_weight
                for s, w in zip(sources, agreement_weights)
            },
            meta_confidence=0.0
        )
    
    async def _beta_distribution_aggregation(self, sources: List[ConfidenceSource]) -> AggregatedConfidence:
        """Aggregate using Beta distribution modeling."""
        # Convert each confidence to beta distribution parameters
        total_alpha = 0.0
        total_beta = 0.0
        
        for source in sources:
            # Estimate alpha and beta from confidence and weight
            # Higher weight means more "observations"
            n_observations = max(2, source.weight * 10)
            alpha = source.confidence * n_observations
            beta = (1 - source.confidence) * n_observations
            
            total_alpha += alpha
            total_beta += beta
        
        # Final confidence is the mean of the combined beta distribution
        final_confidence = total_alpha / (total_alpha + total_beta)
        
        # Uncertainty is the standard deviation of the beta distribution
        total_params = total_alpha + total_beta
        variance = (total_alpha * total_beta) / ((total_params ** 2) * (total_params + 1))
        uncertainty = math.sqrt(variance)
        
        return AggregatedConfidence(
            final_confidence=final_confidence,
            uncertainty=uncertainty,
            method_used=ConfidenceMethod.BETA_DISTRIBUTION.value,
            source_count=len(sources),
            weight_distribution={s.source_id: s.weight / sum(s.weight for s in sources) for s in sources},
            meta_confidence=0.0
        )
    
    async def _trust_propagation_aggregation(self, sources: List[ConfidenceSource]) -> AggregatedConfidence:
        """Aggregate using trust propagation method."""
        # Simple trust propagation: sources with higher historical accuracy get more trust
        trust_scores = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for source in sources:
                    cursor = conn.execute(
                        "SELECT accuracy_score FROM source_reliability WHERE source_id = ?",
                        (source.source_id,)
                    )
                    result = cursor.fetchone()
                    trust_scores[source.source_id] = result[0] if result else 0.5
        except sqlite3.Error:
            # Fallback to equal trust
            trust_scores = {s.source_id: 0.5 for s in sources}
        
        # Weight by trust scores
        total_trust = sum(trust_scores[s.source_id] * s.weight for s in sources)
        if total_trust == 0:
            total_trust = len(sources)
        
        final_confidence = sum(
            s.confidence * trust_scores[s.source_id] * s.weight
            for s in sources
        ) / total_trust
        
        # Uncertainty based on trust distribution
        trust_values = list(trust_scores.values())
        trust_variance = statistics.variance(trust_values) if len(trust_values) > 1 else 0.0
        uncertainty = math.sqrt(trust_variance)
        
        return AggregatedConfidence(
            final_confidence=final_confidence,
            uncertainty=uncertainty,
            method_used=ConfidenceMethod.TRUST_PROPAGATION.value,
            source_count=len(sources),
            weight_distribution={
                s.source_id: (trust_scores[s.source_id] * s.weight) / total_trust
                for s in sources
            },
            meta_confidence=0.0
        )
    
    async def _calculate_meta_confidence(
        self, 
        sources: List[ConfidenceSource], 
        result: AggregatedConfidence
    ) -> float:
        """Calculate confidence in the confidence estimate itself."""
        # Factors that increase meta-confidence:
        # 1. More sources
        # 2. Agreement between sources
        # 3. High reliability of sources
        # 4. Recent timestamps
        
        if not sources:
            return 0.0
        
        # Source count factor (more sources = higher meta-confidence)
        count_factor = min(1.0, len(sources) / 10.0)  # Cap at 10 sources
        
        # Agreement factor (lower uncertainty = higher meta-confidence)
        agreement_factor = max(0.0, 1.0 - result.uncertainty)
        
        # Reliability factor (average source reliability)
        try:
            with sqlite3.connect(self.db_path) as conn:
                reliabilities = []
                for source in sources:
                    cursor = conn.execute(
                        "SELECT accuracy_score FROM source_reliability WHERE source_id = ?",
                        (source.source_id,)
                    )
                    result_row = cursor.fetchone()
                    reliabilities.append(result_row[0] if result_row else 0.5)
                
                reliability_factor = statistics.mean(reliabilities)
        except sqlite3.Error:
            reliability_factor = 0.5
        
        # Recency factor (newer sources = higher meta-confidence)
        current_time = time.time()
        recency_scores = []
        for source in sources:
            age_hours = (current_time - source.timestamp) / 3600
            recency_score = max(0.0, 1.0 - (age_hours / 24))  # Decay over 24 hours
            recency_scores.append(recency_score)
        
        recency_factor = statistics.mean(recency_scores)
        
        # Combine factors
        meta_confidence = (
            count_factor * 0.3 +
            agreement_factor * 0.3 +
            reliability_factor * 0.2 +
            recency_factor * 0.2
        )
        
        return min(1.0, meta_confidence)
    
    async def update_source_reliability(
        self, 
        source_id: str, 
        actual_outcome: float, 
        predicted_confidence: float
    ):
        """Update source reliability based on actual outcomes."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Calculate prediction accuracy
                accuracy = 1.0 - abs(actual_outcome - predicted_confidence)
                
                # Get current reliability stats
                cursor = conn.execute(
                    "SELECT accuracy_score, prediction_count, correct_predictions FROM source_reliability WHERE source_id = ?",
                    (source_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    # Update existing record
                    current_accuracy, count, correct = result
                    new_count = count + 1
                    new_correct = correct + (1 if accuracy > 0.8 else 0)
                    new_accuracy = (current_accuracy * count + accuracy) / new_count
                    
                    conn.execute("""
                        UPDATE source_reliability 
                        SET accuracy_score = ?, prediction_count = ?, correct_predictions = ?, last_updated = CURRENT_TIMESTAMP
                        WHERE source_id = ?
                    """, (new_accuracy, new_count, new_correct, source_id))
                else:
                    # Create new record
                    conn.execute("""
                        INSERT INTO source_reliability (source_id, accuracy_score, prediction_count, correct_predictions)
                        VALUES (?, ?, 1, ?)
                    """, (source_id, accuracy, 1 if accuracy > 0.8 else 0))
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error("Failed to update source reliability: %s", str(e))
    
    async def get_confidence_stats(self) -> Dict[str, Any]:
        """Get confidence aggregation system statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Source reliability stats
                cursor = conn.execute("""
                    SELECT COUNT(*) as total_sources, 
                           AVG(accuracy_score) as avg_accuracy,
                           MIN(accuracy_score) as min_accuracy,
                           MAX(accuracy_score) as max_accuracy
                    FROM source_reliability
                """)
                reliability_stats = cursor.fetchone()
                
                # Confidence history stats
                cursor = conn.execute("""
                    SELECT method_used, COUNT(*) as usage_count, AVG(accuracy_error) as avg_error
                    FROM confidence_history
                    GROUP BY method_used
                    ORDER BY usage_count DESC
                """)
                method_stats = cursor.fetchall()
                
                return {
                    'total_sources': reliability_stats[0] if reliability_stats else 0,
                    'average_accuracy': reliability_stats[1] if reliability_stats else 0.0,
                    'accuracy_range': {
                        'min': reliability_stats[2] if reliability_stats else 0.0,
                        'max': reliability_stats[3] if reliability_stats else 0.0
                    },
                    'method_usage': [
                        {'method': row[0], 'count': row[1], 'avg_error': row[2]}
                        for row in method_stats
                    ],
                    'available_methods': [method.value for method in ConfidenceMethod]
                }
                
        except sqlite3.Error as e:
            logger.error("Failed to get confidence stats: %s", str(e))
            return {}