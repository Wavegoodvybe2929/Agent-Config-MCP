"""
Multi-Criteria Decision Analysis (MCDA) Module

This module implements TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
method for evaluating agent alternatives based on multiple criteria including capability match,
current load, success rate, response time, and expertise level.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Criterion:
    """Represents a decision criterion with weight and type."""
    name: str
    weight: float
    is_benefit: bool = True  # True for benefit criteria (higher is better), False for cost criteria (lower is better)
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class Alternative:
    """Represents a decision alternative with values for each criterion."""
    id: str
    values: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MCDAAnalyzer:
    """
    Multi-Criteria Decision Analysis for agent selection using TOPSIS method.
    
    TOPSIS is a compensatory decision-making method that considers the geometric distance
    to both positive ideal solution (best alternative) and negative ideal solution (worst alternative).
    """
    
    def __init__(self, custom_criteria: Optional[List[Criterion]] = None):
        """
        Initialize MCDA analyzer with criteria.
        
        Args:
            custom_criteria: Optional custom criteria list, uses default if None
        """
        if custom_criteria:
            self.criteria = custom_criteria
        else:
            self.criteria = [
                Criterion("capability_match", 0.30, True),      # Higher capability match is better
                Criterion("current_load", 0.20, False),        # Lower current load is better  
                Criterion("success_rate", 0.25, True),         # Higher success rate is better
                Criterion("response_time", 0.15, False),       # Lower response time is better
                Criterion("expertise_level", 0.10, True)       # Higher expertise is better
            ]
        
        # Validate criteria weights sum to 1.0
        total_weight = sum(criterion.weight for criterion in self.criteria)
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"Criteria weights sum to {total_weight}, normalizing to 1.0")
            for criterion in self.criteria:
                criterion.weight = criterion.weight / total_weight
    
    async def analyze_alternatives(
        self, 
        alternatives: List[Alternative],
        normalize_values: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Analyze alternatives using TOPSIS method.
        
        Args:
            alternatives: List of alternatives to evaluate
            normalize_values: Whether to normalize criterion values
            
        Returns:
            List of tuples (alternative_id, closeness_score) sorted by score descending
        """
        if not alternatives:
            return []
        
        if len(alternatives) == 1:
            return [(alternatives[0].id, 1.0)]
        
        try:
            # Build decision matrix
            decision_matrix = self._build_decision_matrix(alternatives)
            
            if decision_matrix.size == 0:
                logger.error("Empty decision matrix created")
                return [(alt.id, 0.0) for alt in alternatives]
            
            # Normalize decision matrix
            if normalize_values:
                normalized_matrix = self._normalize_matrix(decision_matrix)
            else:
                normalized_matrix = decision_matrix
            
            # Apply weights
            weighted_matrix = self._calculate_weighted_matrix(normalized_matrix)
            
            # Find ideal solutions
            positive_ideal, negative_ideal = self._find_ideal_solutions(weighted_matrix)
            
            # Calculate distances to ideal solutions
            positive_distances, negative_distances = self._calculate_distances(
                weighted_matrix, positive_ideal, negative_ideal
            )
            
            # Calculate relative closeness to ideal solution
            closeness_scores = self._calculate_relative_closeness(
                positive_distances, negative_distances
            )
            
            # Create results with alternative IDs
            results = list(zip([alt.id for alt in alternatives], closeness_scores))
            
            # Sort by closeness score (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"MCDA analysis completed for {len(alternatives)} alternatives")
            return results
            
        except Exception as e:
            logger.error(f"Error in MCDA analysis: {str(e)}")
            # Return alternatives with equal scores as fallback
            return [(alt.id, 0.5) for alt in alternatives]
    
    def _build_decision_matrix(self, alternatives: List[Alternative]) -> np.ndarray:
        """Build decision matrix from alternatives and criteria."""
        matrix = []
        
        for alternative in alternatives:
            row = []
            for criterion in self.criteria:
                value = alternative.values.get(criterion.name, 0.0)
                
                # Apply bounds if specified
                if criterion.min_value is not None:
                    value = max(value, criterion.min_value)
                if criterion.max_value is not None:
                    value = min(value, criterion.max_value)
                
                row.append(value)
            matrix.append(row)
        
        return np.array(matrix, dtype=float)
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize decision matrix using vector normalization.
        
        Each column (criterion) is normalized by dividing each element by the square root
        of the sum of squares of all elements in that column.
        """
        if matrix.size == 0:
            return matrix
        
        normalized = np.zeros_like(matrix)
        
        for j in range(matrix.shape[1]):
            column = matrix[:, j]
            column_norm = np.sqrt(np.sum(column ** 2))
            
            if column_norm > 0:
                normalized[:, j] = column / column_norm
            else:
                normalized[:, j] = column  # Keep zeros as is
        
        return normalized
    
    def _calculate_weighted_matrix(self, normalized: np.ndarray) -> np.ndarray:
        """Apply criterion weights to normalized matrix."""
        if normalized.size == 0:
            return normalized
        
        weights = np.array([criterion.weight for criterion in self.criteria])
        return normalized * weights
    
    def _find_ideal_solutions(self, weighted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find positive and negative ideal solutions.
        
        For benefit criteria: positive ideal = max, negative ideal = min
        For cost criteria: positive ideal = min, negative ideal = max
        """
        if weighted.size == 0:
            return np.array([]), np.array([])
        
        positive_ideal = np.zeros(weighted.shape[1])
        negative_ideal = np.zeros(weighted.shape[1])
        
        for j, criterion in enumerate(self.criteria):
            column = weighted[:, j]
            
            if criterion.is_benefit:
                positive_ideal[j] = np.max(column)
                negative_ideal[j] = np.min(column)
            else:  # Cost criterion
                positive_ideal[j] = np.min(column)
                negative_ideal[j] = np.max(column)
        
        return positive_ideal, negative_ideal
    
    def _calculate_distances(
        self, 
        weighted: np.ndarray, 
        positive_ideal: np.ndarray, 
        negative_ideal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Euclidean distances to ideal solutions."""
        if weighted.size == 0:
            return np.array([]), np.array([])
        
        positive_distances = np.sqrt(np.sum((weighted - positive_ideal) ** 2, axis=1))
        negative_distances = np.sqrt(np.sum((weighted - negative_ideal) ** 2, axis=1))
        
        return positive_distances, negative_distances
    
    def _calculate_relative_closeness(
        self, 
        positive_distances: np.ndarray, 
        negative_distances: np.ndarray
    ) -> np.ndarray:
        """
        Calculate relative closeness to ideal solution.
        
        Closeness = negative_distance / (positive_distance + negative_distance)
        Values range from 0 to 1, where 1 is best and 0 is worst.
        """
        if positive_distances.size == 0:
            return np.array([])
        
        # Handle division by zero
        total_distances = positive_distances + negative_distances
        
        # If both distances are zero, the alternative is at the ideal point
        closeness = np.where(
            total_distances == 0,
            1.0,  # Perfect score if at ideal point
            negative_distances / total_distances
        )
        
        return closeness
    
    def get_criteria_info(self) -> List[Dict[str, Any]]:
        """Get information about current criteria configuration."""
        return [
            {
                "name": criterion.name,
                "weight": criterion.weight,
                "type": "benefit" if criterion.is_benefit else "cost",
                "min_value": criterion.min_value,
                "max_value": criterion.max_value
            }
            for criterion in self.criteria
        ]
    
    def update_criteria_weights(self, weight_updates: Dict[str, float]) -> bool:
        """
        Update criteria weights.
        
        Args:
            weight_updates: Dictionary mapping criterion name to new weight
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Update weights
            for criterion in self.criteria:
                if criterion.name in weight_updates:
                    criterion.weight = weight_updates[criterion.name]
            
            # Normalize weights to sum to 1.0
            total_weight = sum(criterion.weight for criterion in self.criteria)
            if total_weight > 0:
                for criterion in self.criteria:
                    criterion.weight = criterion.weight / total_weight
                return True
            else:
                logger.error("Total weight is zero after update")
                return False
                
        except Exception as e:
            logger.error(f"Error updating criteria weights: {str(e)}")
            return False
    
    async def sensitivity_analysis(
        self, 
        alternatives: List[Alternative],
        weight_variations: Dict[str, List[float]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Perform sensitivity analysis by varying criteria weights.
        
        Args:
            alternatives: List of alternatives to analyze
            weight_variations: Dictionary mapping criterion names to lists of weight values to test
            
        Returns:
            Dictionary mapping weight scenario names to ranking results
        """
        results = {}
        original_weights = {criterion.name: criterion.weight for criterion in self.criteria}
        
        try:
            # Test each weight variation
            for criterion_name, weights in weight_variations.items():
                for i, weight in enumerate(weights):
                    scenario_name = f"{criterion_name}_weight_{weight:.2f}"
                    
                    # Update weight
                    weight_update = {criterion_name: weight}
                    if self.update_criteria_weights(weight_update):
                        # Analyze with new weights
                        ranking = await self.analyze_alternatives(alternatives)
                        results[scenario_name] = ranking
            
            # Restore original weights
            self.update_criteria_weights(original_weights)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {str(e)}")
            # Restore original weights on error
            self.update_criteria_weights(original_weights)
            return {}