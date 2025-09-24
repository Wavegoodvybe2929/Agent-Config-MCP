"""
Predictive Success Models for MCP Swarm Intelligence Server

This module implements predictive models for task success probability based on 
historical patterns, context analysis, and machine learning approaches.
"""

import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import sqlite3
from pathlib import Path

# Machine learning imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Scikit-learn not available. Using heuristic predictions.")

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    """Types of predictions supported"""
    TASK_SUCCESS = "task_success"
    COORDINATION_SUCCESS = "coordination_success"
    PERFORMANCE_METRIC = "performance_metric"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    COMPLETION_TIME = "completion_time"

class ModelType(Enum):
    """Types of predictive models"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    ENSEMBLE = "ensemble"
    NEURAL_NETWORK = "neural_network"

@dataclass
class PredictionRequest:
    """Request for success prediction"""
    prediction_type: PredictionType
    context: Dict[str, Any]
    target_metrics: Dict[str, float]
    constraints: Dict[str, Any]
    confidence_threshold: float = 0.7

@dataclass
class PredictionResult:
    """Result of a prediction request"""
    prediction_type: PredictionType
    success_probability: float
    confidence_score: float
    predicted_metrics: Dict[str, float]
    risk_factors: List[str]
    recommendations: List[str]
    model_used: str
    uncertainty: float

@dataclass
class ModelPerformance:
    """Performance metrics for a predictive model"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_samples: int
    last_updated: datetime

class PredictiveSuccessModels:
    """
    Predictive models for task success probability using machine learning
    and historical pattern analysis.
    """
    
    def __init__(
        self,
        database_path: str = "data/memory.db",
        model_update_interval: int = 24,  # hours
        min_training_samples: int = 50
    ):
        self.database_path = Path(database_path)
        self.model_update_interval = model_update_interval
        self.min_training_samples = min_training_samples
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.feature_encoders: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        
        # Training data cache
        self.training_data_cache: Dict[str, Dict[str, Any]] = {}
        self.last_training_update: Dict[str, datetime] = {}
        
        # Prediction cache
        self.prediction_cache: Dict[str, Tuple[PredictionResult, datetime]] = {}
        self.cache_ttl = timedelta(hours=1)
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize predictive models for different prediction types"""
        if ML_AVAILABLE:
            # Task success classification model
            self.models['task_success'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=5,
                random_state=42
            )
            
            # Coordination success model
            self.models['coordination_success'] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
            
            # Performance metric regression model
            self.models['performance_metric'] = RandomForestRegressor(
                n_estimators=80,
                max_depth=10,
                random_state=42
            )
            
            # Resource efficiency model
            self.models['resource_efficiency'] = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
            
            # Completion time prediction model
            self.models['completion_time'] = GradientBoostingRegressor(
                n_estimators=120,
                max_depth=10,
                learning_rate=0.05,
                random_state=42
            )
        else:
            logger.info("Using heuristic models for predictions")
    
    def _get_db_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    async def predict_success(self, request: PredictionRequest) -> PredictionResult:
        """
        Predict success probability for a given request
        
        Args:
            request: Prediction request with context and requirements
            
        Returns:
            Detailed prediction result with probability and analysis
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result:
                return cached_result
            
            # Extract features from request
            features = self._extract_prediction_features(request)
            
            # Select appropriate model
            model_name = self._select_model(request.prediction_type)
            
            if ML_AVAILABLE and model_name in self.models:
                # Use ML model for prediction
                result = await self._ml_prediction(request, features, model_name)
            else:
                # Use heuristic prediction
                result = await self._heuristic_prediction(request, features)
            
            # Cache the result
            self._cache_prediction(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error("Error predicting success: %s", e)
            return self._create_default_prediction(request)
    
    async def predict_batch(
        self, 
        requests: List[PredictionRequest]
    ) -> List[PredictionResult]:
        """
        Predict success for multiple requests efficiently
        
        Args:
            requests: List of prediction requests
            
        Returns:
            List of prediction results
        """
        try:
            results = []
            
            # Group requests by prediction type for batch processing
            grouped_requests = self._group_requests_by_type(requests)
            
            for pred_type, type_requests in grouped_requests.items():
                # Process each group with optimized batch prediction
                type_results = await self._batch_predict_by_type(pred_type, type_requests)
                results.extend(type_results)
            
            return results
            
        except Exception as e:
            logger.error("Error in batch prediction: %s", e)
            return [self._create_default_prediction(req) for req in requests]
    
    async def train_models(self, force_retrain: bool = False) -> Dict[str, ModelPerformance]:
        """
        Train or retrain predictive models using historical data
        
        Args:
            force_retrain: Force retraining even if recently updated
            
        Returns:
            Performance metrics for each trained model
        """
        try:
            performance_results = {}
            
            # Load training data from database
            training_data = await self._load_training_data()
            
            if not training_data:
                logger.warning("No training data available")
                return {}
            
            for pred_type in PredictionType:
                type_key = pred_type.value
                
                # Check if retraining is needed
                if not force_retrain and not self._needs_retraining(type_key):
                    continue
                
                # Prepare training data for this prediction type
                X, y = self._prepare_training_data(training_data, pred_type)
                
                if len(X) < self.min_training_samples:
                    logger.warning(
                        "Insufficient training samples for %s: %d < %d", 
                        type_key, len(X), self.min_training_samples
                    )
                    continue
                
                # Train the model
                performance = await self._train_model(type_key, X, y)
                if performance:
                    performance_results[type_key] = performance
                    self.last_training_update[type_key] = datetime.now()
            
            return performance_results
            
        except Exception as e:
            logger.error("Error training models: %s", e)
            return {}
    
    async def evaluate_model_performance(
        self, 
        model_name: str,
        test_data: Optional[Dict[str, Any]] = None
    ) -> Optional[ModelPerformance]:
        """
        Evaluate performance of a specific model
        
        Args:
            model_name: Name of the model to evaluate
            test_data: Optional test data, will load from DB if None
            
        Returns:
            Performance metrics for the model
        """
        try:
            if model_name not in self.models:
                logger.error("Model not found: %s", model_name)
                return None
            
            # Load test data if not provided
            if test_data is None:
                test_data = await self._load_test_data(model_name)
            
            if not test_data:
                logger.warning("No test data available for %s", model_name)
                return None
            
            # Prepare test features and targets
            pred_type = PredictionType(model_name)
            X_test, y_test = self._prepare_training_data(test_data, pred_type)
            
            if len(X_test) == 0:
                return None
            
            # Make predictions and calculate metrics
            model = self.models[model_name]
            y_pred = model.predict(X_test)
            
            # Calculate performance metrics based on model type
            if self._is_classification_model(model_name):
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
            else:
                # For regression models, use different metrics
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                # Convert regression metrics to classification-like metrics
                accuracy = max(0, r2)  # R² as accuracy proxy
                precision = max(0, 1 - mse)  # Inverse MSE as precision proxy
                recall = accuracy
                f1 = 2 * (accuracy * precision) / (accuracy + precision) if (accuracy + precision) > 0 else 0
            
            performance = ModelPerformance(
                model_name=model_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_samples=len(X_test),
                last_updated=datetime.now()
            )
            
            self.model_performance[model_name] = performance
            return performance
            
        except Exception as e:
            logger.error("Error evaluating model performance: %s", e)
            return None
    
    async def get_prediction_insights(
        self, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get insights about prediction factors and model confidence
        
        Args:
            context: Current context for analysis
            
        Returns:
            Insights about prediction factors and confidence
        """
        try:
            insights = {
                'context_analysis': self._analyze_context_factors(context),
                'risk_assessment': await self._assess_risks(context),
                'confidence_factors': self._identify_confidence_factors(context),
                'historical_patterns': await self._find_historical_patterns(context),
                'model_reliability': self._get_model_reliability_scores()
            }
            
            return insights
            
        except Exception as e:
            logger.error("Error generating prediction insights: %s", e)
            return {}
    
    def _extract_prediction_features(self, request: PredictionRequest) -> Dict[str, Any]:
        """Extract features from prediction request for model input"""
        features = {}
        
        # Context features
        context = request.context
        features.update({
            f"context_{k}": v for k, v in context.items()
            if isinstance(v, (int, float, str, bool))
        })
        
        # Target metric features
        features.update({
            f"target_{k}": v for k, v in request.target_metrics.items()
            if isinstance(v, (int, float))
        })
        
        # Constraint features
        constraints = request.constraints
        features.update({
            f"constraint_{k}": v for k, v in constraints.items()
            if isinstance(v, (int, float, bool))
        })
        
        # Derived features
        features['context_complexity'] = len(context)
        features['num_targets'] = len(request.target_metrics)
        features['num_constraints'] = len(constraints)
        features['confidence_threshold'] = request.confidence_threshold
        features['prediction_time'] = datetime.now().timestamp()
        
        return features
    
    def _select_model(self, prediction_type: PredictionType) -> str:
        """Select appropriate model for prediction type"""
        return prediction_type.value
    
    async def _ml_prediction(
        self, 
        request: PredictionRequest, 
        features: Dict[str, Any], 
        model_name: str
    ) -> PredictionResult:
        """Make prediction using machine learning model"""
        try:
            model = self.models[model_name]
            
            # Convert features to model input format
            feature_vector = self._features_to_vector(features, model_name)
            
            if feature_vector is None:
                # Fall back to heuristic prediction
                return await self._heuristic_prediction(request, features)
            
            # Make prediction
            if self._is_classification_model(model_name):
                prediction = model.predict_proba([feature_vector])[0]
                success_prob = prediction[1] if len(prediction) > 1 else prediction[0]
                confidence = max(prediction)
            else:
                prediction = model.predict([feature_vector])[0]
                success_prob = min(1.0, max(0.0, prediction))  # Clamp to [0,1]
                confidence = 0.8  # Default confidence for regression
            
            # Generate predicted metrics
            predicted_metrics = self._generate_predicted_metrics(
                request, success_prob
            )
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(features, success_prob)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                request, success_prob, risk_factors
            )
            
            # Calculate uncertainty
            uncertainty = 1.0 - confidence
            
            return PredictionResult(
                prediction_type=request.prediction_type,
                success_probability=success_prob,
                confidence_score=confidence,
                predicted_metrics=predicted_metrics,
                risk_factors=risk_factors,
                recommendations=recommendations,
                model_used=f"ML-{model_name}",
                uncertainty=uncertainty
            )
            
        except Exception as e:
            logger.error("Error in ML prediction: %s", e)
            return await self._heuristic_prediction(request, features)
    
    async def _heuristic_prediction(
        self, 
        request: PredictionRequest, 
        features: Dict[str, Any]
    ) -> PredictionResult:
        """Make prediction using heuristic methods"""
        try:
            # Simple heuristic based on feature analysis
            context_score = self._calculate_context_score(features)
            complexity_penalty = self._calculate_complexity_penalty(features)
            historical_bias = await self._get_historical_bias(request.prediction_type)
            
            # Combine factors for success probability
            success_prob = (context_score * 0.5 + 
                          (1.0 - complexity_penalty) * 0.3 + 
                          historical_bias * 0.2)
            
            success_prob = min(1.0, max(0.0, success_prob))
            
            # Heuristic confidence based on feature completeness
            confidence = min(0.8, len(features) / 20.0)  # Max 80% confidence for heuristics
            
            # Generate predicted metrics
            predicted_metrics = self._generate_predicted_metrics(
                request, success_prob
            )
            
            # Simple risk assessment
            risk_factors = []
            if success_prob < 0.5:
                risk_factors.append("Low success probability predicted")
            if complexity_penalty > 0.7:
                risk_factors.append("High task complexity")
            if confidence < 0.5:
                risk_factors.append("Limited prediction confidence")
            
            # Basic recommendations
            recommendations = []
            if success_prob < 0.7:
                recommendations.append("Consider task simplification")
            if len(features) < 10:
                recommendations.append("Provide more context for better prediction")
            
            return PredictionResult(
                prediction_type=request.prediction_type,
                success_probability=success_prob,
                confidence_score=confidence,
                predicted_metrics=predicted_metrics,
                risk_factors=risk_factors,
                recommendations=recommendations,
                model_used="heuristic",
                uncertainty=1.0 - confidence
            )
            
        except Exception as e:
            logger.error("Error in heuristic prediction: %s", e)
            return self._create_default_prediction(request)
    
    def _calculate_context_score(self, features: Dict[str, Any]) -> float:
        """Calculate a score based on context quality"""
        score = 0.5  # Base score
        
        # Bonus for having key context features
        key_indicators = ['agents_available', 'resource_level', 'time_available', 'priority']
        for indicator in key_indicators:
            if any(indicator in key for key in features.keys()):
                score += 0.1
        
        return min(1.0, score)
    
    def _calculate_complexity_penalty(self, features: Dict[str, Any]) -> float:
        """Calculate penalty based on task complexity"""
        penalty = 0.0
        
        # Penalty for high complexity indicators
        if features.get('context_complexity', 0) > 10:
            penalty += 0.2
        if features.get('num_constraints', 0) > 5:
            penalty += 0.2
        if features.get('num_targets', 0) > 3:
            penalty += 0.1
        
        return min(1.0, penalty)
    
    async def _get_historical_bias(self, prediction_type: PredictionType) -> float:
        """Get historical success bias for prediction type"""
        try:
            conn = self._get_db_connection()
            try:
                cursor = conn.execute("""
                    SELECT AVG(CASE 
                        WHEN json_extract(success_metrics, '$.completion_rate') > 0.7 
                        THEN 1.0 ELSE 0.0 END) as avg_success
                    FROM adaptation_events
                    WHERE event_type = ?
                    AND timestamp > datetime('now', '-30 days')
                """, (prediction_type.value,))
                
                result = cursor.fetchone()
                return result['avg_success'] if result and result['avg_success'] else 0.5
            finally:
                conn.close()
        except Exception as e:
            logger.error("Error getting historical bias: %s", e)
            return 0.5
    
    def _generate_predicted_metrics(
        self, 
        request: PredictionRequest, 
        success_prob: float
    ) -> Dict[str, float]:
        """Generate predicted performance metrics"""
        base_metrics = {
            'completion_rate': success_prob,
            'quality_score': success_prob * 0.9,  # Slight quality discount
            'efficiency': success_prob * 0.8,     # Efficiency tends to be lower
            'user_satisfaction': success_prob * 0.95
        }
        
        # Adjust based on target metrics
        for metric, target in request.target_metrics.items():
            if metric in base_metrics:
                # Adjust prediction based on target difficulty
                adjustment = min(1.0, target / base_metrics[metric])
                base_metrics[metric] *= adjustment
        
        return base_metrics
    
    def _identify_risk_factors(
        self, 
        features: Dict[str, Any], 
        success_prob: float
    ) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        if success_prob < 0.3:
            risks.append("Very low success probability")
        elif success_prob < 0.6:
            risks.append("Below-average success probability")
        
        if features.get('context_complexity', 0) > 15:
            risks.append("High context complexity")
        
        if features.get('num_constraints', 0) > 8:
            risks.append("Many constraints may limit options")
        
        return risks
    
    def _generate_recommendations(
        self, 
        request: PredictionRequest, 
        success_prob: float, 
        risk_factors: List[str]
    ) -> List[str]:
        """Generate recommendations based on prediction"""
        recommendations = []
        
        if success_prob < 0.5:
            recommendations.append("Consider breaking task into smaller parts")
            recommendations.append("Allocate additional resources")
        
        if len(risk_factors) > 2:
            recommendations.append("Address identified risk factors before proceeding")
        
        if request.confidence_threshold > 0.8 and success_prob < 0.8:
            recommendations.append("Current prediction below confidence threshold")
        
        return recommendations
    
    def _create_default_prediction(self, request: PredictionRequest) -> PredictionResult:
        """Create default prediction result when prediction fails"""
        return PredictionResult(
            prediction_type=request.prediction_type,
            success_probability=0.5,
            confidence_score=0.1,
            predicted_metrics={'completion_rate': 0.5},
            risk_factors=["Prediction system unavailable"],
            recommendations=["Manual assessment recommended"],
            model_used="default",
            uncertainty=0.9
        )
    
    def _generate_cache_key(self, request: PredictionRequest) -> str:
        """Generate cache key for prediction request"""
        import hashlib
        request_str = json.dumps({
            'type': request.prediction_type.value,
            'context': request.context,
            'targets': request.target_metrics,
            'constraints': request.constraints,
            'threshold': request.confidence_threshold
        }, sort_keys=True)
        return hashlib.md5(request_str.encode()).hexdigest()
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[PredictionResult]:
        """Get cached prediction if still valid"""
        if cache_key in self.prediction_cache:
            result, timestamp = self.prediction_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return result
            else:
                # Remove expired cache entry
                del self.prediction_cache[cache_key]
        return None
    
    def _cache_prediction(self, cache_key: str, result: PredictionResult):
        """Cache prediction result"""
        self.prediction_cache[cache_key] = (result, datetime.now())
        
        # Limit cache size
        if len(self.prediction_cache) > 1000:
            # Remove oldest entries
            sorted_cache = sorted(
                self.prediction_cache.items(), 
                key=lambda x: x[1][1]
            )
            for key, _ in sorted_cache[:100]:  # Remove oldest 100
                del self.prediction_cache[key]
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all predictive models"""
        try:
            status = {
                'ml_available': ML_AVAILABLE,
                'models_initialized': len(self.models),
                'models_trained': len(self.model_performance),
                'cache_size': len(self.prediction_cache),
                'last_training': {
                    name: update.isoformat() 
                    for name, update in self.last_training_update.items()
                },
                'model_performance': {
                    name: {
                        'accuracy': perf.accuracy,
                        'confidence': perf.f1_score,
                        'samples': perf.training_samples
                    } 
                    for name, perf in self.model_performance.items()
                }
            }
            
            return status
            
        except Exception as e:
            logger.error("Error getting model status: %s", e)
            return {'error': str(e), 'ml_available': ML_AVAILABLE}
    
    # Helper methods for ML functionality
    def _features_to_vector(self, features: Dict[str, Any], model_name: str) -> Optional[List[float]]:
        """Convert features to numeric vector for ML model"""
        if not ML_AVAILABLE:
            return None
        
        try:
            numeric_features = []
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    numeric_features.append(float(value))
                elif isinstance(value, bool):
                    numeric_features.append(1.0 if value else 0.0)
                elif isinstance(value, str):
                    numeric_features.append(float(abs(hash(value)) % 1000))
            
            return numeric_features if numeric_features else None
            
        except Exception as e:
            logger.error("Error converting features to vector: %s", e)
            return None
    
    def _is_classification_model(self, model_name: str) -> bool:
        """Check if model is classification or regression"""
        classification_models = {'task_success', 'resource_efficiency'}
        return model_name in classification_models
    
    def _needs_retraining(self, model_name: str) -> bool:
        """Check if model needs retraining"""
        if model_name not in self.last_training_update:
            return True
        
        last_update = self.last_training_update[model_name]
        time_since_update = datetime.now() - last_update
        return time_since_update.total_seconds() > (self.model_update_interval * 3600)
    
    async def _load_training_data(self) -> Dict[str, Any]:
        """Load training data from database"""
        # Placeholder - would load from adaptation_events and learning_patterns tables
        return {}
    
    async def _load_test_data(self, model_name: str) -> Dict[str, Any]:
        """Load test data for model evaluation"""
        # Placeholder - would load separate test dataset
        return {}
    
    def _prepare_training_data(self, data: Dict[str, Any], pred_type: PredictionType) -> Tuple[List, List]:
        """Prepare training data for specific prediction type"""
        # Placeholder - would prepare X, y arrays from raw data
        return [], []
    
    async def _train_model(self, model_name: str, X: List, y: List) -> Optional[ModelPerformance]:
        """Train a specific model"""
        # Placeholder - would implement actual model training
        return None
    
    def _group_requests_by_type(self, requests: List[PredictionRequest]) -> Dict[PredictionType, List[PredictionRequest]]:
        """Group requests by prediction type for batch processing"""
        groups = {}
        for request in requests:
            if request.prediction_type not in groups:
                groups[request.prediction_type] = []
            groups[request.prediction_type].append(request)
        return groups
    
    async def _batch_predict_by_type(self, pred_type: PredictionType, requests: List[PredictionRequest]) -> List[PredictionResult]:
        """Batch predict for requests of the same type"""
        results = []
        for request in requests:
            result = await self.predict_success(request)
            results.append(result)
        return results
    
    def _analyze_context_factors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context factors for insights"""
        return {'context_quality': 'medium', 'key_factors': []}
    
    async def _assess_risks(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks based on context"""
        return {'risk_level': 'medium', 'primary_risks': []}
    
    def _identify_confidence_factors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Identify factors affecting prediction confidence"""
        return {'confidence_level': 'medium', 'factors': []}
    
    async def _find_historical_patterns(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Find historical patterns matching current context"""
        return {'matching_patterns': 0, 'success_rate': 0.5}
    
    def _get_model_reliability_scores(self) -> Dict[str, float]:
        """Get reliability scores for each model"""
        return {name: perf.f1_score for name, perf in self.model_performance.items()}