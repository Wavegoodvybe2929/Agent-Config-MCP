"""
Particle Swarm Optimization (PSO) for Consensus Building

This module implements particle swarm optimization algorithms for building
consensus among agents in the MCP Swarm Intelligence Server. The implementation
supports multi-modal optimization for complex decision spaces and includes
hybrid PSO variants for specific coordination problems.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts in consensus building."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    EXPERT_OVERRIDE = "expert_override"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    PHEROMONE_GUIDED = "pheromone_guided"


@dataclass
class Particle:
    """Represents a particle in the swarm with position and velocity."""
    id: str
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float
    current_fitness: float = float('inf')
    age: int = 0
    stagnation_count: int = 0


@dataclass
class ConsensusOption:
    """Represents a potential consensus option with support metrics."""
    id: str
    parameters: Dict[str, Any]
    support_count: int
    confidence: float
    weighted_support: float = 0.0
    expertise_backing: float = 0.0
    consistency_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class AgentVote:
    """Represents an agent's vote with expertise weighting."""
    agent_id: str
    option_id: str
    confidence: float
    expertise_level: float
    reasoning: Optional[str] = None
    vote_strength: float = 1.0


@dataclass
class ConsensusResult:
    """Results of consensus building process."""
    winning_option: ConsensusOption
    convergence_iterations: int
    confidence_score: float
    unanimous: bool
    dissenting_agents: List[str]
    final_fitness: float
    optimization_metadata: Dict[str, Any]


class ParticleSwarmConsensus:
    """
    Particle Swarm Optimization for building consensus among agents.
    
    This implementation uses PSO principles to find optimal consensus
    solutions by treating each potential consensus option as a particle
    in a multi-dimensional solution space. The algorithm optimizes for:
    - Maximum agent agreement
    - Weighted expertise consideration
    - Conflict minimization
    - Solution consistency
    """
    
    def __init__(
        self,
        swarm_size: int = 30,
        max_iterations: int = 100,
        w: float = 0.729,           # Inertia weight
        c1: float = 1.49445,        # Cognitive coefficient
        c2: float = 1.49445,        # Social coefficient
        convergence_threshold: float = 1e-6,
        velocity_clamp: float = 0.5,
        elite_size: int = 5,
        adaptive_parameters: bool = True
    ):
        """
        Initialize PSO parameters for consensus building.
        
        Args:
            swarm_size: Number of particles in the swarm
            max_iterations: Maximum optimization iterations
            w: Inertia weight (momentum)
            c1: Cognitive coefficient (personal best attraction)
            c2: Social coefficient (global best attraction)
            convergence_threshold: Convergence detection threshold
            velocity_clamp: Maximum velocity magnitude
            elite_size: Number of elite particles to maintain
            adaptive_parameters: Whether to adapt parameters during optimization
        """
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w_initial = w
        self.w_current = w
        self.c1_initial = c1
        self.c1_current = c1
        self.c2_initial = c2
        self.c2_current = c2
        self.convergence_threshold = convergence_threshold
        self.velocity_clamp = velocity_clamp
        self.elite_size = elite_size
        self.adaptive_parameters = adaptive_parameters
        
        # Optimization state
        self.particles: List[Particle] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_fitness: float = float('inf')
        self.global_best_particle_id: Optional[str] = None
        
        # Performance tracking
        self.iteration_history: List[Dict[str, Any]] = []
        self.convergence_iteration: Optional[int] = None
        self.stagnation_count: int = 0
        self.diversity_history: List[float] = []
        
        # Problem-specific parameters
        self.fitness_function: Optional[Callable] = None
        self.search_space_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.problem_dimension: int = 0
    
    async def build_consensus(
        self,
        options: List[ConsensusOption],
        agent_votes: List[AgentVote],
        conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.WEIGHTED_AVERAGE,
        custom_fitness: Optional[Callable] = None
    ) -> ConsensusResult:
        """
        Build consensus using PSO optimization.
        
        Args:
            options: Available consensus options
            agent_votes: Votes from participating agents
            conflict_resolution: Strategy for handling conflicts
            custom_fitness: Custom fitness function for optimization
            
        Returns:
            ConsensusResult with optimal consensus and metadata
        """
        if not options or not agent_votes:
            raise ValueError("Options and agent votes are required for consensus building")
        
        logger.info(f"Starting PSO consensus building: {len(options)} options, {len(agent_votes)} votes")
        
        # Setup optimization problem
        self._setup_consensus_problem(options, agent_votes, conflict_resolution, custom_fitness)
        
        # Initialize particle swarm
        self._initialize_swarm()
        
        # Run PSO optimization
        await self._optimize_consensus()
        
        # Extract consensus result
        result = self._extract_consensus_result(options, agent_votes)
        
        logger.info(f"Consensus building completed: {result.confidence_score:.3f} confidence")
        
        return result
    
    def _setup_consensus_problem(
        self,
        options: List[ConsensusOption],
        agent_votes: List[AgentVote],
        conflict_resolution: ConflictResolutionStrategy,
        custom_fitness: Optional[Callable]
    ) -> None:
        """Setup the consensus optimization problem."""
        # Define problem dimension (number of options + weighting parameters)
        self.problem_dimension = len(options) + len(set(vote.agent_id for vote in agent_votes))
        
        # Set search space bounds
        lower_bounds = np.zeros(self.problem_dimension)
        upper_bounds = np.ones(self.problem_dimension)
        self.search_space_bounds = (lower_bounds, upper_bounds)
        
        # Setup fitness function
        if custom_fitness:
            self.fitness_function = custom_fitness
        else:
            self.fitness_function = self._create_consensus_fitness_function(
                options, agent_votes, conflict_resolution
            )
    
    def _create_consensus_fitness_function(
        self,
        options: List[ConsensusOption],
        agent_votes: List[AgentVote],
        conflict_resolution: ConflictResolutionStrategy
    ) -> Callable[[np.ndarray], float]:
        """Create fitness function for consensus optimization."""
        
        def fitness(position: np.ndarray) -> float:
            """
            Fitness function that maximizes consensus quality.
            Position vector represents weights for options and agent influence.
            """
            num_options = len(options)
            option_weights = position[:num_options]
            agent_influence = position[num_options:]
            
            # Normalize weights
            if np.sum(option_weights) > 0:
                option_weights = option_weights / np.sum(option_weights)
            else:
                option_weights = np.ones(num_options) / num_options
            
            # Calculate consensus metrics
            agreement_score = self._calculate_agreement_score(
                option_weights, options, agent_votes, agent_influence
            )
            
            consistency_score = self._calculate_consistency_score(
                option_weights, options
            )
            
            expertise_alignment = self._calculate_expertise_alignment(
                option_weights, agent_votes, agent_influence
            )
            
            conflict_penalty = self._calculate_conflict_penalty(
                option_weights, agent_votes, conflict_resolution
            )
            
            # Combine scores (higher is better, but we minimize in PSO)
            fitness_score = (
                agreement_score * 0.4 +
                consistency_score * 0.2 +
                expertise_alignment * 0.3 +
                (1.0 - conflict_penalty) * 0.1
            )
            
            # Return negative for minimization
            return -fitness_score
        
        return fitness
    
    def _calculate_agreement_score(
        self,
        option_weights: np.ndarray,
        options: List[ConsensusOption],
        agent_votes: List[AgentVote],
        agent_influence: np.ndarray
    ) -> float:
        """Calculate how well the weighted options align with agent preferences."""
        total_agreement = 0.0
        agent_ids = list(set(vote.agent_id for vote in agent_votes))
        
        for i, agent_id in enumerate(agent_ids):
            agent_votes_filtered = [v for v in agent_votes if v.agent_id == agent_id]
            
            for vote in agent_votes_filtered:
                option_idx = next((j for j, opt in enumerate(options) if opt.id == vote.option_id), -1)
                if option_idx >= 0:
                    # Weight vote by agent influence, option weight, and vote confidence
                    agreement = (
                        option_weights[option_idx] *
                        agent_influence[i] *
                        vote.confidence *
                        vote.vote_strength
                    )
                    total_agreement += agreement
        
        return min(total_agreement, 1.0)
    
    def _calculate_consistency_score(
        self,
        option_weights: np.ndarray,
        options: List[ConsensusOption]
    ) -> float:
        """Calculate consistency of the weighted solution."""
        if len(options) <= 1:
            return 1.0
        
        # Calculate weighted parameter consistency
        consistency = 0.0
        total_comparisons = 0
        
        for i, opt1 in enumerate(options):
            for j, opt2 in enumerate(options):
                if i < j:
                    # Compare parameter similarity weighted by option weights
                    similarity = self._calculate_parameter_similarity(opt1, opt2)
                    weight_product = option_weights[i] * option_weights[j]
                    consistency += similarity * weight_product
                    total_comparisons += weight_product
        
        return consistency / total_comparisons if total_comparisons > 0 else 0.0
    
    def _calculate_parameter_similarity(self, opt1: ConsensusOption, opt2: ConsensusOption) -> float:
        """Calculate similarity between two options' parameters."""
        # Simple parameter similarity based on common keys
        common_keys = set(opt1.parameters.keys()) & set(opt2.parameters.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = opt1.parameters[key], opt2.parameters[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                if val1 == val2 == 0:
                    similarities.append(1.0)
                else:
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        similarity = 1.0 - abs(val1 - val2) / max_val
                    else:
                        similarity = 1.0
                    similarities.append(max(0.0, similarity))
            elif val1 == val2:
                # Exact match for other types
                similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _calculate_expertise_alignment(
        self,
        option_weights: np.ndarray,
        agent_votes: List[AgentVote],
        agent_influence: np.ndarray
    ) -> float:
        """Calculate how well the solution aligns with expert opinions."""
        expertise_alignment = 0.0
        total_expertise = 0.0
        
        agent_ids = list(set(vote.agent_id for vote in agent_votes))
        
        for i, agent_id in enumerate(agent_ids):
            agent_votes_filtered = [v for v in agent_votes if v.agent_id == agent_id]
            agent_expertise = np.mean([v.expertise_level for v in agent_votes_filtered])
            
            # Calculate this agent's alignment with the weighted solution
            agent_alignment = 0.0
            for vote in agent_votes_filtered:
                option_idx = next((j for j, _ in enumerate(option_weights) if _ > 0), -1)
                if option_idx >= 0:
                    agent_alignment += option_weights[option_idx] * vote.confidence
            
            # Weight by expertise and influence
            expertise_alignment += agent_alignment * agent_expertise * agent_influence[i]
            total_expertise += agent_expertise * agent_influence[i]
        
        return expertise_alignment / total_expertise if total_expertise > 0 else 0.0
    
    def _calculate_conflict_penalty(
        self,
        option_weights: np.ndarray,
        agent_votes: List[AgentVote],
        conflict_resolution: ConflictResolutionStrategy
    ) -> float:
        """Calculate penalty for conflicts in the solution."""
        if conflict_resolution == ConflictResolutionStrategy.MAJORITY_VOTE:
            # Penalize solutions that don't have clear majority
            max_weight = np.max(option_weights)
            return 1.0 - max_weight if max_weight < 0.51 else 0.0
        
        elif conflict_resolution == ConflictResolutionStrategy.WEIGHTED_AVERAGE:
            # Penalize highly dispersed solutions
            entropy = -np.sum(option_weights * np.log(option_weights + 1e-10))
            max_entropy = np.log(len(option_weights))
            return entropy / max_entropy if max_entropy > 0 else 0.0
        
        else:
            # Generic conflict penalty based on vote disagreement
            disagreement = 0.0
            total_votes = len(agent_votes)
            
            if total_votes > 1:
                # Calculate pairwise disagreements
                for i, vote1 in enumerate(agent_votes):
                    for j, vote2 in enumerate(agent_votes):
                        if i < j and vote1.option_id != vote2.option_id:
                            disagreement += 1.0
                
                disagreement = disagreement / (total_votes * (total_votes - 1) / 2)
            
            return disagreement
    
    def _initialize_swarm(self) -> None:
        """Initialize the particle swarm."""
        if self.search_space_bounds is None:
            raise RuntimeError("Search space bounds not initialized")
            
        self.particles = []
        lower_bounds, upper_bounds = self.search_space_bounds
        
        for i in range(self.swarm_size):
            # Random initialization within bounds
            position = np.random.uniform(lower_bounds, upper_bounds)
            velocity = np.random.uniform(-0.1, 0.1, self.problem_dimension)
            
            particle = Particle(
                id=f"particle_{i}",
                position=position.copy(),
                velocity=velocity,
                best_position=position.copy(),
                best_fitness=float('inf')
            )
            
            # Evaluate initial fitness
            if self.fitness_function is None:
                raise RuntimeError("Fitness function not initialized")
            particle.current_fitness = self.fitness_function(position)
            particle.best_fitness = particle.current_fitness
            
            self.particles.append(particle)
            
            # Update global best
            if particle.current_fitness < self.global_best_fitness:
                self.global_best_fitness = particle.current_fitness
                self.global_best_position = position.copy()
                self.global_best_particle_id = particle.id
        
        logger.info(f"Initialized {len(self.particles)} particles, best fitness: {self.global_best_fitness:.6f}")
    
    async def _optimize_consensus(self) -> None:
        """Run the main PSO optimization loop."""
        self.iteration_history = []
        self.stagnation_count = 0
        
        for iteration in range(self.max_iterations):
            iteration_start = datetime.now()
            
            # Update particle positions and velocities
            await self._update_particles()
            
            # Calculate diversity metrics
            diversity = self._calculate_swarm_diversity()
            self.diversity_history.append(diversity)
            
            # Adaptive parameter adjustment
            if self.adaptive_parameters:
                self._adapt_parameters(iteration, diversity)
            
            # Record iteration statistics
            fitnesses = [p.current_fitness for p in self.particles]
            iteration_stats = {
                "iteration": iteration,
                "global_best_fitness": self.global_best_fitness,
                "average_fitness": np.mean(fitnesses),
                "fitness_std": np.std(fitnesses),
                "diversity": diversity,
                "w": self.w_current,
                "c1": self.c1_current,
                "c2": self.c2_current,
                "duration": (datetime.now() - iteration_start).total_seconds()
            }
            self.iteration_history.append(iteration_stats)
            
            # Check for convergence
            if self._check_convergence():
                self.convergence_iteration = iteration
                logger.info(f"PSO converged at iteration {iteration}")
                break
            
            # Prevent premature convergence with diversity injection
            if diversity < 0.01 and iteration > 10:
                self._inject_diversity()
        
        logger.info(f"PSO optimization completed: {len(self.iteration_history)} iterations")
    
    async def _update_particles(self) -> None:
        """Update all particle positions and velocities."""
        if self.search_space_bounds is None or self.fitness_function is None:
            raise RuntimeError("PSO not properly initialized")
            
        lower_bounds, upper_bounds = self.search_space_bounds
        
        for particle in self.particles:
            # Update velocity
            r1, r2 = np.random.random(2)
            
            cognitive_velocity = self.c1_current * r1 * (particle.best_position - particle.position)
            social_velocity = self.c2_current * r2 * (self.global_best_position - particle.position)
            
            particle.velocity = (
                self.w_current * particle.velocity +
                cognitive_velocity +
                social_velocity
            )
            
            # Apply velocity clamping
            velocity_magnitude = np.linalg.norm(particle.velocity)
            if velocity_magnitude > self.velocity_clamp:
                particle.velocity = particle.velocity / velocity_magnitude * self.velocity_clamp
            
            # Update position
            particle.position += particle.velocity
            
            # Apply boundary constraints
            particle.position = np.clip(particle.position, lower_bounds, upper_bounds)
            
            # Evaluate fitness
            new_fitness = self.fitness_function(particle.position)
            particle.current_fitness = new_fitness
            
            # Update personal best
            if new_fitness < particle.best_fitness:
                particle.best_fitness = new_fitness
                particle.best_position = particle.position.copy()
                particle.stagnation_count = 0
            else:
                particle.stagnation_count += 1
            
            # Update global best
            if new_fitness < self.global_best_fitness:
                self.global_best_fitness = new_fitness
                self.global_best_position = particle.position.copy()
                self.global_best_particle_id = particle.id
                self.stagnation_count = 0
            
            particle.age += 1
        
        self.stagnation_count += 1
    
    def _calculate_swarm_diversity(self) -> float:
        """Calculate diversity of the particle swarm."""
        if len(self.particles) < 2:
            return 0.0
        
        positions = np.array([p.position for p in self.particles])
        
        # Calculate average pairwise distance
        total_distance = 0.0
        num_pairs = 0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                total_distance += distance
                num_pairs += 1
        
        average_distance = total_distance / num_pairs if num_pairs > 0 else 0.0
        
        # Normalize by maximum possible distance
        if self.search_space_bounds is None:
            return 0.0
        max_distance = np.linalg.norm(
            self.search_space_bounds[1] - self.search_space_bounds[0]
        )
        
        return float(average_distance / max_distance) if max_distance > 0 else 0.0
    
    def _adapt_parameters(self, iteration: int, diversity: float) -> None:
        """Adapt PSO parameters based on optimization progress."""
        progress = iteration / self.max_iterations
        
        # Linearly decrease inertia weight
        self.w_current = self.w_initial * (1.0 - progress) + 0.4 * progress
        
        # Adapt cognitive and social coefficients based on diversity
        if diversity < 0.1:  # Low diversity - encourage exploration
            self.c1_current = self.c1_initial * 1.2
            self.c2_current = self.c2_initial * 0.8
        elif diversity > 0.5:  # High diversity - encourage exploitation
            self.c1_current = self.c1_initial * 0.8
            self.c2_current = self.c2_initial * 1.2
        else:  # Normal diversity
            self.c1_current = self.c1_initial
            self.c2_current = self.c2_initial
    
    def _inject_diversity(self) -> None:
        """Inject diversity into the swarm to prevent premature convergence."""
        if self.search_space_bounds is None or self.fitness_function is None:
            return
            
        # Reinitialize worst performing particles
        self.particles.sort(key=lambda p: p.current_fitness, reverse=True)
        num_to_reinitialize = max(1, self.swarm_size // 4)
        
        lower_bounds, upper_bounds = self.search_space_bounds
        
        for i in range(num_to_reinitialize):
            particle = self.particles[i]
            
            # Reinitialize position
            particle.position = np.random.uniform(lower_bounds, upper_bounds)
            particle.velocity = np.random.uniform(-0.1, 0.1, self.problem_dimension)
            
            # Evaluate new fitness
            particle.current_fitness = self.fitness_function(particle.position)
            
            # Reset personal best if improved
            if particle.current_fitness < particle.best_fitness:
                particle.best_fitness = particle.current_fitness
                particle.best_position = particle.position.copy()
            
            particle.stagnation_count = 0
    
    def _check_convergence(self) -> bool:
        """Check if the algorithm has converged."""
        if len(self.iteration_history) < 10:
            return False
        
        # Check fitness improvement stagnation
        recent_best = [h["global_best_fitness"] for h in self.iteration_history[-10:]]
        fitness_variance = np.var(recent_best)
        
        if fitness_variance < self.convergence_threshold:
            return True
        
        # Check for diversity-based convergence
        if len(self.diversity_history) >= 5:
            recent_diversity = self.diversity_history[-5:]
            if np.mean(recent_diversity) < 0.01:
                return True
        
        return False
    
    def _extract_consensus_result(
        self,
        options: List[ConsensusOption],
        agent_votes: List[AgentVote]
    ) -> ConsensusResult:
        """Extract the final consensus result from optimization."""
        if self.global_best_position is None:
            raise RuntimeError("No optimization solution found")
        
        # Extract option weights from best position
        num_options = len(options)
        option_weights = self.global_best_position[:num_options]
        
        # Normalize weights
        if np.sum(option_weights) > 0:
            option_weights = option_weights / np.sum(option_weights)
        
        # Find winning option
        winning_idx = np.argmax(option_weights)
        winning_option = options[winning_idx]
        
        # Calculate confidence metrics
        confidence_score = float(option_weights[winning_idx])
        
        # Determine unanimity and dissenting agents
        winning_votes = [v for v in agent_votes if v.option_id == winning_option.id]
        all_agents = set(v.agent_id for v in agent_votes)
        supporting_agents = set(v.agent_id for v in winning_votes)
        dissenting_agents = list(all_agents - supporting_agents)
        unanimous = len(dissenting_agents) == 0
        
        # Prepare optimization metadata
        optimization_metadata = {
            "total_iterations": len(self.iteration_history),
            "convergence_iteration": self.convergence_iteration,
            "final_fitness": self.global_best_fitness,
            "option_weights": option_weights.tolist(),
            "swarm_diversity": self.diversity_history[-1] if self.diversity_history else 0.0,
            "parameter_adaptation": self.adaptive_parameters,
            "stagnation_count": self.stagnation_count
        }
        
        return ConsensusResult(
            winning_option=winning_option,
            convergence_iterations=len(self.iteration_history),
            confidence_score=confidence_score,
            unanimous=unanimous,
            dissenting_agents=dissenting_agents,
            final_fitness=abs(self.global_best_fitness),  # Convert back to positive
            optimization_metadata=optimization_metadata
        )
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get detailed optimization statistics."""
        if not self.iteration_history:
            return {"status": "not_run"}
        
        fitnesses = [h["global_best_fitness"] for h in self.iteration_history]
        durations = [h["duration"] for h in self.iteration_history]
        diversities = [h["diversity"] for h in self.iteration_history]
        
        return {
            "total_iterations": len(self.iteration_history),
            "convergence_iteration": self.convergence_iteration,
            "best_fitness": self.global_best_fitness,
            "initial_fitness": fitnesses[0],
            "final_fitness": fitnesses[-1],
            "fitness_improvement": fitnesses[0] - fitnesses[-1],
            "average_iteration_time": np.mean(durations),
            "total_optimization_time": sum(durations),
            "final_diversity": diversities[-1] if diversities else 0.0,
            "average_diversity": np.mean(diversities) if diversities else 0.0,
            "parameter_adaptation": self.adaptive_parameters,
            "swarm_size": self.swarm_size,
            "convergence_efficiency": self._calculate_convergence_efficiency()
        }
    
    def _calculate_convergence_efficiency(self) -> float:
        """Calculate how efficiently the algorithm converged."""
        if not self.convergence_iteration:
            return 0.0
        
        return 1.0 - (self.convergence_iteration / self.max_iterations)