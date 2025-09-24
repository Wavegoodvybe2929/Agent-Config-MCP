"""
Evolutionary Parameter Optimizer for MCP Swarm Intelligence Server

This module implements evolutionary optimization algorithms for swarm parameters,
using genetic algorithms, differential evolution, and particle swarm optimization
to find optimal configurations for swarm coordination.
"""

import json
import logging
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path

# Scientific computing imports
try:
    import numpy as np
    from scipy.optimize import differential_evolution, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    np = None
    logging.warning("SciPy not available. Using basic evolutionary algorithms.")

logger = logging.getLogger(__name__)

class OptimizationAlgorithm(Enum):
    """Supported optimization algorithms"""
    GENETIC_ALGORITHM = "genetic_algorithm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"
    HYBRID = "hybrid"

class ParameterType(Enum):
    """Types of parameters that can be optimized"""
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"

@dataclass
class ParameterBounds:
    """Parameter bounds and constraints"""
    name: str
    param_type: ParameterType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[List[str]] = None
    step_size: Optional[float] = None

@dataclass
class Individual:
    """Individual in evolutionary population"""
    parameters: Dict[str, Any]
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationResult:
    """Result of optimization run"""
    best_parameters: Dict[str, Any]
    best_fitness: float
    generations: int
    evaluations: int
    convergence_history: List[float]
    algorithm_used: OptimizationAlgorithm
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class EvolutionaryParameterOptimizer:
    """
    Evolutionary optimization algorithms for swarm parameter tuning using
    genetic algorithms, differential evolution, and particle swarm optimization.
    """
    
    def __init__(
        self,
        database_path: str = "data/memory.db",
        population_size: int = 50,
        max_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_ratio: float = 0.1
    ):
        self.database_path = Path(database_path)
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        
        # Optimization state
        self.populations: Dict[str, List[Individual]] = {}
        self.optimization_history: List[OptimizationResult] = []
        self.parameter_bounds: Dict[str, ParameterBounds] = {}
        self.fitness_cache: Dict[str, float] = {}
        
        # Algorithm-specific parameters
        self.pso_params = {
            'w': 0.7,  # Inertia weight
            'c1': 1.5,  # Cognitive parameter
            'c2': 1.5,  # Social parameter
        }
        
        self.de_params = {
            'f': 0.8,  # Differential weight
            'cr': 0.9,  # Crossover probability
        }
        
        self.sa_params = {
            'initial_temp': 100.0,
            'cooling_rate': 0.95,
            'min_temp': 0.01
        }
        
        # Performance tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'best_fitness_achieved': float('-inf'),
            'average_convergence_time': 0.0
        }
    
    def _get_db_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    async def initialize_database(self):
        """Initialize database tables for optimization history"""
        conn = self._get_db_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    algorithm TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    bounds TEXT NOT NULL,
                    best_fitness REAL NOT NULL,
                    best_parameters TEXT NOT NULL,
                    generations INTEGER NOT NULL,
                    evaluations INTEGER NOT NULL,
                    execution_time REAL NOT NULL,
                    convergence_history TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parameter_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parameter_set TEXT NOT NULL,
                    fitness_score REAL NOT NULL,
                    evaluation_context TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fitness_score 
                ON parameter_performance(fitness_score DESC)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    async def optimize_parameters(
        self,
        fitness_function: Callable[[Dict[str, Any]], float],
        parameter_bounds: Dict[str, ParameterBounds],
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GENETIC_ALGORITHM,
        max_evaluations: Optional[int] = None,
        convergence_threshold: float = 1e-6,
        early_stopping_generations: int = 20
    ) -> OptimizationResult:
        """
        Optimize parameters using specified evolutionary algorithm
        
        Args:
            fitness_function: Function to evaluate parameter fitness
            parameter_bounds: Bounds and constraints for each parameter
            algorithm: Optimization algorithm to use
            max_evaluations: Maximum number of fitness evaluations
            convergence_threshold: Threshold for convergence detection
            early_stopping_generations: Stop if no improvement for this many generations
            
        Returns:
            Optimization result with best parameters and performance metrics
        """
        try:
            start_time = datetime.now()
            
            # Store parameter bounds
            self.parameter_bounds = parameter_bounds
            
            # Select and run optimization algorithm
            if algorithm == OptimizationAlgorithm.GENETIC_ALGORITHM:
                result = await self._genetic_algorithm(
                    fitness_function, max_evaluations, convergence_threshold, early_stopping_generations
                )
            elif algorithm == OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION:
                result = await self._differential_evolution(
                    fitness_function, max_evaluations, convergence_threshold, early_stopping_generations
                )
            elif algorithm == OptimizationAlgorithm.PARTICLE_SWARM:
                result = await self._particle_swarm_optimization(
                    fitness_function, max_evaluations, convergence_threshold, early_stopping_generations
                )
            elif algorithm == OptimizationAlgorithm.SIMULATED_ANNEALING:
                result = await self._simulated_annealing(
                    fitness_function, max_evaluations, convergence_threshold
                )
            elif algorithm == OptimizationAlgorithm.HYBRID:
                result = await self._hybrid_optimization(
                    fitness_function, max_evaluations, convergence_threshold, early_stopping_generations
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            result.algorithm_used = algorithm
            
            # Store result in database
            await self._store_optimization_result(result)
            
            # Update statistics
            self.optimization_stats['total_optimizations'] += 1
            if result.best_fitness > self.optimization_stats['best_fitness_achieved']:
                self.optimization_stats['best_fitness_achieved'] = result.best_fitness
                self.optimization_stats['successful_optimizations'] += 1
            
            # Update average convergence time
            current_avg = self.optimization_stats['average_convergence_time']
            total_opts = self.optimization_stats['total_optimizations']
            self.optimization_stats['average_convergence_time'] = (
                (current_avg * (total_opts - 1) + execution_time) / total_opts
            )
            
            logger.info("Optimization completed: fitness=%.6f, time=%.2fs", 
                       result.best_fitness, execution_time)
            
            return result
            
        except Exception as e:
            logger.error("Error in parameter optimization: %s", e)
            return self._create_default_result(algorithm)
    
    async def _genetic_algorithm(
        self,
        fitness_function: Callable[[Dict[str, Any]], float],
        max_evaluations: Optional[int],
        convergence_threshold: float,
        early_stopping_generations: int
    ) -> OptimizationResult:
        """Implement genetic algorithm optimization"""
        try:
            # Initialize population
            population = self._initialize_population()
            
            # Evaluate initial population
            for individual in population:
                individual.fitness = fitness_function(individual.parameters)
            
            convergence_history = []
            evaluations = len(population)
            generations = 0
            no_improvement_count = 0
            best_fitness = max(ind.fitness for ind in population)
            
            while generations < self.max_generations:
                # Selection, crossover, and mutation
                new_population = self._genetic_selection(population)
                new_population = self._genetic_crossover(new_population)
                new_population = self._genetic_mutation(new_population)
                
                # Evaluate new individuals
                for individual in new_population:
                    if not hasattr(individual, 'fitness') or individual.fitness == 0.0:
                        individual.fitness = fitness_function(individual.parameters)
                        evaluations += 1
                        
                        if max_evaluations and evaluations >= max_evaluations:
                            break
                
                # Elitism: keep best individuals
                population.extend(new_population)
                population.sort(key=lambda x: x.fitness, reverse=True)
                population = population[:self.population_size]
                
                # Track convergence
                current_best = population[0].fitness
                convergence_history.append(current_best)
                
                # Check for convergence
                if abs(current_best - best_fitness) < convergence_threshold:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                    best_fitness = current_best
                
                if no_improvement_count >= early_stopping_generations:
                    logger.info("Early stopping at generation %d", generations)
                    break
                
                generations += 1
                
                if max_evaluations and evaluations >= max_evaluations:
                    break
            
            best_individual = population[0]
            
            return OptimizationResult(
                best_parameters=best_individual.parameters,
                best_fitness=best_individual.fitness,
                generations=generations,
                evaluations=evaluations,
                convergence_history=convergence_history,
                algorithm_used=OptimizationAlgorithm.GENETIC_ALGORITHM,
                execution_time=0.0,  # Will be set by caller
                metadata={'final_population_size': len(population)}
            )
            
        except Exception as e:
            logger.error("Error in genetic algorithm: %s", e)
            return self._create_default_result(OptimizationAlgorithm.GENETIC_ALGORITHM)
    
    async def _differential_evolution(
        self,
        fitness_function: Callable[[Dict[str, Any]], float],
        max_evaluations: Optional[int],
        convergence_threshold: float,
        early_stopping_generations: int
    ) -> OptimizationResult:
        """Implement differential evolution optimization"""
        try:
            if SCIPY_AVAILABLE:
                # Use SciPy's differential evolution
                bounds = self._get_scipy_bounds()
                
                def objective(x):
                    params = self._vector_to_parameters(x)
                    return -fitness_function(params)  # Minimize negative fitness
                
                result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=self.max_generations,
                    popsize=15,  # Population size multiplier
                    atol=convergence_threshold
                )
                
                best_params = self._vector_to_parameters(result.x)
                
                return OptimizationResult(
                    best_parameters=best_params,
                    best_fitness=-result.fun,  # Convert back to maximization
                    generations=result.nit,
                    evaluations=result.nfev,
                    convergence_history=[],  # SciPy doesn't provide this
                    algorithm_used=OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION,
                    execution_time=0.0,
                    metadata={'scipy_success': result.success, 'scipy_message': result.message}
                )
            else:
                # Implement basic differential evolution
                return await self._basic_differential_evolution(
                    fitness_function, max_evaluations, convergence_threshold, early_stopping_generations
                )
                
        except Exception as e:
            logger.error("Error in differential evolution: %s", e)
            return self._create_default_result(OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION)
    
    async def _particle_swarm_optimization(
        self,
        fitness_function: Callable[[Dict[str, Any]], float],
        max_evaluations: Optional[int],
        convergence_threshold: float,
        early_stopping_generations: int
    ) -> OptimizationResult:
        """Implement particle swarm optimization"""
        try:
            # Initialize particles with positions and velocities
            particles = []
            for _ in range(self.population_size):
                particle = {
                    'position': self._generate_random_parameters(),
                    'velocity': self._initialize_velocity(),
                    'best_position': None,
                    'best_fitness': float('-inf'),
                    'fitness': 0.0
                }
                particles.append(particle)
            
            # Global best
            global_best_position = None
            global_best_fitness = float('-inf')
            
            convergence_history = []
            evaluations = 0
            generations = 0
            no_improvement_count = 0
            
            while generations < self.max_generations:
                for particle in particles:
                    # Evaluate fitness
                    fitness = fitness_function(particle['position'])
                    particle['fitness'] = fitness
                    evaluations += 1
                    
                    # Update personal best
                    if fitness > particle['best_fitness']:
                        particle['best_fitness'] = fitness
                        particle['best_position'] = particle['position'].copy()
                    
                    # Update global best
                    if fitness > global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = particle['position'].copy()
                        no_improvement_count = 0
                    
                    if max_evaluations and evaluations >= max_evaluations:
                        break
                
                # Update particle velocities and positions
                for particle in particles:
                    self._update_particle(particle, global_best_position)
                
                convergence_history.append(global_best_fitness)
                
                # Check convergence
                if len(convergence_history) > 1:
                    improvement = abs(convergence_history[-1] - convergence_history[-2])
                    if improvement < convergence_threshold:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0
                
                if no_improvement_count >= early_stopping_generations:
                    logger.info("PSO early stopping at generation %d", generations)
                    break
                
                generations += 1
                
                if max_evaluations and evaluations >= max_evaluations:
                    break
            
            return OptimizationResult(
                best_parameters=global_best_position,
                best_fitness=global_best_fitness,
                generations=generations,
                evaluations=evaluations,
                convergence_history=convergence_history,
                algorithm_used=OptimizationAlgorithm.PARTICLE_SWARM,
                execution_time=0.0,
                metadata={'final_particles': len(particles)}
            )
            
        except Exception as e:
            logger.error("Error in particle swarm optimization: %s", e)
            return self._create_default_result(OptimizationAlgorithm.PARTICLE_SWARM)
    
    async def _simulated_annealing(
        self,
        fitness_function: Callable[[Dict[str, Any]], float],
        max_evaluations: Optional[int],
        convergence_threshold: float
    ) -> OptimizationResult:
        """Implement simulated annealing optimization"""
        try:
            # Initialize with random solution
            current_solution = self._generate_random_parameters()
            current_fitness = fitness_function(current_solution)
            
            best_solution = current_solution.copy()
            best_fitness = current_fitness
            
            temperature = self.sa_params['initial_temp']
            cooling_rate = self.sa_params['cooling_rate']
            min_temp = self.sa_params['min_temp']
            
            convergence_history = []
            evaluations = 1
            iterations = 0
            
            while temperature > min_temp and iterations < self.max_generations * 10:
                # Generate neighbor solution
                neighbor = self._generate_neighbor(current_solution)
                neighbor_fitness = fitness_function(neighbor)
                evaluations += 1
                
                # Calculate acceptance probability
                delta = neighbor_fitness - current_fitness
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    current_solution = neighbor
                    current_fitness = neighbor_fitness
                    
                    # Update best solution
                    if neighbor_fitness > best_fitness:
                        best_solution = neighbor.copy()
                        best_fitness = neighbor_fitness
                
                convergence_history.append(best_fitness)
                
                # Cool down
                temperature *= cooling_rate
                iterations += 1
                
                if max_evaluations and evaluations >= max_evaluations:
                    break
            
            return OptimizationResult(
                best_parameters=best_solution,
                best_fitness=best_fitness,
                generations=iterations,
                evaluations=evaluations,
                convergence_history=convergence_history,
                algorithm_used=OptimizationAlgorithm.SIMULATED_ANNEALING,
                execution_time=0.0,
                metadata={'final_temperature': temperature}
            )
            
        except Exception as e:
            logger.error("Error in simulated annealing: %s", e)
            return self._create_default_result(OptimizationAlgorithm.SIMULATED_ANNEALING)
    
    async def _hybrid_optimization(
        self,
        fitness_function: Callable[[Dict[str, Any]], float],
        max_evaluations: Optional[int],
        convergence_threshold: float,
        early_stopping_generations: int
    ) -> OptimizationResult:
        """Implement hybrid optimization combining multiple algorithms"""
        try:
            # Allocate evaluations across algorithms
            eval_per_algo = (max_evaluations or 1000) // 3
            
            # Run genetic algorithm first for exploration
            ga_result = await self._genetic_algorithm(
                fitness_function, eval_per_algo, convergence_threshold, early_stopping_generations
            )
            
            # Use GA result as starting point for differential evolution
            de_result = await self._differential_evolution(
                fitness_function, eval_per_algo, convergence_threshold, early_stopping_generations
            )
            
            # Final refinement with simulated annealing
            sa_result = await self._simulated_annealing(
                fitness_function, eval_per_algo, convergence_threshold
            )
            
            # Select best result
            results = [ga_result, de_result, sa_result]
            best_result = max(results, key=lambda r: r.best_fitness)
            
            # Combine convergence histories
            combined_history = (ga_result.convergence_history + 
                              de_result.convergence_history + 
                              sa_result.convergence_history)
            
            return OptimizationResult(
                best_parameters=best_result.best_parameters,
                best_fitness=best_result.best_fitness,
                generations=sum(r.generations for r in results),
                evaluations=sum(r.evaluations for r in results),
                convergence_history=combined_history,
                algorithm_used=OptimizationAlgorithm.HYBRID,
                execution_time=0.0,
                metadata={
                    'ga_fitness': ga_result.best_fitness,
                    'de_fitness': de_result.best_fitness,
                    'sa_fitness': sa_result.best_fitness,
                    'best_algorithm': best_result.algorithm_used.value
                }
            )
            
        except Exception as e:
            logger.error("Error in hybrid optimization: %s", e)
            return self._create_default_result(OptimizationAlgorithm.HYBRID)
    
    def _initialize_population(self) -> List[Individual]:
        """Initialize random population for genetic algorithm"""
        population = []
        for i in range(self.population_size):
            parameters = self._generate_random_parameters()
            individual = Individual(
                parameters=parameters,
                fitness=0.0,
                age=0,
                generation=0
            )
            population.append(individual)
        return population
    
    def _generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameter values within bounds"""
        parameters = {}
        
        for name, bounds in self.parameter_bounds.items():
            if bounds.param_type == ParameterType.CONTINUOUS:
                value = random.uniform(bounds.min_value, bounds.max_value)
            elif bounds.param_type == ParameterType.INTEGER:
                value = random.randint(int(bounds.min_value), int(bounds.max_value))
            elif bounds.param_type == ParameterType.CATEGORICAL:
                value = random.choice(bounds.categories)
            elif bounds.param_type == ParameterType.BOOLEAN:
                value = random.choice([True, False])
            else:
                value = 0.5  # Default value
            
            parameters[name] = value
        
        return parameters
    
    def _genetic_selection(self, population: List[Individual]) -> List[Individual]:
        """Select parents for reproduction using tournament selection"""
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Tournament selection
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x.fitness)
            
            # Create copy for new generation
            new_individual = Individual(
                parameters=winner.parameters.copy(),
                fitness=0.0,  # Will be evaluated later
                age=winner.age + 1,
                generation=winner.generation + 1
            )
            selected.append(new_individual)
        
        return selected
    
    def _genetic_crossover(self, population: List[Individual]) -> List[Individual]:
        """Perform crossover operation"""
        offspring = []
        
        for i in range(0, len(population) - 1, 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[0]
            
            if random.random() < self.crossover_rate:
                child1_params = {}
                child2_params = {}
                
                for param_name in parent1.parameters.keys():
                    if random.random() < 0.5:
                        child1_params[param_name] = parent1.parameters[param_name]
                        child2_params[param_name] = parent2.parameters[param_name]
                    else:
                        child1_params[param_name] = parent2.parameters[param_name]
                        child2_params[param_name] = parent1.parameters[param_name]
                
                child1 = Individual(parameters=child1_params, generation=parent1.generation + 1)
                child2 = Individual(parameters=child2_params, generation=parent2.generation + 1)
                
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])
        
        return offspring
    
    def _genetic_mutation(self, population: List[Individual]) -> List[Individual]:
        """Perform mutation operation"""
        for individual in population:
            for param_name, bounds in self.parameter_bounds.items():
                if random.random() < self.mutation_rate:
                    if bounds.param_type == ParameterType.CONTINUOUS:
                        # Gaussian mutation
                        current_value = individual.parameters[param_name]
                        mutation_range = (bounds.max_value - bounds.min_value) * 0.1
                        new_value = current_value + random.gauss(0, mutation_range)
                        new_value = max(bounds.min_value, min(bounds.max_value, new_value))
                        individual.parameters[param_name] = new_value
                        
                    elif bounds.param_type == ParameterType.INTEGER:
                        # Random integer mutation
                        individual.parameters[param_name] = random.randint(
                            int(bounds.min_value), int(bounds.max_value)
                        )
                        
                    elif bounds.param_type == ParameterType.CATEGORICAL:
                        # Random category mutation
                        individual.parameters[param_name] = random.choice(bounds.categories)
                        
                    elif bounds.param_type == ParameterType.BOOLEAN:
                        # Boolean flip
                        individual.parameters[param_name] = not individual.parameters[param_name]
        
        return population
    
    def _initialize_velocity(self) -> Dict[str, float]:
        """Initialize particle velocity for PSO"""
        velocity = {}
        for name, bounds in self.parameter_bounds.items():
            if bounds.param_type in [ParameterType.CONTINUOUS, ParameterType.INTEGER]:
                max_vel = (bounds.max_value - bounds.min_value) * 0.1
                velocity[name] = random.uniform(-max_vel, max_vel)
            else:
                velocity[name] = 0.0
        return velocity
    
    def _update_particle(self, particle: Dict[str, Any], global_best: Dict[str, Any]):
        """Update particle position and velocity for PSO"""
        w = self.pso_params['w']
        c1 = self.pso_params['c1']
        c2 = self.pso_params['c2']
        
        for param_name, bounds in self.parameter_bounds.items():
            if bounds.param_type in [ParameterType.CONTINUOUS, ParameterType.INTEGER]:
                # Update velocity
                r1, r2 = random.random(), random.random()
                cognitive = c1 * r1 * (particle['best_position'][param_name] - particle['position'][param_name])
                social = c2 * r2 * (global_best[param_name] - particle['position'][param_name])
                
                particle['velocity'][param_name] = (w * particle['velocity'][param_name] + 
                                                  cognitive + social)
                
                # Update position
                new_pos = particle['position'][param_name] + particle['velocity'][param_name]
                
                # Apply bounds
                new_pos = max(bounds.min_value, min(bounds.max_value, new_pos))
                
                if bounds.param_type == ParameterType.INTEGER:
                    new_pos = int(round(new_pos))
                
                particle['position'][param_name] = new_pos
    
    def _generate_neighbor(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Generate neighbor solution for simulated annealing"""
        neighbor = solution.copy()
        
        # Randomly modify one parameter
        param_name = random.choice(list(self.parameter_bounds.keys()))
        bounds = self.parameter_bounds[param_name]
        
        if bounds.param_type == ParameterType.CONTINUOUS:
            current_value = neighbor[param_name]
            perturbation_range = (bounds.max_value - bounds.min_value) * 0.05
            new_value = current_value + random.gauss(0, perturbation_range)
            neighbor[param_name] = max(bounds.min_value, min(bounds.max_value, new_value))
            
        elif bounds.param_type == ParameterType.INTEGER:
            neighbor[param_name] = random.randint(int(bounds.min_value), int(bounds.max_value))
            
        elif bounds.param_type == ParameterType.CATEGORICAL:
            neighbor[param_name] = random.choice(bounds.categories)
            
        elif bounds.param_type == ParameterType.BOOLEAN:
            neighbor[param_name] = not neighbor[param_name]
        
        return neighbor
    
    def _get_scipy_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds in SciPy format for continuous parameters"""
        bounds = []
        for bounds_obj in self.parameter_bounds.values():
            if bounds_obj.param_type == ParameterType.CONTINUOUS:
                bounds.append((bounds_obj.min_value, bounds_obj.max_value))
            elif bounds_obj.param_type == ParameterType.INTEGER:
                bounds.append((bounds_obj.min_value, bounds_obj.max_value))
            else:
                # For categorical/boolean, use dummy bounds
                bounds.append((0, 1))
        return bounds
    
    def _vector_to_parameters(self, vector: List[float]) -> Dict[str, Any]:
        """Convert optimization vector back to parameter dictionary"""
        parameters = {}
        i = 0
        
        for name, bounds in self.parameter_bounds.items():
            if bounds.param_type == ParameterType.CONTINUOUS:
                parameters[name] = vector[i]
                i += 1
            elif bounds.param_type == ParameterType.INTEGER:
                parameters[name] = int(round(vector[i]))
                i += 1
            elif bounds.param_type == ParameterType.CATEGORICAL:
                # Map to category index
                index = int(round(vector[i] * (len(bounds.categories) - 1)))
                parameters[name] = bounds.categories[index]
                i += 1
            elif bounds.param_type == ParameterType.BOOLEAN:
                parameters[name] = vector[i] > 0.5
                i += 1
        
        return parameters
    
    async def _basic_differential_evolution(
        self,
        fitness_function: Callable[[Dict[str, Any]], float],
        max_evaluations: Optional[int],
        convergence_threshold: float,
        early_stopping_generations: int
    ) -> OptimizationResult:
        """Basic differential evolution implementation"""
        population = self._initialize_population()
        
        # Evaluate initial population
        for individual in population:
            individual.fitness = fitness_function(individual.parameters)
        
        convergence_history = []
        evaluations = len(population)
        generations = 0
        
        while generations < self.max_generations:
            new_population = []
            
            for i, target in enumerate(population):
                # Select three random individuals different from target
                candidates = [ind for j, ind in enumerate(population) if j != i]
                if len(candidates) >= 3:
                    a, b, c = random.sample(candidates, 3)
                    
                    # Create donor vector
                    donor_params = {}
                    for param_name, bounds in self.parameter_bounds.items():
                        if bounds.param_type == ParameterType.CONTINUOUS:
                            donor_value = (a.parameters[param_name] + 
                                         self.de_params['f'] * 
                                         (b.parameters[param_name] - c.parameters[param_name]))
                            donor_value = max(bounds.min_value, min(bounds.max_value, donor_value))
                            donor_params[param_name] = donor_value
                        else:
                            donor_params[param_name] = a.parameters[param_name]
                    
                    # Crossover
                    trial_params = {}
                    for param_name in target.parameters.keys():
                        if random.random() < self.de_params['cr']:
                            trial_params[param_name] = donor_params[param_name]
                        else:
                            trial_params[param_name] = target.parameters[param_name]
                    
                    # Evaluate trial individual
                    trial_fitness = fitness_function(trial_params)
                    evaluations += 1
                    
                    # Selection
                    if trial_fitness > target.fitness:
                        new_individual = Individual(
                            parameters=trial_params,
                            fitness=trial_fitness,
                            generation=generations + 1
                        )
                        new_population.append(new_individual)
                    else:
                        new_population.append(target)
                    
                    if max_evaluations and evaluations >= max_evaluations:
                        break
                else:
                    new_population.append(target)
            
            population = new_population
            best_fitness = max(ind.fitness for ind in population)
            convergence_history.append(best_fitness)
            
            generations += 1
            
            if max_evaluations and evaluations >= max_evaluations:
                break
        
        best_individual = max(population, key=lambda x: x.fitness)
        
        return OptimizationResult(
            best_parameters=best_individual.parameters,
            best_fitness=best_individual.fitness,
            generations=generations,
            evaluations=evaluations,
            convergence_history=convergence_history,
            algorithm_used=OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION,
            execution_time=0.0
        )
    
    async def _store_optimization_result(self, result: OptimizationResult):
        """Store optimization result in database"""
        try:
            conn = self._get_db_connection()
            try:
                conn.execute("""
                    INSERT INTO optimization_runs 
                    (algorithm, parameters, bounds, best_fitness, best_parameters, 
                     generations, evaluations, execution_time, convergence_history, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.algorithm_used.value,
                    json.dumps({name: bounds.__dict__ for name, bounds in self.parameter_bounds.items()}),
                    json.dumps({name: bounds.__dict__ for name, bounds in self.parameter_bounds.items()}),
                    result.best_fitness,
                    json.dumps(result.best_parameters),
                    result.generations,
                    result.evaluations,
                    result.execution_time,
                    json.dumps(result.convergence_history),
                    json.dumps(result.metadata)
                ))
                conn.commit()
            finally:
                conn.close()
                
        except Exception as e:
            logger.error("Error storing optimization result: %s", e)
    
    def _create_default_result(self, algorithm: OptimizationAlgorithm) -> OptimizationResult:
        """Create default result when optimization fails"""
        default_params = {}
        for name, bounds in self.parameter_bounds.items():
            if bounds.param_type == ParameterType.CONTINUOUS:
                default_params[name] = (bounds.min_value + bounds.max_value) / 2
            elif bounds.param_type == ParameterType.INTEGER:
                default_params[name] = int((bounds.min_value + bounds.max_value) / 2)
            elif bounds.param_type == ParameterType.CATEGORICAL:
                default_params[name] = bounds.categories[0]
            elif bounds.param_type == ParameterType.BOOLEAN:
                default_params[name] = False
        
        return OptimizationResult(
            best_parameters=default_params,
            best_fitness=0.0,
            generations=0,
            evaluations=0,
            convergence_history=[],
            algorithm_used=algorithm,
            execution_time=0.0,
            metadata={'error': 'Optimization failed, using default values'}
        )
    
    async def get_optimization_history(self, limit: int = 50) -> List[OptimizationResult]:
        """Get recent optimization history"""
        try:
            conn = self._get_db_connection()
            try:
                cursor = conn.execute("""
                    SELECT * FROM optimization_runs 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                results = []
                
                for row in rows:
                    result = OptimizationResult(
                        best_parameters=json.loads(row['best_parameters']),
                        best_fitness=row['best_fitness'],
                        generations=row['generations'],
                        evaluations=row['evaluations'],
                        convergence_history=json.loads(row['convergence_history']),
                        algorithm_used=OptimizationAlgorithm(row['algorithm']),
                        execution_time=row['execution_time'],
                        metadata=json.loads(row['metadata'] or '{}')
                    )
                    results.append(result)
                
                return results
            finally:
                conn.close()
                
        except Exception as e:
            logger.error("Error getting optimization history: %s", e)
            return []
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization system status"""
        try:
            recent_history = await self.get_optimization_history(10)
            
            status = {
                'scipy_available': SCIPY_AVAILABLE,
                'optimization_stats': self.optimization_stats,
                'recent_optimizations': len(recent_history),
                'algorithms_used': list(set(r.algorithm_used.value for r in recent_history)),
                'average_fitness': sum(r.best_fitness for r in recent_history) / len(recent_history) if recent_history else 0.0,
                'population_size': self.population_size,
                'max_generations': self.max_generations,
                'current_parameters': {
                    'mutation_rate': self.mutation_rate,
                    'crossover_rate': self.crossover_rate,
                    'elitism_ratio': self.elitism_ratio
                }
            }
            
            return status
            
        except Exception as e:
            logger.error("Error getting optimization status: %s", e)
            return {'error': str(e), 'scipy_available': SCIPY_AVAILABLE}