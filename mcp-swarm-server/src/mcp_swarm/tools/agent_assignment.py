"""
Agent Assignment MCP Tool

This module implements the optimal agent assignment tool for the MCP Swarm Intelligence Server.
It provides MCP-compliant tool interface for assigning tasks to agents using swarm intelligence
algorithms including ACO optimization, multi-criteria decision analysis, and load balancing.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ..swarm import SwarmCoordinator, Agent, Task
from .mcda import MCDAAnalyzer, Alternative
from .load_balancer import LoadBalancer
from .fuzzy_matcher import FuzzyCapabilityMatcher
from .explanation import AssignmentExplainer, AssignmentReason

logger = logging.getLogger(__name__)


class AgentAssignmentTool:
    """MCP tool for optimal agent assignment using swarm intelligence."""
    
    name = "assign_agents"
    description = "Assign tasks to optimal agents using swarm intelligence algorithms"
    
    parameters = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "List of tasks to assign to agents",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Unique task identifier"},
                        "requirements": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "Required capabilities for task completion"
                        },
                        "complexity": {
                            "type": "number", 
                            "minimum": 0.0, 
                            "maximum": 1.0,
                            "description": "Task complexity level (0.0 = simple, 1.0 = very complex)"
                        },
                        "priority": {
                            "type": "integer", 
                            "minimum": 1, 
                            "maximum": 5,
                            "description": "Task priority (1 = low, 5 = critical)"
                        },
                        "deadline": {
                            "type": "number", 
                            "description": "Unix timestamp for task deadline (optional)"
                        },
                        "estimated_duration": {
                            "type": "number",
                            "minimum": 0.1,
                            "description": "Estimated task duration in hours"
                        }
                    },
                    "required": ["id", "requirements", "complexity", "priority"]
                }
            },
            "constraints": {
                "type": "object",
                "description": "Assignment constraints and preferences",
                "properties": {
                    "max_load_per_agent": {
                        "type": "number", 
                        "minimum": 0.1, 
                        "maximum": 1.0, 
                        "default": 0.8,
                        "description": "Maximum load threshold per agent"
                    },
                    "require_specific_agent": {
                        "type": "string", 
                        "description": "Force assignment to specific agent ID"
                    },
                    "exclude_agents": {
                        "type": "array", 
                        "items": {"type": "string"}, 
                        "default": [],
                        "description": "List of agent IDs to exclude from assignment"
                    },
                    "prefer_balanced_load": {
                        "type": "boolean",
                        "default": True,
                        "description": "Prefer balanced load distribution across agents"
                    },
                    "optimization_iterations": {
                        "type": "integer",
                        "minimum": 10,
                        "maximum": 500,
                        "default": 100,
                        "description": "Number of ACO optimization iterations"
                    }
                }
            },
            "options": {
                "type": "object",
                "description": "Additional options for assignment",
                "properties": {
                    "explain_assignments": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include detailed assignment explanations"
                    },
                    "include_alternatives": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include alternative agent options for each task"
                    },
                    "use_fuzzy_matching": {
                        "type": "boolean",
                        "default": True,
                        "description": "Use fuzzy logic for capability matching"
                    },
                    "real_time_load_check": {
                        "type": "boolean",
                        "default": True,
                        "description": "Check real-time agent loads before assignment"
                    }
                }
            }
        },
        "required": ["tasks"]
    }
    
    def __init__(self, swarm_coordinator: SwarmCoordinator, agent_registry):
        """
        Initialize agent assignment tool.
        
        Args:
            swarm_coordinator: Main swarm coordination engine
            agent_registry: Registry for agent discovery and management
        """
        super().__init__()
        self.swarm = swarm_coordinator
        self.agent_registry = agent_registry
        
        # Initialize analysis components
        self.mcda_analyzer = MCDAAnalyzer()
        self.load_balancer = LoadBalancer()
        self.fuzzy_matcher = FuzzyCapabilityMatcher()
        self.explainer = AssignmentExplainer()
        
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent assignment with optimization.
        
        Args:
            arguments: Tool execution arguments
            
        Returns:
            Dictionary containing assignment results with explanations
        """
        try:
            # Parse and validate input
            tasks_data = arguments["tasks"]
            constraints = arguments.get("constraints", {})
            options = arguments.get("options", {})
            
            # Convert task data to Task objects
            tasks = []
            for task_data in tasks_data:
                task = Task(
                    id=task_data["id"],
                    requirements=task_data["requirements"],
                    complexity=task_data["complexity"],
                    priority=task_data["priority"],
                    deadline=datetime.fromtimestamp(task_data["deadline"]) if task_data.get("deadline") else None,
                    estimated_duration=task_data.get("estimated_duration", 1.0)
                )
                tasks.append(task)
            
            logger.info("Processing assignment request for %d tasks", len(tasks))
            
            # Get available agents
            available_agents = await self._get_available_agents(constraints)
            
            if not available_agents:
                return {
                    "error": "No available agents found matching constraints",
                    "tasks_requested": len(tasks),
                    "constraints": constraints
                }
            
            # Perform real-time load check if requested
            if options.get("real_time_load_check", True):
                available_agents = await self._filter_by_load(available_agents, constraints)
            
            # Perform assignment using ACO optimization
            assignment_result = await self._perform_assignment(
                available_agents, tasks, constraints, options
            )
            
            # Generate explanations if requested
            if options.get("explain_assignments", True):
                assignment_result["explanations"] = await self._generate_explanations(
                    assignment_result["assignments"], available_agents, tasks, options
                )
            
            # Include alternatives if requested
            if options.get("include_alternatives", False):
                assignment_result["alternatives"] = await self._generate_alternatives(
                    available_agents, tasks, assignment_result["assignments"]
                )
            
            # Calculate performance metrics
            assignment_result["metrics"] = await self._calculate_metrics(
                assignment_result["assignments"], available_agents, tasks
            )
            
            logger.info("Assignment completed: %d tasks assigned", len(assignment_result['assignments']))
            
            return assignment_result
            
        except Exception as e:
            logger.error("Error in agent assignment: %s", str(e))
            return {
                "error": f"Assignment failed: {str(e)}",
                "tasks_requested": len(arguments.get("tasks", [])),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_available_agents(self, constraints: Dict[str, Any]) -> List[Agent]:
        """Get available agents applying basic constraints."""
        all_agents = await self.agent_registry.get_available_agents()
        
        # Apply exclusion constraints
        exclude_agents = set(constraints.get("exclude_agents", []))
        filtered_agents = [agent for agent in all_agents if agent.id not in exclude_agents]
        
        # Apply specific agent requirement
        specific_agent = constraints.get("require_specific_agent")
        if specific_agent:
            filtered_agents = [agent for agent in filtered_agents if agent.id == specific_agent]
        
        # Filter by availability
        available_agents = [agent for agent in filtered_agents if agent.availability]
        
        return available_agents
    
    async def _filter_by_load(self, agents: List[Agent], constraints: Dict[str, Any]) -> List[Agent]:
        """Filter agents by current load constraints."""
        max_load = constraints.get("max_load_per_agent", 0.8)
        
        # Update load balancer with current agent states
        await self.load_balancer.update_agent_loads(agents)
        
        # Filter by load capacity
        filtered_agents = []
        for agent in agents:
            if await self.load_balancer.can_accept_task(agent.id, max_load):
                filtered_agents.append(agent)
        
        return filtered_agents
    
    async def _perform_assignment(
        self, 
        agents: List[Agent], 
        tasks: List[Task], 
        constraints: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform optimal assignment using swarm intelligence."""
        
        # Configure ACO optimizer
        iterations = constraints.get("optimization_iterations", 100)
        
        # Use fuzzy matching for capability assessment if requested
        if options.get("use_fuzzy_matching", True):
            # Store fuzzy scores for later use in explanations
            fuzzy_scores = {}
            for agent in agents:
                fuzzy_scores[agent.id] = await self._calculate_fuzzy_scores(agent, tasks)
        
        # Perform ACO optimization
        assignments = await self.swarm.assign_tasks(agents, tasks, max_iterations=iterations)
        
        # Calculate assignment quality metrics
        success_probability = await self._calculate_success_probability(assignments, agents, tasks)
        load_distribution = await self._calculate_load_distribution(assignments)
        
        return {
            "assignments": assignments,
            "success_probability": success_probability,
            "load_distribution": load_distribution,
            "optimization_iterations": iterations,
            "agents_considered": len(agents),
            "tasks_assigned": len([a for a in assignments.values() if a is not None])
        }
    
    async def _calculate_fuzzy_scores(self, agent: Agent, tasks: List[Task]) -> Dict[str, float]:
        """Calculate fuzzy capability scores for agent-task combinations."""
        fuzzy_scores = {}
        
        for task in tasks:
            matches = await self.fuzzy_matcher.match_capabilities(
                agent_capabilities={cap: 1.0 for cap in agent.capabilities},
                required_capabilities=task.requirements,
                task_complexity=task.complexity
            )
            
            # Aggregate match scores
            if matches:
                fuzzy_scores[task.id] = sum(match.match_degree for match in matches) / len(matches)
            else:
                fuzzy_scores[task.id] = 0.0
        
        return fuzzy_scores
    
    async def _generate_explanations(
        self, 
        assignments: Dict[str, str], 
        agents: List[Agent], 
        tasks: List[Task],
        options: Dict[str, Any]
    ) -> Dict[str, List[AssignmentReason]]:
        """Generate detailed explanations for assignments."""
        
        # Create agent and task lookup dictionaries
        agent_map = {agent.id: agent for agent in agents}
        task_map = {task.id: task for task in tasks}
        
        explanations = {}
        
        for task_id, agent_id in assignments.items():
            if agent_id and task_id in task_map and agent_id in agent_map:
                explanation = await self.explainer.generate_explanation(
                    agent=agent_map[agent_id],
                    task=task_map[task_id],
                    assignment_context={
                        "total_agents": len(agents),
                        "total_tasks": len(tasks),
                        "optimization_used": "ACO",
                        "fuzzy_matching": options.get("use_fuzzy_matching", True)
                    }
                )
                explanations[task_id] = explanation
        
        return explanations
    
    async def _generate_alternatives(
        self, 
        agents: List[Agent], 
        tasks: List[Task], 
        assignments: Dict[str, str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate alternative agent options for each task."""
        alternatives = {}
        
        for task in tasks:
            # Use MCDA to rank all suitable agents for this task
            suitable_agents = [
                agent for agent in agents 
                if any(req in agent.capabilities for req in task.requirements)
            ]
            
            if suitable_agents:
                # Create alternatives for MCDA analysis
                mcda_alternatives = []
                for agent in suitable_agents:
                    alternative = Alternative(
                        id=agent.id,
                        values={
                            "capability_match": self._calculate_capability_match(agent, task),
                            "current_load": agent.current_load,
                            "success_rate": agent.success_rate,
                            "response_time": 1.0 / (agent.average_completion_time + 0.1),
                            "expertise_level": sum(agent.specialty_weights.get(req, 0.5) for req in task.requirements) / len(task.requirements)
                        }
                    )
                    mcda_alternatives.append(alternative)
                
                # Analyze alternatives
                ranked_alternatives = await self.mcda_analyzer.analyze_alternatives(mcda_alternatives)
                
                # Format results
                alternatives[task.id] = [
                    {
                        "agent_id": alt_id,
                        "score": score,
                        "assigned": assignments.get(task.id) == alt_id
                    }
                    for alt_id, score in ranked_alternatives[:5]  # Top 5 alternatives
                ]
        
        return alternatives
    
    def _calculate_capability_match(self, agent: Agent, task: Task) -> float:
        """Calculate capability match score between agent and task."""
        required_caps = set(task.requirements)
        agent_caps = set(agent.capabilities)
        
        if not required_caps:
            return 1.0
        
        matched_caps = required_caps.intersection(agent_caps)
        return len(matched_caps) / len(required_caps)
    
    async def _calculate_success_probability(
        self, 
        assignments: Dict[str, Optional[str]], 
        agents: List[Agent], 
        tasks: List[Task]
    ) -> float:
        """Calculate overall success probability for assignments."""
        if not assignments:
            return 0.0
        
        agent_map = {agent.id: agent for agent in agents}
        task_map = {task.id: task for task in tasks}
        
        success_scores = []
        for task_id, agent_id in assignments.items():
            if agent_id and task_id in task_map and agent_id in agent_map:
                agent = agent_map[agent_id]
                task = task_map[task_id]
                
                # Calculate success score based on multiple factors
                capability_score = self._calculate_capability_match(agent, task)
                load_score = 1.0 - agent.current_load
                experience_score = agent.success_rate
                
                # Weight the factors
                success_score = (
                    capability_score * 0.4 +
                    load_score * 0.3 +
                    experience_score * 0.3
                )
                success_scores.append(success_score)
        
        return sum(success_scores) / len(success_scores) if success_scores else 0.0
    
    async def _calculate_load_distribution(
        self, 
        assignments: Dict[str, Optional[str]]
    ) -> Dict[str, Any]:
        """Calculate load distribution metrics."""
        agent_task_counts = {}
        for agent_id in assignments.values():
            if agent_id:
                agent_task_counts[agent_id] = agent_task_counts.get(agent_id, 0) + 1
        
        if not agent_task_counts:
            return {"balanced": True, "std_deviation": 0.0, "assignments_per_agent": {}}
        
        # Calculate statistics
        task_counts = list(agent_task_counts.values())
        mean_tasks = sum(task_counts) / len(task_counts)
        variance = sum((count - mean_tasks) ** 2 for count in task_counts) / len(task_counts)
        std_deviation = variance ** 0.5
        
        return {
            "balanced": std_deviation < 1.0,  # Consider balanced if std dev < 1 task
            "std_deviation": std_deviation,
            "mean_tasks_per_agent": mean_tasks,
            "assignments_per_agent": agent_task_counts
        }
    
    async def _calculate_metrics(
        self, 
        assignments: Dict[str, Optional[str]], 
        agents: List[Agent], 
        tasks: List[Task]
    ) -> Dict[str, Any]:
        """Calculate comprehensive assignment metrics."""
        total_tasks = len(tasks)
        assigned_tasks = len([a for a in assignments.values() if a is not None])
        unassigned_tasks = total_tasks - assigned_tasks
        
        # Calculate agent utilization
        utilized_agents = len(set(a for a in assignments.values() if a is not None))
        total_agents = len(agents)
        
        return {
            "assignment_rate": assigned_tasks / total_tasks if total_tasks > 0 else 0.0,
            "tasks_assigned": assigned_tasks,
            "tasks_unassigned": unassigned_tasks,
            "agent_utilization": utilized_agents / total_agents if total_agents > 0 else 0.0,
            "agents_utilized": utilized_agents,
            "agents_available": total_agents,
            "average_capability_match": await self._calculate_average_capability_match(assignments, agents, tasks)
        }
    
    async def _calculate_average_capability_match(
        self, 
        assignments: Dict[str, Optional[str]], 
        agents: List[Agent], 
        tasks: List[Task]
    ) -> float:
        """Calculate average capability match across all assignments."""
        agent_map = {agent.id: agent for agent in agents}
        task_map = {task.id: task for task in tasks}
        
        matches = []
        for task_id, agent_id in assignments.items():
            if agent_id and task_id in task_map and agent_id in agent_map:
                match_score = self._calculate_capability_match(agent_map[agent_id], task_map[task_id])
                matches.append(match_score)
        
        return sum(matches) / len(matches) if matches else 0.0