"""
Capability Matrix Generator for MCP Swarm Intelligence Server

This module generates comprehensive capability matrices and analyzes agent
intersections to optimize agent selection and coordination.
"""

import networkx as nx
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .config_scanner import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class CapabilityGap:
    """Represents a gap in capability coverage"""
    capability: str
    description: str
    required_domain: Optional[str] = None
    severity: str = "medium"  # low, medium, high, critical
    suggested_agents: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.suggested_agents is None:
            self.suggested_agents = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilityOverlap:
    """Represents overlapping capabilities between agents"""
    capability: str
    agents: List[str]
    overlap_score: float
    redundancy_level: str = "normal"  # low, normal, high, excessive
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentCluster:
    """Represents a cluster of related agents"""
    cluster_id: str
    agents: List[str]
    shared_capabilities: List[str]
    coordination_strength: float
    cluster_type: str  # functional, domain, intersection
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilityMatrix:
    """Complete capability matrix with analysis results"""
    agents: Dict[str, AgentConfig]
    capability_map: Dict[str, List[str]]  # capability -> agents
    intersection_map: Dict[str, List[str]]  # agent -> intersecting agents
    domain_map: Dict[str, List[str]]  # domain -> agents
    capability_gaps: List[CapabilityGap]
    capability_overlaps: List[CapabilityOverlap]
    agent_clusters: List[AgentCluster]
    graph: Optional[Any] = None  # NetworkX graph
    coverage_score: float = 0.0
    redundancy_score: float = 0.0
    coordination_efficiency: float = 0.0
    generation_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.generation_timestamp is None:
            self.generation_timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['graph'] = None  # NetworkX graphs aren't serializable
        if self.generation_timestamp:
            result['generation_timestamp'] = self.generation_timestamp.isoformat()
        return result


class CapabilityMatrixGenerator:
    """
    Generates comprehensive capability matrices and analyzes agent ecosystems.
    
    This class provides detailed analysis of agent capabilities, intersections,
    gaps, and optimization opportunities for the agent ecosystem.
    """
    
    def __init__(self):
        self.capability_graph: Optional[nx.Graph] = None
        self.intersection_graph: Optional[nx.Graph] = None
        
        # Standard capability categories for analysis
        self.capability_categories = {
            'development': [
                'python_implementation', 'code_quality', 'refactoring',
                'mcp_features', 'api_development', 'debugging'
            ],
            'architecture': [
                'system_design', 'performance_optimization', 'security_analysis',
                'scalability_planning', 'integration_patterns'
            ],
            'testing': [
                'unit_testing', 'integration_testing', 'performance_testing',
                'automated_testing', 'test_coverage'
            ],
            'swarm_intelligence': [
                'aco_algorithms', 'pso_algorithms', 'collective_decision_making',
                'coordination_patterns', 'swarm_optimization'
            ],
            'memory_management': [
                'persistent_memory', 'memory_optimization', 'data_persistence',
                'cross_session_learning', 'knowledge_storage'
            ],
            'documentation': [
                'technical_writing', 'api_documentation', 'user_guides',
                'code_documentation', 'knowledge_management'
            ],
            'project_management': [
                'task_coordination', 'timeline_management', 'resource_allocation',
                'quality_assurance', 'progress_tracking'
            ]
        }
    
    async def generate_capability_matrix(
        self, 
        agents: Dict[str, AgentConfig]
    ) -> CapabilityMatrix:
        """
        Generate comprehensive capability matrix from agent configurations.
        
        Args:
            agents: Dictionary of agent configurations
            
        Returns:
            Complete capability matrix with analysis
        """
        logger.info("Generating capability matrix for %d agents", len(agents))
        
        # Build capability and domain mappings
        capability_map = self._build_capability_map(agents)
        domain_map = self._build_domain_map(agents)
        intersection_map = self._analyze_intersections(agents)
        
        # Build capability graph
        capability_graph = self._build_capability_graph(agents)
        
        # Analyze gaps and overlaps
        capability_gaps = self._identify_capability_gaps(capability_map)
        capability_overlaps = self._identify_capability_overlaps(capability_map)
        
        # Find agent clusters
        agent_clusters = await self._discover_agent_clusters(agents, intersection_map)
        
        # Calculate metrics
        coverage_score = self._calculate_coverage_score(capability_map)
        redundancy_score = self._calculate_redundancy_score(capability_overlaps)
        coordination_efficiency = self._calculate_coordination_efficiency(
            agents, intersection_map
        )
        
        matrix = CapabilityMatrix(
            agents=agents,
            capability_map=capability_map,
            intersection_map=intersection_map,
            domain_map=domain_map,
            capability_gaps=capability_gaps,
            capability_overlaps=capability_overlaps,
            agent_clusters=agent_clusters,
            graph=capability_graph,
            coverage_score=coverage_score,
            redundancy_score=redundancy_score,
            coordination_efficiency=coordination_efficiency
        )
        
        logger.info("Capability matrix generated successfully")
        return matrix
    
    def _build_capability_map(self, agents: Dict[str, AgentConfig]) -> Dict[str, List[str]]:
        """Build mapping from capabilities to agents that provide them"""
        capability_map = {}
        
        for agent_name, config in agents.items():
            for capability in config.capabilities:
                if capability not in capability_map:
                    capability_map[capability] = []
                capability_map[capability].append(agent_name)
        
        return capability_map
    
    def _build_domain_map(self, agents: Dict[str, AgentConfig]) -> Dict[str, List[str]]:
        """Build mapping from domains to agents in those domains"""
        domain_map = {}
        
        for agent_name, config in agents.items():
            if config.domain:
                if config.domain not in domain_map:
                    domain_map[config.domain] = []
                domain_map[config.domain].append(agent_name)
        
        return domain_map
    
    def _analyze_intersections(
        self, 
        agents: Dict[str, AgentConfig]
    ) -> Dict[str, List[str]]:
        """Analyze intersection patterns between agents"""
        intersection_map = {}
        
        for agent_name, config in agents.items():
            intersection_map[agent_name] = []
            
            # Direct intersections from configuration
            for intersection_name in config.intersections:
                # Handle partial matches (e.g., 'code' for 'specialists.code')
                matching_agents = [
                    name for name in agents.keys()
                    if name.endswith(f".{intersection_name}") or name == intersection_name
                ]
                intersection_map[agent_name].extend(matching_agents)
            
            # Capability-based intersections
            for other_name, other_config in agents.items():
                if other_name != agent_name:
                    shared_capabilities = set(config.capabilities) & set(other_config.capabilities)
                    if shared_capabilities and other_name not in intersection_map[agent_name]:
                        intersection_map[agent_name].append(other_name)
        
        return intersection_map
    
    def _build_capability_graph(self, agents: Dict[str, AgentConfig]) -> nx.Graph:
        """Build graph representation of agent capabilities and relationships"""
        graph = nx.Graph()
        
        # Add agent nodes
        for agent_name, config in agents.items():
            graph.add_node(
                agent_name,
                node_type='agent',
                domain=config.domain,
                priority=config.priority,
                memory_enabled=config.memory_enabled,
                coordination_style=config.coordination_style
            )
        
        # Add capability nodes
        all_capabilities = set()
        for config in agents.values():
            all_capabilities.update(config.capabilities)
        
        for capability in all_capabilities:
            graph.add_node(
                capability,
                node_type='capability',
                category=self._get_capability_category(capability)
            )
        
        # Add edges between agents and capabilities
        for agent_name, config in agents.items():
            for capability in config.capabilities:
                graph.add_edge(agent_name, capability, edge_type='provides')
        
        # Add edges between intersecting agents
        for agent_name, config in agents.items():
            for intersection in config.intersections:
                # Find matching agents
                matching_agents = [
                    name for name in agents.keys()
                    if name.endswith(f".{intersection}") or name == intersection
                ]
                for matching_agent in matching_agents:
                    if matching_agent in agents:
                        graph.add_edge(
                            agent_name, 
                            matching_agent, 
                            edge_type='intersects'
                        )
        
        self.capability_graph = graph
        return graph
    
    def _get_capability_category(self, capability: str) -> str:
        """Get category for a capability"""
        for category, capabilities in self.capability_categories.items():
            if capability in capabilities:
                return category
        return 'other'
    
    def _identify_capability_gaps(
        self, 
        capability_map: Dict[str, List[str]]
    ) -> List[CapabilityGap]:
        """Identify gaps in capability coverage"""
        gaps = []
        
        # Check for missing standard capabilities
        all_standard_capabilities = set()
        for capabilities in self.capability_categories.values():
            all_standard_capabilities.update(capabilities)
        
        covered_capabilities = set(capability_map.keys())
        missing_capabilities = all_standard_capabilities - covered_capabilities
        
        for capability in missing_capabilities:
            category = self._get_capability_category(capability)
            severity = self._assess_gap_severity(capability, category)
            
            gap = CapabilityGap(
                capability=capability,
                description=f"Missing {category} capability: {capability}",
                required_domain=category,
                severity=severity
            )
            gaps.append(gap)
        
        # Check for under-covered capabilities (only one agent)
        for capability, agents in capability_map.items():
            if len(agents) == 1:
                category = self._get_capability_category(capability)
                severity = "medium" if category in ['development', 'testing'] else "low"
                
                gap = CapabilityGap(
                    capability=capability,
                    description=f"Under-covered capability: {capability} (only {agents[0]})",
                    severity=severity,
                    suggested_agents=[agents[0]]
                )
                gaps.append(gap)
        
        return gaps
    
    def _assess_gap_severity(self, capability: str, category: str) -> str:
        """Assess severity of a capability gap"""
        critical_capabilities = {
            'python_implementation', 'mcp_features', 'debugging',
            'system_design', 'security_analysis', 'automated_testing'
        }
        
        high_priority_categories = {'development', 'testing', 'security'}
        
        if capability in critical_capabilities:
            return "critical"
        elif category in high_priority_categories:
            return "high"
        else:
            return "medium"
    
    def _identify_capability_overlaps(
        self, 
        capability_map: Dict[str, List[str]]
    ) -> List[CapabilityOverlap]:
        """Identify overlapping capabilities between agents"""
        overlaps = []
        
        for capability, agents in capability_map.items():
            if len(agents) > 1:
                overlap_score = len(agents) / max(len(capability_map), 1)
                
                if len(agents) > 5:
                    redundancy_level = "excessive"
                elif len(agents) > 3:
                    redundancy_level = "high"
                elif len(agents) > 1:
                    redundancy_level = "normal"
                else:
                    redundancy_level = "low"
                
                overlap = CapabilityOverlap(
                    capability=capability,
                    agents=agents,
                    overlap_score=overlap_score,
                    redundancy_level=redundancy_level
                )
                overlaps.append(overlap)
        
        return overlaps
    
    async def _discover_agent_clusters(
        self, 
        agents: Dict[str, AgentConfig],
        intersection_map: Dict[str, List[str]]
    ) -> List[AgentCluster]:
        """Discover clusters of related agents"""
        clusters = []
        processed_agents = set()
        
        # Domain-based clusters
        domain_map = self._build_domain_map(agents)
        for domain, domain_agents in domain_map.items():
            if len(domain_agents) > 1:
                shared_capabilities = self._find_shared_capabilities(
                    domain_agents, agents
                )
                coordination_strength = self._calculate_cluster_coordination(
                    domain_agents, intersection_map
                )
                
                cluster = AgentCluster(
                    cluster_id=f"domain_{domain}",
                    agents=domain_agents,
                    shared_capabilities=shared_capabilities,
                    coordination_strength=coordination_strength,
                    cluster_type="domain"
                )
                clusters.append(cluster)
                processed_agents.update(domain_agents)
        
        # Intersection-based clusters
        for agent_name, intersecting_agents in intersection_map.items():
            if agent_name not in processed_agents and len(intersecting_agents) >= 2:
                cluster_agents = [agent_name] + intersecting_agents[:3]  # Limit cluster size
                shared_capabilities = self._find_shared_capabilities(
                    cluster_agents, agents
                )
                coordination_strength = self._calculate_cluster_coordination(
                    cluster_agents, intersection_map
                )
                
                if coordination_strength > 0.5:  # Only strong clusters
                    cluster = AgentCluster(
                        cluster_id=f"intersection_{agent_name}",
                        agents=cluster_agents,
                        shared_capabilities=shared_capabilities,
                        coordination_strength=coordination_strength,
                        cluster_type="intersection"
                    )
                    clusters.append(cluster)
                    processed_agents.update(cluster_agents)
        
        return clusters
    
    def _find_shared_capabilities(
        self, 
        agent_names: List[str], 
        agents: Dict[str, AgentConfig]
    ) -> List[str]:
        """Find capabilities shared by all agents in the list"""
        if not agent_names:
            return []
        
        shared = set(agents[agent_names[0]].capabilities)
        for agent_name in agent_names[1:]:
            if agent_name in agents:
                shared &= set(agents[agent_name].capabilities)
        
        return list(shared)
    
    def _calculate_cluster_coordination(
        self, 
        agent_names: List[str],
        intersection_map: Dict[str, List[str]]
    ) -> float:
        """Calculate coordination strength within a cluster"""
        if len(agent_names) <= 1:
            return 0.0
        
        total_connections = 0
        possible_connections = len(agent_names) * (len(agent_names) - 1) // 2
        
        for i, agent1 in enumerate(agent_names):
            for agent2 in agent_names[i+1:]:
                if agent1 in intersection_map and agent2 in intersection_map[agent1]:
                    total_connections += 1
                elif agent2 in intersection_map and agent1 in intersection_map[agent2]:
                    total_connections += 1
        
        return total_connections / max(possible_connections, 1)
    
    def _calculate_coverage_score(self, capability_map: Dict[str, List[str]]) -> float:
        """Calculate capability coverage score"""
        all_standard_capabilities = set()
        for capabilities in self.capability_categories.values():
            all_standard_capabilities.update(capabilities)
        
        covered_capabilities = set(capability_map.keys()) & all_standard_capabilities
        coverage = len(covered_capabilities) / max(len(all_standard_capabilities), 1)
        
        return round(coverage, 3)
    
    def _calculate_redundancy_score(self, overlaps: List[CapabilityOverlap]) -> float:
        """Calculate redundancy score"""
        if not overlaps:
            return 0.0
        
        total_redundancy = sum(
            len(overlap.agents) - 1 for overlap in overlaps 
            if overlap.redundancy_level in ['high', 'excessive']
        )
        
        total_capabilities = len(overlaps)
        redundancy = total_redundancy / max(total_capabilities, 1)
        
        return round(redundancy, 3)
    
    def _calculate_coordination_efficiency(
        self, 
        agents: Dict[str, AgentConfig],
        intersection_map: Dict[str, List[str]]
    ) -> float:
        """Calculate overall coordination efficiency"""
        if not agents:
            return 0.0
        
        total_intersections = sum(len(intersections) for intersections in intersection_map.values())
        max_possible_intersections = len(agents) * (len(agents) - 1)
        
        intersection_ratio = total_intersections / max(max_possible_intersections, 1)
        
        # Normalize to reasonable range
        efficiency = min(intersection_ratio * 2, 1.0)
        
        return round(efficiency, 3)
    
    def get_capability_recommendations(
        self, 
        matrix: CapabilityMatrix,
        agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get recommendations for improving capability coverage"""
        recommendations = {
            "critical_gaps": [],
            "redundancy_reductions": [],
            "coordination_improvements": [],
            "agent_specific": {}
        }
        
        # Critical gaps
        critical_gaps = [
            gap for gap in matrix.capability_gaps 
            if gap.severity in ['critical', 'high']
        ]
        recommendations["critical_gaps"] = [gap.to_dict() for gap in critical_gaps]
        
        # Redundancy reductions
        excessive_overlaps = [
            overlap for overlap in matrix.capability_overlaps
            if overlap.redundancy_level == 'excessive'
        ]
        recommendations["redundancy_reductions"] = [
            overlap.to_dict() for overlap in excessive_overlaps
        ]
        
        # Coordination improvements
        weak_clusters = [
            cluster for cluster in matrix.agent_clusters
            if cluster.coordination_strength < 0.5
        ]
        recommendations["coordination_improvements"] = [
            cluster.to_dict() for cluster in weak_clusters
        ]
        
        # Agent-specific recommendations
        if agent_name and agent_name in matrix.agents:
            agent_config = matrix.agents[agent_name]
            intersections = matrix.intersection_map.get(agent_name, [])
            
            recommendations["agent_specific"][agent_name] = {
                "current_intersections": intersections,
                "suggested_intersections": self._suggest_intersections(
                    agent_config, matrix
                ),
                "capability_gaps": self._find_agent_capability_gaps(
                    agent_config, matrix
                )
            }
        
        return recommendations
    
    def _suggest_intersections(
        self, 
        agent_config: AgentConfig, 
        matrix: CapabilityMatrix
    ) -> List[str]:
        """Suggest additional intersections for an agent"""
        suggestions = []
        
        # Find agents with complementary capabilities
        for other_name, other_config in matrix.agents.items():
            if other_name != agent_config.name:
                # Check for complementary domains
                if (agent_config.domain != other_config.domain and 
                    other_name not in matrix.intersection_map.get(agent_config.name, [])):
                    
                    # Check for some capability overlap
                    shared_caps = set(agent_config.capabilities) & set(other_config.capabilities)
                    if len(shared_caps) > 0:
                        suggestions.append(other_name)
        
        return suggestions[:3]  # Limit suggestions
    
    def _find_agent_capability_gaps(
        self, 
        agent_config: AgentConfig, 
        matrix: CapabilityMatrix
    ) -> List[str]:
        """Find capability gaps for a specific agent"""
        domain_category = agent_config.domain
        if domain_category in self.capability_categories:
            standard_capabilities = set(self.capability_categories[domain_category])
            agent_capabilities = set(agent_config.capabilities)
            gaps = list(standard_capabilities - agent_capabilities)
            return gaps[:5]  # Limit gaps
        
        return []


async def main():
    """Example usage of CapabilityMatrixGenerator"""
    from .config_scanner import AgentConfigScanner
    from pathlib import Path
    
    # Initialize scanner and generator
    scanner = AgentConfigScanner(Path("../../agent-config"))
    generator = CapabilityMatrixGenerator()
    
    # Scan for configurations
    agents = await scanner.scan_agent_configurations()
    
    # Generate capability matrix
    matrix = await generator.generate_capability_matrix(agents)
    
    print(f"Capability Matrix Generated:")
    print(f"  Coverage Score: {matrix.coverage_score}")
    print(f"  Redundancy Score: {matrix.redundancy_score}")
    print(f"  Coordination Efficiency: {matrix.coordination_efficiency}")
    print(f"  Capability Gaps: {len(matrix.capability_gaps)}")
    print(f"  Agent Clusters: {len(matrix.agent_clusters)}")
    
    # Get recommendations
    recommendations = generator.get_capability_recommendations(matrix)
    print(f"  Critical Gaps: {len(recommendations['critical_gaps'])}")
    print(f"  Redundancy Issues: {len(recommendations['redundancy_reductions'])}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())