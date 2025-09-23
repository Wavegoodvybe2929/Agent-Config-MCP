"""
Hive Mind Query Tool - MCP Interface for Collective Knowledge Management

Implements comprehensive MCP tool interface for hive mind intelligence.
Integrates semantic search, knowledge synthesis, confidence aggregation, and real-time updates.

According to orchestrator routing:
- Primary: hive_mind_specialist.md
- Secondary: mcp_specialist.md, memory_management_specialist.md
"""
import asyncio
import logging
import time
from typing import List, Dict, Any

# Import MCP components (with fallbacks for development)
# MCP types will be used at runtime for proper MCP server integration

# Import our hive mind components
from .semantic_search import SemanticSearchEngine, SearchConfig
from .knowledge_synthesis import KnowledgeSynthesisEngine, KnowledgeSource
from .confidence_aggregation import ConfidenceAggregationSystem, ConfidenceSource, ConfidenceMethod
from .knowledge_updater import KnowledgeUpdater, KnowledgeUpdate, UpdateType, UpdatePriority

logger = logging.getLogger(__name__)

class HiveMindQueryTool:
    """
    Comprehensive MCP tool for hive mind knowledge management.
    
    Features:
    - Semantic search across collective knowledge base
    - Multi-source knowledge synthesis with conflict resolution
    - Confidence aggregation and uncertainty quantification
    - Real-time knowledge updates with consistency validation
    - Advanced query processing with context awareness
    - Performance monitoring and optimization
    """
    
    def __init__(self, db_path: str):
        """
        Initialize hive mind query tool with database path.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.semantic_search = SemanticSearchEngine(db_path)
        self.synthesis_engine = KnowledgeSynthesisEngine(db_path)
        self.confidence_system = ConfidenceAggregationSystem(db_path)
        self.updater = KnowledgeUpdater(db_path)
        
        # Tool definitions for MCP
        self.tools = self._define_tools()
    
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define MCP tools for hive mind operations."""
        return [
            {
                "name": "hive_mind_search",
                "description": "Search the collective knowledge base using semantic similarity",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text"
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Optional namespace filter",
                            "default": None
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        },
                        "min_similarity": {
                            "type": "number",
                            "description": "Minimum similarity threshold",
                            "default": 0.3,
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "hive_mind_synthesize",
                "description": "Synthesize knowledge from multiple sources with conflict resolution",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "namespace_pattern": {
                            "type": "string",
                            "description": "SQL LIKE pattern for namespace filtering"
                        },
                        "key_pattern": {
                            "type": "string",
                            "description": "SQL LIKE pattern for key filtering"
                        },
                        "min_confidence": {
                            "type": "number",
                            "description": "Minimum confidence threshold",
                            "default": 0.3,
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "strategy": {
                            "type": "string",
                            "description": "Conflict resolution strategy",
                            "enum": ["highest_confidence", "most_recent", "consensus", "weighted_average", "expert_preference"],
                            "default": "highest_confidence"
                        }
                    }
                }
            },
            {
                "name": "hive_mind_aggregate_confidence",
                "description": "Aggregate confidence scores from multiple sources with uncertainty quantification",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sources": {
                            "type": "array",
                            "description": "List of confidence sources",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source_id": {"type": "string"},
                                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                    "weight": {"type": "number", "default": 1.0, "minimum": 0.0},
                                    "metadata": {"type": "object", "default": {}}
                                },
                                "required": ["source_id", "confidence"]
                            }
                        },
                        "method": {
                            "type": "string",
                            "description": "Confidence aggregation method",
                            "enum": ["weighted_average", "bayesian_update", "entropy_based", "consensus_based", "beta_distribution", "trust_propagation"],
                            "default": "weighted_average"
                        },
                        "include_temporal_decay": {
                            "type": "boolean",
                            "description": "Apply temporal decay to confidence scores",
                            "default": True
                        }
                    },
                    "required": ["sources"]
                }
            },
            {
                "name": "hive_mind_update_knowledge",
                "description": "Update knowledge base with new information and consistency validation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "description": "Update operation type",
                            "enum": ["create", "update", "delete", "merge", "validate"]
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Knowledge namespace"
                        },
                        "key": {
                            "type": "string",
                            "description": "Knowledge key"
                        },
                        "content": {
                            "type": "string",
                            "description": "Knowledge content (required for create/update/merge)"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata",
                            "default": {}
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score",
                            "default": 1.0,
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "priority": {
                            "type": "string",
                            "description": "Update priority",
                            "enum": ["low", "normal", "high", "critical"],
                            "default": "normal"
                        },
                        "source_id": {
                            "type": "string",
                            "description": "Source identifier for tracking"
                        }
                    },
                    "required": ["operation", "namespace", "key"]
                }
            },
            {
                "name": "hive_mind_query_advanced",
                "description": "Advanced query combining search, synthesis, and confidence analysis",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query text"
                        },
                        "include_synthesis": {
                            "type": "boolean",
                            "description": "Include knowledge synthesis in results",
                            "default": True
                        },
                        "include_confidence": {
                            "type": "boolean",
                            "description": "Include confidence analysis",
                            "default": True
                        },
                        "context_namespace": {
                            "type": "string",
                            "description": "Context namespace for focused search"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum search results",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "hive_mind_stats",
                "description": "Get comprehensive statistics about the hive mind system",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_search_stats": {
                            "type": "boolean",
                            "description": "Include semantic search statistics",
                            "default": True
                        },
                        "include_synthesis_stats": {
                            "type": "boolean",
                            "description": "Include synthesis engine statistics",
                            "default": True
                        },
                        "include_confidence_stats": {
                            "type": "boolean",
                            "description": "Include confidence system statistics",
                            "default": True
                        },
                        "include_update_stats": {
                            "type": "boolean",
                            "description": "Include knowledge updater statistics",
                            "default": True
                        }
                    }
                }
            }
        ]
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle MCP tool calls for hive mind operations.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            if tool_name == "hive_mind_search":
                return await self._handle_search(arguments)
            elif tool_name == "hive_mind_synthesize":
                return await self._handle_synthesize(arguments)
            elif tool_name == "hive_mind_aggregate_confidence":
                return await self._handle_aggregate_confidence(arguments)
            elif tool_name == "hive_mind_update_knowledge":
                return await self._handle_update_knowledge(arguments)
            elif tool_name == "hive_mind_query_advanced":
                return await self._handle_advanced_query(arguments)
            elif tool_name == "hive_mind_stats":
                return await self._handle_get_stats(arguments)
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": [tool["name"] for tool in self.tools]
                }
                
        except (ValueError, TypeError, KeyError) as e:
            logger.error("Tool call failed: %s", str(e))
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "tool_name": tool_name
            }
    
    async def _handle_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle semantic search requests."""
        query = arguments["query"]
        namespace = arguments.get("namespace")
        max_results = arguments.get("max_results", 10)
        min_similarity = arguments.get("min_similarity", 0.3)
        
        # Update search configuration
        config = SearchConfig(
            max_results=max_results,
            min_similarity_threshold=min_similarity
        )
        self.semantic_search.config = config
        
        # Perform search
        results = await self.semantic_search.search_knowledge(
            query=query,
            namespace=namespace,
            max_results=max_results
        )
        
        # Format results for MCP response
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.content,
                "namespace": result.namespace,
                "key": result.key,
                "confidence": result.confidence,
                "similarity_score": result.similarity_score,
                "metadata": result.metadata,
                "access_count": result.access_count
            })
        
        return {
            "success": True,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "query": query,
            "search_config": {
                "max_results": max_results,
                "min_similarity": min_similarity,
                "namespace_filter": namespace
            }
        }
    
    async def _handle_synthesize(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle knowledge synthesis requests."""
        namespace_pattern = arguments.get("namespace_pattern")
        key_pattern = arguments.get("key_pattern")
        min_confidence = arguments.get("min_confidence", 0.3)
        strategy = arguments.get("strategy", "highest_confidence")
        
        # Perform synthesis
        result = await self.synthesis_engine.synthesize_from_database(
            namespace_pattern=namespace_pattern,
            key_pattern=key_pattern,
            min_confidence=min_confidence,
            strategy=strategy
        )
        
        if result is None:
            return {
                "success": False,
                "error": "No sources found for synthesis",
                "search_criteria": {
                    "namespace_pattern": namespace_pattern,
                    "key_pattern": key_pattern,
                    "min_confidence": min_confidence
                }
            }
        
        return {
            "success": True,
            "synthesized_content": result.synthesized_content,
            "source_count": result.source_count,
            "confidence_score": result.confidence_score,
            "synthesis_method": result.synthesis_method,
            "conflict_resolution": result.conflict_resolution,
            "redundancy_eliminated": result.redundancy_eliminated,
            "metadata_combined": result.metadata_combined,
            "synthesis_timestamp": result.synthesis_timestamp
        }
    
    async def _handle_aggregate_confidence(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle confidence aggregation requests."""
        sources_data = arguments["sources"]
        method_name = arguments.get("method", "weighted_average")
        include_temporal_decay = arguments.get("include_temporal_decay", True)
        
        # Convert to ConfidenceSource objects
        sources = []
        for source_data in sources_data:
            source = ConfidenceSource(
                source_id=source_data["source_id"],
                confidence=source_data["confidence"],
                weight=source_data.get("weight", 1.0),
                metadata=source_data.get("metadata", {})
            )
            sources.append(source)
        
        # Get method enum
        try:
            method = ConfidenceMethod(method_name)
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid confidence method: {method_name}",
                "available_methods": [m.value for m in ConfidenceMethod]
            }
        
        # Perform aggregation
        result = await self.confidence_system.aggregate_confidence(
            sources=sources,
            method=method,
            include_temporal_decay=include_temporal_decay
        )
        
        return {
            "success": True,
            "final_confidence": result.final_confidence,
            "uncertainty": result.uncertainty,
            "method_used": result.method_used,
            "source_count": result.source_count,
            "weight_distribution": result.weight_distribution,
            "meta_confidence": result.meta_confidence,
            "aggregation_timestamp": result.aggregation_timestamp
        }
    
    async def _handle_update_knowledge(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle knowledge update requests."""
        operation = arguments["operation"]
        namespace = arguments["namespace"]
        key = arguments["key"]
        content = arguments.get("content", "")
        metadata = arguments.get("metadata", {})
        confidence = arguments.get("confidence", 1.0)
        priority_str = arguments.get("priority", "normal")
        source_id = arguments.get("source_id", "")
        
        # Convert operation and priority
        try:
            update_type = UpdateType(operation)
            priority = UpdatePriority[priority_str.upper()]
        except (ValueError, KeyError):
            return {
                "success": False,
                "error": f"Invalid operation or priority: {operation}, {priority_str}",
                "valid_operations": [t.value for t in UpdateType],
                "valid_priorities": [p.name.lower() for p in UpdatePriority]
            }
        
        # Create update
        update = KnowledgeUpdate(
            update_id="",  # Will be generated
            update_type=update_type,
            namespace=namespace,
            key=key,
            content=content,
            metadata=metadata,
            priority=priority,
            source_id=source_id,
            confidence=confidence
        )
        
        # Submit update
        update_id = await self.updater.submit_update(update)
        
        # Wait a moment and get status
        await asyncio.sleep(0.1)  # Brief wait for processing
        status = await self.updater.get_update_status(update_id)
        
        return {
            "success": True,
            "update_id": update_id,
            "status": status,
            "operation": operation,
            "namespace": namespace,
            "key": key
        }
    
    async def _handle_advanced_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle advanced queries combining multiple capabilities."""
        query = arguments["query"]
        include_synthesis = arguments.get("include_synthesis", True)
        include_confidence = arguments.get("include_confidence", True)
        context_namespace = arguments.get("context_namespace")
        max_results = arguments.get("max_results", 10)
        
        result = {
            "success": True,
            "query": query,
            "timestamp": time.time()
        }
        
        # Perform semantic search
        search_results = await self.semantic_search.search_knowledge(
            query=query,
            namespace=context_namespace,
            max_results=max_results
        )
        
        result["search_results"] = [
            {
                "content": r.content,
                "namespace": r.namespace,
                "key": r.key,
                "similarity_score": r.similarity_score,
                "confidence": r.confidence
            }
            for r in search_results
        ]
        
        # Perform synthesis if requested
        if include_synthesis and search_results:
            # Convert search results to knowledge sources
            sources = []
            for search_result in search_results:
                source = KnowledgeSource(
                    namespace=search_result.namespace,
                    key=search_result.key,
                    content=search_result.content,
                    confidence=search_result.confidence,
                    metadata=search_result.metadata
                )
                sources.append(source)
            
            synthesis_result = await self.synthesis_engine.synthesize_knowledge(sources)
            result["synthesis"] = {
                "synthesized_content": synthesis_result.synthesized_content,
                "confidence_score": synthesis_result.confidence_score,
                "source_count": synthesis_result.source_count,
                "synthesis_method": synthesis_result.synthesis_method
            }
        
        # Perform confidence analysis if requested
        if include_confidence and search_results:
            confidence_sources = []
            for search_result in search_results:
                conf_source = ConfidenceSource(
                    source_id=f"{search_result.namespace}:{search_result.key}",
                    confidence=search_result.confidence,
                    weight=search_result.similarity_score
                )
                confidence_sources.append(conf_source)
            
            confidence_result = await self.confidence_system.aggregate_confidence(
                sources=confidence_sources,
                method=ConfidenceMethod.WEIGHTED_AVERAGE
            )
            result["confidence_analysis"] = {
                "overall_confidence": confidence_result.final_confidence,
                "uncertainty": confidence_result.uncertainty,
                "meta_confidence": confidence_result.meta_confidence
            }
        
        return result
    
    async def _handle_get_stats(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle statistics requests."""
        include_search = arguments.get("include_search_stats", True)
        include_synthesis = arguments.get("include_synthesis_stats", True)
        include_confidence = arguments.get("include_confidence_stats", True)
        include_update = arguments.get("include_update_stats", True)
        
        stats = {
            "success": True,
            "timestamp": time.time()
        }
        
        if include_search:
            stats["search_stats"] = await self.semantic_search.get_search_stats()
        
        if include_synthesis:
            stats["synthesis_stats"] = await self.synthesis_engine.get_synthesis_stats()
        
        if include_confidence:
            stats["confidence_stats"] = await self.confidence_system.get_confidence_stats()
        
        if include_update:
            stats["update_stats"] = await self.updater.get_updater_stats()
        
        return stats
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get MCP tool definitions."""
        return self.tools
    
    async def initialize(self):
        """Initialize the hive mind query tool."""
        try:
            # Test database connections
            search_stats = await self.semantic_search.get_search_stats()
            logger.info("Hive mind query tool initialized. Knowledge entries: %d", 
                       search_stats.get("total_entries", 0))
            return True
        except (ValueError, TypeError) as e:
            logger.error("Failed to initialize hive mind query tool: %s", str(e))
            return False
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            self.semantic_search.clear_cache()
            logger.info("Hive mind query tool cleanup completed")
        except (ValueError, TypeError) as e:
            logger.error("Cleanup failed: %s", str(e))

# Factory function for easy instantiation
def create_hive_mind_query_tool(db_path: str) -> HiveMindQueryTool:
    """
    Factory function to create hive mind query tool.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Configured HiveMindQueryTool instance
    """
    return HiveMindQueryTool(db_path)