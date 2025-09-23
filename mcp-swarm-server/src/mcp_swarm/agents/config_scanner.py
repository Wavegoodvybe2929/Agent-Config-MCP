"""
Agent Configuration Scanner for MCP Swarm Intelligence Server

This module provides comprehensive agent discovery and configuration management,
scanning markdown files with YAML frontmatter to build a complete agent registry.
"""

import os
import yaml
import re
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Represents a complete agent configuration with all metadata"""
    agent_type: str
    name: str
    capabilities: List[str]
    intersections: List[str]
    domain: Optional[str] = None
    priority: int = 5
    memory_enabled: bool = False
    coordination_style: str = "standard"
    metadata: Optional[Dict[str, Any]] = None
    file_path: Optional[Path] = None
    last_modified: Optional[datetime] = None
    content_hash: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        if self.file_path:
            result['file_path'] = str(self.file_path)
        if self.last_modified:
            result['last_modified'] = self.last_modified.isoformat()
        return result


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    agent_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



class AgentConfigScanner:
    """
    Comprehensive agent configuration scanner and manager.
    
    Scans markdown files with YAML frontmatter to discover and manage
    agent configurations for the MCP Swarm Intelligence Server.
    """
    
    def __init__(self, config_directory: Path):
        self.config_directory = Path(config_directory)
        self.discovered_agents: Dict[str, AgentConfig] = {}
        self.scan_in_progress = False
        
        # Required fields for valid agent configuration
        self.required_fields = {
            'agent_type', 'domain', 'capabilities'
        }
        
        # Valid agent types
        self.valid_agent_types = {
            'specialist', 'coordinator', 'manager', 'validator'
        }
        
    async def scan_agent_configurations(self, force_rescan: bool = False) -> Dict[str, AgentConfig]:
        """
        Scan directory for agent configuration files.
        
        Args:
            force_rescan: Force complete rescan even if already scanned
            
        Returns:
            Dictionary of discovered agent configurations
        """
        if self.scan_in_progress:
            logger.warning("Scan already in progress, waiting...")
            while self.scan_in_progress:
                await asyncio.sleep(0.1)
            return self.discovered_agents
        
        try:
            self.scan_in_progress = True
            
            if not self.config_directory.exists():
                logger.error(f"Configuration directory not found: {self.config_directory}")
                return {}
            
            logger.info(f"Scanning agent configurations in: {self.config_directory}")
            
            # Clear previous results if force rescan
            if force_rescan:
                self.discovered_agents.clear()
            
            # Scan for markdown files
            discovered_configs = {}
            config_files = list(self.config_directory.rglob("*.md"))
            
            logger.info(f"Found {len(config_files)} markdown files to scan")
            
            for config_file in config_files:
                try:
                    config = await self.parse_markdown_config(config_file)
                    if config:
                        agent_name = self.extract_agent_name(config_file)
                        config.name = agent_name
                        config.file_path = config_file
                        config.last_modified = datetime.fromtimestamp(
                            config_file.stat().st_mtime
                        )
                        
                        discovered_configs[agent_name] = config
                        logger.debug("Successfully parsed agent config: %s", agent_name)
                    else:
                        logger.debug("Skipped non-agent file: %s", config_file)
                        
                except Exception as e:
                    logger.error("Error parsing %s: %s", config_file, e)
                    continue
            
            # Update discovered agents
            self.discovered_agents.update(discovered_configs)
            
            logger.info(
                "Agent discovery complete: %d agents found",
                len(self.discovered_agents)
            )
            
            return self.discovered_agents
            
        finally:
            self.scan_in_progress = False
    
    async def parse_markdown_config(self, file_path: Path) -> Optional[AgentConfig]:
        """
        Parse individual markdown configuration file.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            AgentConfig if valid, None otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract frontmatter
            frontmatter = self._extract_frontmatter(content)
            if not frontmatter:
                return None
            
            # Validate required fields
            if not self._has_required_fields(frontmatter):
                return None
            
            # Build agent config
            config = AgentConfig(
                agent_type=frontmatter.get('agent_type', 'specialist'),
                name="",  # Will be set from filename
                capabilities=frontmatter.get('capabilities', []),
                intersections=frontmatter.get('intersections', []),
                domain=frontmatter.get('domain'),
                priority=frontmatter.get('priority', 5),
                memory_enabled=frontmatter.get('memory_enabled', False),
                coordination_style=frontmatter.get('coordination_style', 'standard'),
                metadata=frontmatter
            )
            
            # Calculate content hash for change detection
            config.content_hash = str(hash(content))
            
            return config
            
        except Exception as e:
            logger.error(f"Error parsing markdown config {file_path}: {e}")
            return None
    
    def _extract_frontmatter(self, content: str) -> Dict[str, Any]:
        """
        Extract YAML frontmatter from markdown content.
        
        Args:
            content: Markdown file content
            
        Returns:
            Parsed frontmatter as dictionary
        """
        try:
            # Look for YAML frontmatter between --- markers
            frontmatter_pattern = r'^---\n(.*?)\n---'
            match = re.search(frontmatter_pattern, content, re.DOTALL | re.MULTILINE)
            
            if not match:
                return {}
            
            yaml_content = match.group(1)
            frontmatter = yaml.safe_load(yaml_content)
            
            return frontmatter if isinstance(frontmatter, dict) else {}
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            return {}
        except Exception as e:
            logger.error(f"Frontmatter extraction error: {e}")
            return {}
    
    def _has_required_fields(self, frontmatter: Dict[str, Any]) -> bool:
        """
        Check if frontmatter has required fields for agent configuration.
        
        Args:
            frontmatter: Parsed frontmatter dictionary
            
        Returns:
            True if has required fields, False otherwise
        """
        # Check for required fields
        missing_fields = self.required_fields - set(frontmatter.keys())
        if missing_fields:
            return False
        
        # Validate agent type
        agent_type = frontmatter.get('agent_type')
        if agent_type not in self.valid_agent_types:
            return False
        
        # Validate capabilities is a list
        capabilities = frontmatter.get('capabilities')
        if not isinstance(capabilities, list):
            return False
        
        return True
    
    def extract_agent_name(self, file_path: Path) -> str:
        """Extract agent name from file path"""
        # Remove .md extension and use filename as agent name
        name = file_path.stem
        
        # Handle specialist subdirectory
        if file_path.parent.name == 'specialists':
            return f"specialists.{name}"
        
        return name
    
    def validate_agent_configuration(self, config: AgentConfig) -> ValidationResult:
        """
        Validate agent configuration for completeness and consistency.
        
        Args:
            config: Agent configuration to validate
            
        Returns:
            Validation result with errors and warnings
        """
        errors = []
        warnings = []
        
        # Validate required fields
        if not config.agent_type:
            errors.append("Missing agent_type")
        elif config.agent_type not in self.valid_agent_types:
            errors.append(f"Invalid agent_type: {config.agent_type}")
        
        if not config.domain:
            errors.append("Missing domain")
        
        if not config.capabilities:
            errors.append("Missing capabilities")
        elif not isinstance(config.capabilities, list):
            errors.append("Capabilities must be a list")
        
        # Validate intersections reference existing agents
        if config.intersections:
            unknown_intersections = []
            for intersection in config.intersections:
                if intersection not in self.discovered_agents:
                    # Check if it's a partial match (e.g., 'code' for 'specialists.code')
                    matching_agents = [
                        name for name in self.discovered_agents.keys()
                        if name.endswith(f".{intersection}") or name == intersection
                    ]
                    if not matching_agents:
                        unknown_intersections.append(intersection)
            
            if unknown_intersections:
                warnings.append(
                    f"Unknown intersections: {', '.join(unknown_intersections)}"
                )
        
        # Validate priority range
        if not 1 <= config.priority <= 10:
            warnings.append(f"Priority {config.priority} outside recommended range 1-10")
        
        # Validate coordination style
        valid_styles = {'standard', 'swarm', 'hive', 'queen'}
        if config.coordination_style not in valid_styles:
            warnings.append(
                f"Unknown coordination_style: {config.coordination_style}"
            )
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            agent_name=config.name
        )
    
    async def handle_file_change(self, file_path: Path):
        """Handle individual file change event"""
        try:
            logger.info("Configuration file changed: %s", file_path)
            
            if file_path.exists():
                # Re-parse the changed file
                config = await self.parse_markdown_config(file_path)
                if config:
                    agent_name = self.extract_agent_name(file_path)
                    config.name = agent_name
                    config.file_path = file_path
                    config.last_modified = datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    )
                    
                    old_config = self.discovered_agents.get(agent_name)
                    
                    # Check if content actually changed
                    if old_config and old_config.content_hash == config.content_hash:
                        logger.debug(f"No content change detected for {agent_name}")
                        return
                    
                    self.discovered_agents[agent_name] = config
                    logger.info(f"Updated agent configuration: {agent_name}")
                    
                    # Validate the updated configuration
                    validation = self.validate_agent_configuration(config)
                    if not validation.is_valid:
                        logger.warning(
                            f"Updated config for {agent_name} has validation errors: "
                            f"{', '.join(validation.errors)}"
                        )
            
        except Exception as e:
            logger.error(f"Error handling file change {file_path}: {e}")
    
    async def _handle_file_deletion(self, file_path: Path):
        """Handle file deletion event"""
        try:
            agent_name = self.extract_agent_name(file_path)
            if agent_name in self.discovered_agents:
                del self.discovered_agents[agent_name]
                logger.info(f"Removed deleted agent configuration: {agent_name}")
                
        except Exception as e:
            logger.error(f"Error handling file deletion {file_path}: {e}")
    
    def get_agents_by_domain(self, domain: str) -> List[AgentConfig]:
        """Get all agents in a specific domain"""
        return [
            config for config in self.discovered_agents.values()
            if config.domain == domain
        ]
    
    def get_agents_by_capability(self, capability: str) -> List[AgentConfig]:
        """Get all agents with a specific capability"""
        return [
            config for config in self.discovered_agents.values()
            if capability in config.capabilities
        ]
    
    def get_agent_intersections(self, agent_name: str) -> List[AgentConfig]:
        """Get all agents that intersect with the specified agent"""
        agent_config = self.discovered_agents.get(agent_name)
        if not agent_config:
            return []
        
        intersection_configs = []
        for intersection_name in agent_config.intersections:
            # Handle partial matches
            matching_agents = [
                config for name, config in self.discovered_agents.items()
                if name.endswith(f".{intersection_name}") or name == intersection_name
            ]
            intersection_configs.extend(matching_agents)
        
        return intersection_configs
    
    async def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive discovery statistics"""
        total_agents = len(self.discovered_agents)
        
        # Group by agent type
        type_counts = {}
        domain_counts = {}
        capability_counts = {}
        
        for config in self.discovered_agents.values():
            # Count by type
            type_counts[config.agent_type] = type_counts.get(config.agent_type, 0) + 1
            
            # Count by domain
            if config.domain:
                domain_counts[config.domain] = domain_counts.get(config.domain, 0) + 1
            
            # Count capabilities
            for capability in config.capabilities:
                capability_counts[capability] = capability_counts.get(capability, 0) + 1
        
        # Calculate validation statistics
        validation_results = [
            self.validate_agent_configuration(config)
            for config in self.discovered_agents.values()
        ]
        
        valid_count = sum(1 for result in validation_results if result.is_valid)
        total_errors = sum(len(result.errors) for result in validation_results)
        total_warnings = sum(len(result.warnings) for result in validation_results)
        
        return {
            "total_agents": total_agents,
            "valid_agents": valid_count,
            "invalid_agents": total_agents - valid_count,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "agent_types": type_counts,
            "domains": domain_counts,
            "capabilities": capability_counts,
            "scan_directory": str(self.config_directory),
            "last_scan": datetime.utcnow().isoformat()
        }


async def main():
    """Example usage of AgentConfigScanner"""
    
    # Initialize scanner
    scanner = AgentConfigScanner(Path("../../agent-config"))
    
    # Scan for configurations
    agents = await scanner.scan_agent_configurations()
    
    print(f"Discovered {len(agents)} agents:")
    for name, config in agents.items():
        validation = scanner.validate_agent_configuration(config)
        status = "✅" if validation.is_valid else "❌"
        print(f"  {status} {name} ({config.domain})")
    
    # Get statistics
    stats = await scanner.get_discovery_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())