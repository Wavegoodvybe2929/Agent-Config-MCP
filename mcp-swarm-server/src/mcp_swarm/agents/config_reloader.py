"""
Dynamic Configuration Reloader for MCP Swarm Intelligence Server

This module provides dynamic configuration reloading capabilities,
allowing agent configurations to be updated without system restart.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .config_scanner import AgentConfig, AgentConfigScanner, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ConfigurationChange:
    """Represents a configuration change event"""
    agent_name: str
    change_type: str  # created, modified, deleted
    old_config: Optional[AgentConfig] = None
    new_config: Optional[AgentConfig] = None
    file_path: Optional[Path] = None
    timestamp: Optional[datetime] = None
    validation_result: Optional[ValidationResult] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.file_path:
            result['file_path'] = str(self.file_path)
        if self.timestamp:
            result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class ReloadResult:
    """Result of a configuration reload operation"""
    success: bool
    agent_name: str
    changes_applied: List[str]
    warnings: List[str]
    errors: List[str]
    validation_result: Optional[ValidationResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.validation_result:
            result['validation_result'] = self.validation_result.to_dict()
        return result


class AgentRegistry:
    """Simple agent registry for managing active agent configurations"""
    
    def __init__(self):
        self.agents: Dict[str, AgentConfig] = {}
        self.change_listeners: List[Callable[[ConfigurationChange], None]] = []
    
    def register_agent(self, agent_name: str, config: AgentConfig):
        """Register or update an agent configuration"""
        old_config = self.agents.get(agent_name)
        self.agents[agent_name] = config
        
        change_type = "modified" if old_config else "created"
        change = ConfigurationChange(
            agent_name=agent_name,
            change_type=change_type,
            old_config=old_config,
            new_config=config,
            file_path=config.file_path
        )
        
        self._notify_listeners(change)
    
    def unregister_agent(self, agent_name: str) -> bool:
        """Unregister an agent configuration"""
        if agent_name in self.agents:
            old_config = self.agents.pop(agent_name)
            
            change = ConfigurationChange(
                agent_name=agent_name,
                change_type="deleted",
                old_config=old_config,
                new_config=None
            )
            
            self._notify_listeners(change)
            return True
        
        return False
    
    def get_agent(self, agent_name: str) -> Optional[AgentConfig]:
        """Get agent configuration by name"""
        return self.agents.get(agent_name)
    
    def list_agents(self) -> Dict[str, AgentConfig]:
        """Get all registered agent configurations"""
        return self.agents.copy()
    
    def add_change_listener(self, listener: Callable[[ConfigurationChange], None]):
        """Add a listener for configuration changes"""
        self.change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[ConfigurationChange], None]):
        """Remove a configuration change listener"""
        if listener in self.change_listeners:
            self.change_listeners.remove(listener)
    
    def _notify_listeners(self, change: ConfigurationChange):
        """Notify all listeners of a configuration change"""
        for listener in self.change_listeners:
            try:
                listener(change)
            except Exception as e:
                logger.error("Error in change listener: %s", e)


class DynamicConfigurationReloader:
    """
    Manages dynamic reloading of agent configurations.
    
    Provides hot-reloading capabilities for agent configurations
    without requiring system restart.
    """
    
    def __init__(self, agent_registry: AgentRegistry, config_directory: Path):
        self.agent_registry = agent_registry
        self.config_directory = Path(config_directory)
        self.config_scanner = AgentConfigScanner(config_directory)
        self.reload_queue = asyncio.Queue()
        self.reload_in_progress = False
        self.reload_history: List[ConfigurationChange] = []
        
        # Configuration validation rules
        self.validation_rules = {
            'require_validation': True,
            'allow_capability_changes': True,
            'allow_domain_changes': False,
            'allow_type_changes': False,
            'max_priority_change': 3
        }
    
    async def handle_configuration_change(
        self, 
        file_path: Path, 
        change_type: str
    ) -> ReloadResult:
        """
        Handle configuration file changes.
        
        Args:
            file_path: Path to changed configuration file
            change_type: Type of change (created, modified, deleted)
            
        Returns:
            Result of the reload operation
        """
        agent_name = self.config_scanner.extract_agent_name(file_path)
        
        logger.info("Handling configuration change: %s (%s)", agent_name, change_type)
        
        try:
            if change_type == "deleted":
                return await self._handle_deletion(agent_name, file_path)
            elif change_type in ["created", "modified"]:
                return await self._handle_creation_or_modification(agent_name, file_path)
            else:
                return ReloadResult(
                    success=False,
                    agent_name=agent_name,
                    changes_applied=[],
                    warnings=[],
                    errors=[f"Unknown change type: {change_type}"]
                )
        
        except Exception as e:
            logger.error("Error handling configuration change for %s: %s", agent_name, e)
            return ReloadResult(
                success=False,
                agent_name=agent_name,
                changes_applied=[],
                warnings=[],
                errors=[f"Exception during reload: {str(e)}"]
            )
    
    async def _handle_deletion(self, agent_name: str, _file_path: Path) -> ReloadResult:
        """Handle agent configuration deletion"""
        success = self.agent_registry.unregister_agent(agent_name)
        
        if success:
            logger.info("Successfully removed agent configuration: %s", agent_name)
            return ReloadResult(
                success=True,
                agent_name=agent_name,
                changes_applied=["agent_removed"],
                warnings=[],
                errors=[]
            )
        else:
            return ReloadResult(
                success=False,
                agent_name=agent_name,
                changes_applied=[],
                warnings=[],
                errors=["Agent not found in registry"]
            )
    
    async def _handle_creation_or_modification(
        self, 
        agent_name: str, 
        file_path: Path
    ) -> ReloadResult:
        """Handle agent configuration creation or modification"""
        # Parse the new configuration
        new_config = await self.config_scanner.parse_markdown_config(file_path)
        
        if not new_config:
            return ReloadResult(
                success=False,
                agent_name=agent_name,
                changes_applied=[],
                warnings=[],
                errors=["Failed to parse configuration file"]
            )
        
        # Set agent name and metadata
        new_config.name = agent_name
        new_config.file_path = file_path
        new_config.last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
        
        # Validate the new configuration
        validation_result = self.config_scanner.validate_agent_configuration(new_config)
        
        if not validation_result.is_valid and self.validation_rules['require_validation']:
            return ReloadResult(
                success=False,
                agent_name=agent_name,
                changes_applied=[],
                warnings=validation_result.warnings,
                errors=validation_result.errors,
                validation_result=validation_result
            )
        
        # Check for compatibility with existing configuration
        old_config = self.agent_registry.get_agent(agent_name)
        compatibility_result = self._validate_configuration_change(old_config, new_config)
        
        if not compatibility_result.success:
            return compatibility_result
        
        # Apply the configuration change
        self.agent_registry.register_agent(agent_name, new_config)
        
        # Record the change in history
        change = ConfigurationChange(
            agent_name=agent_name,
            change_type="modified" if old_config else "created",
            old_config=old_config,
            new_config=new_config,
            file_path=file_path,
            validation_result=validation_result
        )
        self.reload_history.append(change)
        
        # Keep history limited
        if len(self.reload_history) > 100:
            self.reload_history = self.reload_history[-50:]
        
        changes_applied = self._detect_configuration_changes(old_config, new_config)
        
        logger.info("Successfully reloaded agent configuration: %s", agent_name)
        
        return ReloadResult(
            success=True,
            agent_name=agent_name,
            changes_applied=changes_applied,
            warnings=validation_result.warnings,
            errors=[],
            validation_result=validation_result
        )
    
    def _validate_configuration_change(
        self, 
        old_config: Optional[AgentConfig],
        new_config: AgentConfig
    ) -> ReloadResult:
        """Validate configuration changes for compatibility"""
        errors = []
        warnings = []
        
        if old_config:
            # Check for disallowed changes
            if (not self.validation_rules['allow_domain_changes'] and 
                old_config.domain != new_config.domain):
                errors.append(f"Domain changes not allowed: {old_config.domain} -> {new_config.domain}")
            
            if (not self.validation_rules['allow_type_changes'] and 
                old_config.agent_type != new_config.agent_type):
                errors.append(f"Agent type changes not allowed: {old_config.agent_type} -> {new_config.agent_type}")
            
            # Check priority change limits
            priority_change = abs(old_config.priority - new_config.priority)
            max_change = self.validation_rules['max_priority_change']
            if priority_change > max_change:
                errors.append(f"Priority change too large: {priority_change} > {max_change}")
            
            # Warn about capability changes
            if (not self.validation_rules['allow_capability_changes'] and 
                set(old_config.capabilities) != set(new_config.capabilities)):
                warnings.append("Capability changes detected - may affect agent coordination")
        
        return ReloadResult(
            success=len(errors) == 0,
            agent_name=new_config.name,
            changes_applied=[],
            warnings=warnings,
            errors=errors
        )
    
    def _detect_configuration_changes(
        self, 
        old_config: Optional[AgentConfig],
        new_config: AgentConfig
    ) -> List[str]:
        """Detect specific changes between configurations"""
        changes = []
        
        if not old_config:
            changes.append("agent_created")
            return changes
        
        # Check for specific field changes
        if old_config.domain != new_config.domain:
            changes.append(f"domain_changed: {old_config.domain} -> {new_config.domain}")
        
        if old_config.priority != new_config.priority:
            changes.append(f"priority_changed: {old_config.priority} -> {new_config.priority}")
        
        if old_config.memory_enabled != new_config.memory_enabled:
            changes.append(f"memory_enabled_changed: {old_config.memory_enabled} -> {new_config.memory_enabled}")
        
        if old_config.coordination_style != new_config.coordination_style:
            changes.append(f"coordination_style_changed: {old_config.coordination_style} -> {new_config.coordination_style}")
        
        # Check capability changes
        old_caps = set(old_config.capabilities)
        new_caps = set(new_config.capabilities)
        
        added_caps = new_caps - old_caps
        removed_caps = old_caps - new_caps
        
        if added_caps:
            changes.append(f"capabilities_added: {', '.join(added_caps)}")
        
        if removed_caps:
            changes.append(f"capabilities_removed: {', '.join(removed_caps)}")
        
        # Check intersection changes
        old_intersections = set(old_config.intersections)
        new_intersections = set(new_config.intersections)
        
        added_intersections = new_intersections - old_intersections
        removed_intersections = old_intersections - new_intersections
        
        if added_intersections:
            changes.append(f"intersections_added: {', '.join(added_intersections)}")
        
        if removed_intersections:
            changes.append(f"intersections_removed: {', '.join(removed_intersections)}")
        
        return changes
    
    async def reload_agent_configuration(self, agent_name: str) -> ReloadResult:
        """
        Reload specific agent configuration from file.
        
        Args:
            agent_name: Name of agent to reload
            
        Returns:
            Result of the reload operation
        """
        # Find the configuration file
        current_config = self.agent_registry.get_agent(agent_name)
        
        if current_config and current_config.file_path:
            file_path = current_config.file_path
        else:
            # Try to find the file by name
            potential_paths = [
                self.config_directory / f"{agent_name}.md",
                self.config_directory / "specialists" / f"{agent_name.split('.')[-1]}.md"
            ]
            
            file_path = None
            for path in potential_paths:
                if path.exists():
                    file_path = path
                    break
            
            if not file_path:
                return ReloadResult(
                    success=False,
                    agent_name=agent_name,
                    changes_applied=[],
                    warnings=[],
                    errors=["Configuration file not found"]
                )
        
        return await self.handle_configuration_change(file_path, "modified")
    
    async def reload_all_configurations(self, force: bool = False) -> Dict[str, ReloadResult]:
        """
        Reload all agent configurations.
        
        Args:
            force: Force reload even if files haven't changed
            
        Returns:
            Dictionary of reload results by agent name
        """
        if self.reload_in_progress and not force:
            logger.warning("Reload already in progress")
            return {}
        
        try:
            self.reload_in_progress = True
            logger.info("Starting full configuration reload")
            
            # Scan for all configurations
            discovered_agents = await self.config_scanner.scan_agent_configurations(force_rescan=True)
            
            results = {}
            
            # Process each discovered agent
            for agent_name, config in discovered_agents.items():
                if config.file_path:
                    result = await self.handle_configuration_change(
                        config.file_path, 
                        "modified"
                    )
                    results[agent_name] = result
            
            # Remove agents that no longer exist
            current_agents = set(self.agent_registry.list_agents().keys())
            discovered_names = set(discovered_agents.keys())
            removed_agents = current_agents - discovered_names
            
            for agent_name in removed_agents:
                result = await self._handle_deletion(agent_name, Path("unknown"))
                results[agent_name] = result
            
            logger.info("Completed full configuration reload: %d agents processed", len(results))
            return results
            
        finally:
            self.reload_in_progress = False
    
    def get_reload_history(self, limit: int = 50) -> List[ConfigurationChange]:
        """Get recent configuration change history"""
        return self.reload_history[-limit:]
    
    def get_reload_statistics(self) -> Dict[str, Any]:
        """Get reload statistics"""
        total_changes = len(self.reload_history)
        
        change_types = {}
        for change in self.reload_history:
            change_types[change.change_type] = change_types.get(change.change_type, 0) + 1
        
        recent_changes = self.reload_history[-10:] if self.reload_history else []
        
        return {
            "total_changes": total_changes,
            "change_types": change_types,
            "recent_changes": [change.to_dict() for change in recent_changes],
            "reload_in_progress": self.reload_in_progress,
            "validation_rules": self.validation_rules
        }
    
    def update_validation_rules(self, rules: Dict[str, Any]):
        """Update validation rules for configuration changes"""
        self.validation_rules.update(rules)
        logger.info("Updated validation rules: %s", self.validation_rules)


async def main():
    """Example usage of DynamicConfigurationReloader"""
    
    # Initialize components
    registry = AgentRegistry()
    reloader = DynamicConfigurationReloader(registry, Path("../../agent-config"))
    
    # Add a change listener
    def log_change(change: ConfigurationChange):
        print(f"Configuration change: {change.agent_name} ({change.change_type})")
    
    registry.add_change_listener(log_change)
    
    # Perform initial load
    results = await reloader.reload_all_configurations()
    
    print(f"Loaded {len(results)} agent configurations")
    for agent_name, result in results.items():
        status = "✅" if result.success else "❌"
        print(f"  {status} {agent_name}: {len(result.changes_applied)} changes")
    
    # Get statistics
    stats = reloader.get_reload_statistics()
    print(f"\nReload Statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())