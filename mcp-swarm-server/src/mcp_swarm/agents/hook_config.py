"""
Agent Hook Configuration Manager for MCP Swarm Intelligence Server.

This module provides configuration loading, parsing, and validation capabilities
for agent hooks defined in agent-config/agent-hooks.md.
"""

from typing import Dict, List, Any, Optional, Union
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
import logging
from datetime import datetime
import asyncio
import aiofiles

from .hook_engine import HookType, HookDefinition

logger = logging.getLogger(__name__)


@dataclass
class HookConfiguration:
    """Configuration for a specific hook."""
    name: str
    hook_type: HookType
    description: str
    priority: int = 10
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of hook configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    hook_count: int = 0
    
    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)


class HookConfigurationManager:
    """
    Manager for loading and validating agent hook configurations from markdown files.
    
    This manager parses the agent-hooks.md file to extract hook definitions,
    validates their completeness and consistency, and provides configuration
    reloading capabilities.
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize hook configuration manager.
        
        Args:
            config_path: Path to the agent-hooks.md configuration file
        """
        self.config_path = Path(config_path)
        self.hook_configurations: Dict[str, HookConfiguration] = {}
        self._last_modified: Optional[datetime] = None
        self._hook_type_mappings = {
            "PRE_TASK_SETUP": HookType.PRE_TASK_SETUP,
            "PRE_TASK_VALIDATION": HookType.PRE_TASK_VALIDATION,
            "TASK_EXECUTION": HookType.TASK_EXECUTION,
            "POST_TASK_VALIDATION": HookType.POST_TASK_VALIDATION,
            "POST_TASK_CLEANUP": HookType.POST_TASK_CLEANUP,
            "INTER_AGENT_COORDINATION": HookType.INTER_AGENT_COORDINATION,
            "AGENT_HANDOFF_PREPARE": HookType.AGENT_HANDOFF_PREPARE,
            "AGENT_HANDOFF_EXECUTE": HookType.AGENT_HANDOFF_EXECUTE,
            "COLLABORATION_INIT": HookType.COLLABORATION_INIT,
            "COLLABORATION_SYNC": HookType.COLLABORATION_SYNC,
            "MEMORY_PERSISTENCE": HookType.MEMORY_PERSISTENCE,
            "CONTINUOUS_INTEGRATION": HookType.CONTINUOUS_INTEGRATION,
            "ERROR_HANDLING": HookType.ERROR_HANDLING,
            "CLEANUP": HookType.CLEANUP,
        }

    async def load_hook_configurations(self) -> Dict[str, HookConfiguration]:
        """
        Load hook configurations from agent-hooks.md file.
        
        Returns:
            Dictionary of hook configurations keyed by hook name
        """
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Hook configuration file not found: {self.config_path}")
            
            # Read file content
            async with aiofiles.open(self.config_path, 'r', encoding='utf-8') as file:
                content = await file.read()
            
            # Update last modified time
            self._last_modified = datetime.fromtimestamp(self.config_path.stat().st_mtime)
            
            # Parse hook configurations
            hook_configs = self._parse_hook_config(content)
            
            # Validate configurations
            validation_result = await self.validate_hook_configurations(hook_configs)
            
            if not validation_result.is_valid:
                logger.error("Hook configuration validation failed: %s", validation_result.errors)
                # Still return configurations but log errors
            
            if validation_result.warnings:
                logger.warning("Hook configuration warnings: %s", validation_result.warnings)
            
            self.hook_configurations = hook_configs
            logger.info("Loaded %d hook configurations", len(hook_configs))
            
            return hook_configs
            
        except Exception as e:
            logger.error("Failed to load hook configurations: %s", e)
            return {}

    def _parse_hook_config(self, content: str) -> Dict[str, HookConfiguration]:
        """
        Parse hook configuration from markdown content.
        
        Args:
            content: Markdown content from agent-hooks.md
            
        Returns:
            Dictionary of parsed hook configurations
        """
        hook_configs = {}
        
        # Parse YAML frontmatter if present
        frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
        global_config = {}
        if frontmatter_match:
            try:
                global_config = yaml.safe_load(frontmatter_match.group(1)) or {}
            except yaml.YAMLError as e:
                logger.warning("Failed to parse YAML frontmatter: %s", e)
        
        # Extract hook definitions using markdown sections
        hook_pattern = r'##\s+([A-Z_]+)\s*\n(.*?)(?=##\s+[A-Z_]+|\Z)'
        hook_matches = re.findall(hook_pattern, content, re.DOTALL | re.MULTILINE)
        
        for hook_name, hook_content in hook_matches:
            hook_name = hook_name.strip()
            
            # Skip if not a valid hook type
            if hook_name not in self._hook_type_mappings:
                continue
            
            try:
                hook_config = self._parse_single_hook(hook_name, hook_content, global_config)
                if hook_config:
                    hook_configs[hook_name] = hook_config
            except Exception as e:
                logger.error("Failed to parse hook %s: %s", hook_name, e)
        
        return hook_configs

    def _parse_single_hook(
        self, 
        hook_name: str, 
        hook_content: str, 
        global_config: Dict[str, Any]
    ) -> Optional[HookConfiguration]:
        """
        Parse a single hook configuration from markdown content.
        
        Args:
            hook_name: Name of the hook
            hook_content: Markdown content for this hook
            global_config: Global configuration from frontmatter
            
        Returns:
            Parsed hook configuration or None if parsing fails
        """
        try:
            hook_type = self._hook_type_mappings[hook_name]
            
            # Extract description from first paragraph
            description_match = re.search(r'-\s*\*\*Trigger\*\*:(.*?)(?=\n-|\Z)', hook_content, re.DOTALL)
            description = description_match.group(1).strip() if description_match else ""
            
            # Extract actions
            actions_match = re.search(r'-\s*\*\*Actions\*\*:\s*\n(.*?)(?=\n\n|\Z)', hook_content, re.DOTALL)
            actions = []
            if actions_match:
                action_lines = actions_match.group(1).strip().split('\n')
                actions = [line.strip().lstrip('- ') for line in action_lines if line.strip().startswith('-')]
            
            # Extract configuration values from content or use defaults
            priority = self._extract_config_value(hook_content, 'priority', 10)
            timeout = self._extract_config_value(hook_content, 'timeout', 30.0)
            retry_count = self._extract_config_value(hook_content, 'retry_count', 3)
            retry_delay = self._extract_config_value(hook_content, 'retry_delay', 1.0)
            enabled = self._extract_config_value(hook_content, 'enabled', True)
            
            # Ensure proper types
            priority = int(priority) if isinstance(priority, (int, float, str)) else 10
            timeout = float(timeout) if isinstance(timeout, (int, float, str)) else 30.0
            retry_count = int(retry_count) if isinstance(retry_count, (int, float, str)) else 3
            retry_delay = float(retry_delay) if isinstance(retry_delay, (int, float, str)) else 1.0
            enabled = bool(enabled) if isinstance(enabled, (bool, int, str)) else True
            
            # Extract dependencies from actions
            dependencies = self._extract_dependencies(actions)
            
            # Apply global configuration overrides
            if hook_name in global_config:
                hook_global = global_config[hook_name]
                priority = int(hook_global.get('priority', priority))
                timeout = float(hook_global.get('timeout', timeout))
                retry_count = int(hook_global.get('retry_count', retry_count))
                retry_delay = float(hook_global.get('retry_delay', retry_delay))
                enabled = bool(hook_global.get('enabled', enabled))
                if 'dependencies' in hook_global:
                    dependencies.extend(hook_global['dependencies'])
            
            return HookConfiguration(
                name=hook_name,
                hook_type=hook_type,
                description=description,
                priority=priority,
                timeout=timeout,
                retry_count=retry_count,
                retry_delay=retry_delay,
                dependencies=list(set(dependencies)),  # Remove duplicates
                enabled=enabled,
                metadata={
                    'actions': actions,
                    'source': 'agent-hooks.md',
                    'parsed_at': datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error("Failed to parse hook %s: %s", hook_name, e)
            return None

    def _extract_config_value(
        self, 
        content: str, 
        key: str, 
        default: Union[int, float, bool, str]
    ) -> Union[int, float, bool, str]:
        """
        Extract configuration value from hook content.
        
        Args:
            content: Hook content to search
            key: Configuration key to find
            default: Default value if not found
            
        Returns:
            Extracted value or default
        """
        # Look for config pattern like "priority: 5" or "timeout: 30.0"
        pattern = rf'{key}:\s*([^\n]+)'
        match = re.search(pattern, content, re.IGNORECASE)
        
        if not match:
            return default
        
        value_str = match.group(1).strip()
        
        try:
            if isinstance(default, bool):
                return value_str.lower() in ('true', 'yes', '1', 'on')
            elif isinstance(default, int):
                return int(value_str)
            elif isinstance(default, float):
                return float(value_str)
            else:
                return value_str
        except (ValueError, TypeError):
            logger.warning("Failed to parse %s value '%s', using default %s", key, value_str, default)
            return default

    def _extract_dependencies(self, actions: List[str]) -> List[str]:
        """
        Extract hook dependencies from action descriptions.
        
        Args:
            actions: List of action descriptions
            
        Returns:
            List of dependency hook names
        """
        dependencies = []
        dependency_keywords = [
            'after', 'requires', 'depends on', 'following', 'wait for',
            'triggered by', 'subsequent to', 'post'
        ]
        
        for action in actions:
            action_lower = action.lower()
            for keyword in dependency_keywords:
                if keyword in action_lower:
                    # Try to extract hook names mentioned near dependency keywords
                    for hook_name in self._hook_type_mappings.keys():
                        if hook_name.lower() in action_lower and hook_name not in dependencies:
                            dependencies.append(hook_name)
        
        return dependencies

    async def validate_hook_configurations(
        self, 
        configs: Dict[str, HookConfiguration]
    ) -> ValidationResult:
        """
        Validate hook configuration completeness and consistency.
        
        Args:
            configs: Dictionary of hook configurations to validate
            
        Returns:
            Validation result with errors and warnings
        """
        result = ValidationResult(is_valid=True, hook_count=len(configs))
        
        # Check for required hooks
        required_hooks = [
            'PRE_TASK_SETUP', 'POST_TASK_VALIDATION', 'POST_TASK_CLEANUP'
        ]
        for required_hook in required_hooks:
            if required_hook not in configs:
                result.add_warning(f"Recommended hook {required_hook} not configured")
        
        # Validate individual hook configurations
        for hook_name, config in configs.items():
            self._validate_single_hook(hook_name, config, configs, result)
        
        # Check for circular dependencies
        self._check_circular_dependencies(configs, result)
        
        # Check for orphaned dependencies
        self._check_orphaned_dependencies(configs, result)
        
        return result

    def _validate_single_hook(
        self, 
        hook_name: str, 
        config: HookConfiguration, 
        all_configs: Dict[str, HookConfiguration],
        result: ValidationResult
    ):
        """
        Validate a single hook configuration.
        
        Args:
            hook_name: Name of the hook being validated
            config: Hook configuration to validate
            all_configs: All hook configurations for dependency checking
            result: Validation result to update
        """
        # Validate basic properties
        if not config.name:
            result.add_error(f"Hook {hook_name} has empty name")
        
        if not config.description:
            result.add_warning(f"Hook {hook_name} has no description")
        
        if config.priority < 0:
            result.add_error(f"Hook {hook_name} has negative priority: {config.priority}")
        
        if config.timeout <= 0:
            result.add_error(f"Hook {hook_name} has invalid timeout: {config.timeout}")
        
        if config.retry_count < 0:
            result.add_error(f"Hook {hook_name} has negative retry count: {config.retry_count}")
        
        if config.retry_delay < 0:
            result.add_error(f"Hook {hook_name} has negative retry delay: {config.retry_delay}")
        
        # Validate dependencies exist
        for dep in config.dependencies:
            if dep not in all_configs:
                result.add_error(f"Hook {hook_name} depends on non-existent hook: {dep}")

    def _check_circular_dependencies(
        self, 
        configs: Dict[str, HookConfiguration], 
        result: ValidationResult
    ):
        """
        Check for circular dependencies in hook configurations.
        
        Args:
            configs: All hook configurations
            result: Validation result to update
        """
        def has_cycle(hook_name: str, visited: set, rec_stack: set) -> bool:
            visited.add(hook_name)
            rec_stack.add(hook_name)
            
            if hook_name in configs:
                for dep in configs[hook_name].dependencies:
                    if dep not in visited:
                        if has_cycle(dep, visited, rec_stack):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.remove(hook_name)
            return False
        
        visited = set()
        for hook_name in configs:
            if hook_name not in visited:
                if has_cycle(hook_name, visited, set()):
                    result.add_error(f"Circular dependency detected involving hook: {hook_name}")

    def _check_orphaned_dependencies(
        self, 
        configs: Dict[str, HookConfiguration], 
        result: ValidationResult
    ):
        """
        Check for dependencies that reference non-existent hooks.
        
        Args:
            configs: All hook configurations
            result: Validation result to update
        """
        all_hook_names = set(configs.keys())
        
        for hook_name, config in configs.items():
            for dep in config.dependencies:
                if dep not in all_hook_names:
                    result.add_error(f"Hook {hook_name} has orphaned dependency: {dep}")

    async def reload_configurations(self) -> bool:
        """
        Reload hook configurations from file if modified.
        
        Returns:
            True if configurations were reloaded, False otherwise
        """
        try:
            if not self.config_path.exists():
                logger.error("Configuration file does not exist: %s", self.config_path)
                return False
            
            current_modified = datetime.fromtimestamp(self.config_path.stat().st_mtime)
            
            if self._last_modified and current_modified <= self._last_modified:
                logger.debug("Configuration file not modified, skipping reload")
                return False
            
            logger.info("Configuration file modified, reloading...")
            configs = await self.load_hook_configurations()
            
            if configs:
                logger.info("Successfully reloaded %d hook configurations", len(configs))
                return True
            else:
                logger.error("Failed to reload configurations")
                return False
                
        except Exception as e:
            logger.error("Error during configuration reload: %s", e)
            return False

    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get summary of current hook configurations.
        
        Returns:
            Dictionary containing configuration summary
        """
        if not self.hook_configurations:
            return {
                "total_hooks": 0,
                "hook_types": [],
                "enabled_hooks": 0,
                "disabled_hooks": 0,
                "total_dependencies": 0,
                "last_loaded": None
            }
        
        hook_types = list(set(config.hook_type.value for config in self.hook_configurations.values()))
        enabled_count = sum(1 for config in self.hook_configurations.values() if config.enabled)
        disabled_count = len(self.hook_configurations) - enabled_count
        total_deps = sum(len(config.dependencies) for config in self.hook_configurations.values())
        
        return {
            "total_hooks": len(self.hook_configurations),
            "hook_types": sorted(hook_types),
            "enabled_hooks": enabled_count,
            "disabled_hooks": disabled_count,
            "total_dependencies": total_deps,
            "last_loaded": self._last_modified.isoformat() if self._last_modified else None,
            "configuration_file": str(self.config_path)
        }

    def get_hook_configuration(self, hook_name: str) -> Optional[HookConfiguration]:
        """
        Get configuration for a specific hook.
        
        Args:
            hook_name: Name of the hook
            
        Returns:
            Hook configuration or None if not found
        """
        return self.hook_configurations.get(hook_name)

    def create_hook_definition(self, config: HookConfiguration) -> HookDefinition:
        """
        Create a HookDefinition from a HookConfiguration.
        
        Args:
            config: Hook configuration to convert
            
        Returns:
            Hook definition ready for registration with hook engine
        """
        return HookDefinition(
            name=config.name,
            hook_type=config.hook_type,
            priority=config.priority,
            dependencies=config.dependencies.copy(),
            timeout=config.timeout,
            retry_count=config.retry_count,
            retry_delay=config.retry_delay,
            handler=None,  # Handler must be set separately
            description=config.description,
            enabled=config.enabled
        )