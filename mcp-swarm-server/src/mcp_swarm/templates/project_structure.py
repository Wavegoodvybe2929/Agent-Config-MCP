"""
Project Structure Templates for MCP Swarm Intelligence Server

This module provides templates and utilities for creating and managing
project directory structures for MCP server projects with agent-config integration.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ProjectStructure:
    """Represents a project directory structure."""
    
    def __init__(
        self,
        name: str,
        directories: List[str],
        files: List[str],
        gitignore_patterns: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.directories = directories
        self.files = files
        self.gitignore_patterns = gitignore_patterns
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert structure to dictionary representation."""
        return {
            "name": self.name,
            "directories": self.directories,
            "files": self.files,
            "gitignore_patterns": self.gitignore_patterns,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectStructure':
        """Create structure from dictionary representation."""
        return cls(
            name=data["name"],
            directories=data["directories"],
            files=data["files"],
            gitignore_patterns=data["gitignore_patterns"],
            metadata=data.get("metadata", {})
        )


class ProjectStructureTemplate:
    """Template for MCP server project directory structure."""
    
    def __init__(self, use_hidden_prefix: bool = True):
        self.use_hidden_prefix = use_hidden_prefix
        self.prefix = "." if use_hidden_prefix else ""
        self.logger = logging.getLogger(__name__)
        
        # Initialize predefined templates
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, ProjectStructure]:
        """Initialize all predefined project templates."""
        templates = {}
        
        # Full MCP project template
        templates["full_mcp_project"] = self._create_full_mcp_template()
        
        # Agent config only template
        templates["agent_config_only"] = self._create_agent_config_template()
        
        # Minimal template
        templates["minimal"] = self._create_minimal_template()
        
        # Development template (for development environments)
        templates["development"] = self._create_development_template()
        
        # Production template (for production deployments)
        templates["production"] = self._create_production_template()
        
        return templates
    
    def _create_full_mcp_template(self) -> ProjectStructure:
        """Create full MCP project structure template."""
        directories = [
            # Core agent configuration
            f"{self.prefix}agent-config",
            f"{self.prefix}agent-config/specialists",
            f"{self.prefix}agent-config/templates",
            
            # MCP server cache and temporary data
            f"{self.prefix}mcp-cache",
            f"{self.prefix}mcp-cache/tools",
            f"{self.prefix}mcp-cache/resources",
            f"{self.prefix}mcp-cache/sessions",
            
            # Swarm intelligence data and algorithms
            f"{self.prefix}swarm-data",
            f"{self.prefix}swarm-data/coordination",
            f"{self.prefix}swarm-data/algorithms",
            f"{self.prefix}swarm-data/consensus",
            f"{self.prefix}swarm-data/strategies",
            
            # Hive mind collective memory
            f"{self.prefix}hive-memory",
            f"{self.prefix}hive-memory/knowledge",
            f"{self.prefix}hive-memory/patterns",
            f"{self.prefix}hive-memory/sessions",
            f"{self.prefix}hive-memory/synthesis",
            
            # Configuration backups and versioning
            f"{self.prefix}config-backups",
            f"{self.prefix}config-backups/daily",
            f"{self.prefix}config-backups/manual",
            
            # Quality gates and validation
            f"{self.prefix}quality-gates",
            f"{self.prefix}quality-gates/validation-reports",
            f"{self.prefix}quality-gates/test-results",
            
            # Runtime data and logs
            f"{self.prefix}runtime-data",
            f"{self.prefix}runtime-data/logs",
            f"{self.prefix}runtime-data/metrics",
            
            # Development tools and utilities
            f"{self.prefix}dev-tools",
            f"{self.prefix}dev-tools/scripts",
            f"{self.prefix}dev-tools/templates"
        ]
        
        files = [
            f"{self.prefix}agent-config/.gitignore",
            f"{self.prefix}mcp-cache/.gitignore",
            f"{self.prefix}swarm-data/.gitignore",
            f"{self.prefix}hive-memory/.gitignore",
            f"{self.prefix}config-backups/.gitignore",
            f"{self.prefix}quality-gates/.gitignore",
            f"{self.prefix}runtime-data/.gitignore",
            f"{self.prefix}dev-tools/.gitignore"
        ]
        
        gitignore_patterns = [
            "# MCP Swarm Intelligence Server Generated Files",
            f"{self.prefix}mcp-cache/",
            f"{self.prefix}swarm-data/runtime/",
            f"{self.prefix}hive-memory/sessions/",
            f"{self.prefix}runtime-data/",
            f"{self.prefix}config-backups/*.backup",
            f"{self.prefix}quality-gates/temp/",
            "",
            "# Temporary and cache files",
            "*.swarm-temp",
            "*.mcp-temp",
            "*.coordination-cache",
            "*.memory-snapshot",
            "",
            "# Runtime logs and metrics",
            "*.runtime.log",
            "*.metrics.json",
            "*.performance.data"
        ]
        
        metadata = {
            "description": "Complete MCP server project with full swarm intelligence capabilities",
            "use_case": "Production-ready MCP server with all features enabled",
            "components": [
                "agent-config", "mcp-cache", "swarm-data", "hive-memory",
                "config-backups", "quality-gates", "runtime-data", "dev-tools"
            ]
        }
        
        return ProjectStructure(
            name="full_mcp_project",
            directories=directories,
            files=files,
            gitignore_patterns=gitignore_patterns,
            metadata=metadata
        )
    
    def _create_agent_config_template(self) -> ProjectStructure:
        """Create agent configuration only template."""
        directories = [
            f"{self.prefix}agent-config",
            f"{self.prefix}agent-config/specialists",
            f"{self.prefix}agent-config/templates",
            f"{self.prefix}config-backups",
            f"{self.prefix}config-backups/agent-configs"
        ]
        
        files = [
            f"{self.prefix}agent-config/.gitignore",
            f"{self.prefix}config-backups/.gitignore"
        ]
        
        gitignore_patterns = [
            "# Agent Configuration Backups",
            f"{self.prefix}config-backups/*.backup",
            f"{self.prefix}config-backups/temp/",
            "",
            "# Agent configuration cache",
            "*.agent-cache"
        ]
        
        metadata = {
            "description": "Agent configuration system only",
            "use_case": "Lightweight agent configuration management",
            "components": ["agent-config", "config-backups"]
        }
        
        return ProjectStructure(
            name="agent_config_only",
            directories=directories,
            files=files,
            gitignore_patterns=gitignore_patterns,
            metadata=metadata
        )
    
    def _create_minimal_template(self) -> ProjectStructure:
        """Create minimal directory structure template."""
        directories = [
            f"{self.prefix}agent-config"
        ]
        
        files = []
        
        gitignore_patterns = [
            "# Minimal MCP configuration",
            "*.temp",
            "*.cache"
        ]
        
        metadata = {
            "description": "Minimal directory structure for basic MCP functionality",
            "use_case": "Lightweight MCP server setup",
            "components": ["agent-config"]
        }
        
        return ProjectStructure(
            name="minimal",
            directories=directories,
            files=files,
            gitignore_patterns=gitignore_patterns,
            metadata=metadata
        )
    
    def _create_development_template(self) -> ProjectStructure:
        """Create development environment template."""
        directories = [
            f"{self.prefix}agent-config",
            f"{self.prefix}agent-config/specialists",
            f"{self.prefix}dev-data",
            f"{self.prefix}dev-data/test-configs",
            f"{self.prefix}dev-data/mock-data",
            f"{self.prefix}dev-cache",
            f"{self.prefix}dev-logs"
        ]
        
        files = [
            f"{self.prefix}agent-config/.gitignore",
            f"{self.prefix}dev-data/.gitignore",
            f"{self.prefix}dev-cache/.gitignore",
            f"{self.prefix}dev-logs/.gitignore"
        ]
        
        gitignore_patterns = [
            "# Development Environment",
            f"{self.prefix}dev-cache/",
            f"{self.prefix}dev-logs/",
            f"{self.prefix}dev-data/temp/",
            "",
            "# Development files",
            "*.dev.log",
            "*.debug.json",
            "*.test-data"
        ]
        
        metadata = {
            "description": "Development environment with testing and debugging support",
            "use_case": "Local development and testing",
            "components": ["agent-config", "dev-data", "dev-cache", "dev-logs"]
        }
        
        return ProjectStructure(
            name="development",
            directories=directories,
            files=files,
            gitignore_patterns=gitignore_patterns,
            metadata=metadata
        )
    
    def _create_production_template(self) -> ProjectStructure:
        """Create production deployment template."""
        directories = [
            f"{self.prefix}agent-config",
            f"{self.prefix}agent-config/specialists",
            f"{self.prefix}prod-cache",
            f"{self.prefix}prod-data",
            f"{self.prefix}prod-data/backups",
            f"{self.prefix}prod-logs",
            f"{self.prefix}monitoring"
        ]
        
        files = [
            f"{self.prefix}agent-config/.gitignore",
            f"{self.prefix}prod-cache/.gitignore",
            f"{self.prefix}prod-data/.gitignore",
            f"{self.prefix}prod-logs/.gitignore",
            f"{self.prefix}monitoring/.gitignore"
        ]
        
        gitignore_patterns = [
            "# Production Environment",
            f"{self.prefix}prod-cache/",
            f"{self.prefix}prod-logs/",
            f"{self.prefix}prod-data/temp/",
            f"{self.prefix}monitoring/temp/",
            "",
            "# Production files",
            "*.prod.log",
            "*.metrics.json",
            "*.backup.gz"
        ]
        
        metadata = {
            "description": "Production deployment with monitoring and backup support",
            "use_case": "Production MCP server deployment",
            "components": ["agent-config", "prod-cache", "prod-data", "prod-logs", "monitoring"]
        }
        
        return ProjectStructure(
            name="production",
            directories=directories,
            files=files,
            gitignore_patterns=gitignore_patterns,
            metadata=metadata
        )
    
    async def create_mcp_project_structure(
        self,
        base_path: Path,
        template_name: str = "full_mcp_project",
        custom_config: Optional[Dict[str, Any]] = None
    ) -> ProjectStructure:
        """Create complete MCP server project structure."""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.templates[template_name]
        
        # Apply custom configuration if provided
        if custom_config:
            template = self._apply_custom_config(template, custom_config)
        
        # Create the structure
        await self._create_structure_from_template(base_path, template)
        
        self.logger.info("Created MCP project structure using template: %s", template_name)
        return template
    
    def _apply_custom_config(
        self,
        template: ProjectStructure,
        custom_config: Dict[str, Any]
    ) -> ProjectStructure:
        """Apply custom configuration to template."""
        # Create a copy to avoid modifying the original
        modified_dirs = list(template.directories)
        modified_files = list(template.files)
        modified_patterns = list(template.gitignore_patterns)
        
        # Add custom directories
        if "additional_directories" in custom_config:
            for directory in custom_config["additional_directories"]:
                if directory not in modified_dirs:
                    modified_dirs.append(directory)
        
        # Add custom files
        if "additional_files" in custom_config:
            for file in custom_config["additional_files"]:
                if file not in modified_files:
                    modified_files.append(file)
        
        # Add custom gitignore patterns
        if "additional_gitignore_patterns" in custom_config:
            modified_patterns.extend(custom_config["additional_gitignore_patterns"])
        
        # Update metadata
        updated_metadata = dict(template.metadata)
        if "metadata" in custom_config:
            updated_metadata.update(custom_config["metadata"])
        
        return ProjectStructure(
            name=f"{template.name}_custom",
            directories=modified_dirs,
            files=modified_files,
            gitignore_patterns=modified_patterns,
            metadata=updated_metadata
        )
    
    async def _create_structure_from_template(
        self,
        base_path: Path,
        template: ProjectStructure
    ) -> None:
        """Create directory structure from template."""
        # Ensure base path exists
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Create directories
        for directory in template.directories:
            dir_path = base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug("Created directory: %s", dir_path)
        
        # Create files (primarily gitignore files)
        for file_path in template.files:
            full_path = base_path / file_path
            if not full_path.exists():
                content = self._generate_gitignore_content(
                    file_path, 
                    template.gitignore_patterns
                )
                full_path.write_text(content, encoding='utf-8')
                self.logger.debug("Created file: %s", full_path)
    
    def _generate_gitignore_content(
        self,
        file_path: str,
        patterns: List[str]
    ) -> str:
        """Generate gitignore content for specific directory."""
        if ".gitignore" not in file_path:
            return ""
        
        # Determine context from file path
        context_patterns = []
        if "agent-config" in file_path:
            context_patterns = [
                "# Agent Configuration",
                "*.agent-temp",
                "*.config-cache",
                "temp/"
            ]
        elif "mcp-cache" in file_path:
            context_patterns = [
                "# MCP Cache",
                "*.cache",
                "*.tmp",
                "sessions/",
                "temp/"
            ]
        elif "swarm-data" in file_path:
            context_patterns = [
                "# Swarm Intelligence Data",
                "*.swarm-temp",
                "runtime/",
                "temp/"
            ]
        elif "hive-memory" in file_path:
            context_patterns = [
                "# Hive Mind Memory",
                "*.memory-temp",
                "sessions/",
                "temp/"
            ]
        else:
            context_patterns = [
                "# Generated files",
                "*.temp",
                "*.tmp",
                "temp/"
            ]
        
        header = [
            "# Generated by MCP Swarm Intelligence Server",
            "# Project Structure Template",
            f"# Generated at: {datetime.now().isoformat()}",
            ""
        ]
        
        return "\n".join(header + context_patterns + [""] + patterns)
    
    def get_template(self, template_name: str) -> Optional[ProjectStructure]:
        """Get a specific template by name."""
        return self.templates.get(template_name)
    
    def list_templates(self) -> Dict[str, Dict[str, Any]]:
        """List all available templates with their metadata."""
        return {
            name: {
                "directories_count": len(template.directories),
                "files_count": len(template.files),
                "gitignore_patterns_count": len(template.gitignore_patterns),
                "metadata": template.metadata
            }
            for name, template in self.templates.items()
        }
    
    def save_template(self, template: ProjectStructure, file_path: Path) -> None:
        """Save template to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(template.to_dict(), f, indent=2)
        self.logger.info("Saved template to: %s", file_path)
    
    def load_template(self, file_path: Path) -> ProjectStructure:
        """Load template from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        template = ProjectStructure.from_dict(data)
        self.logger.info("Loaded template from: %s", file_path)
        return template
    
    def validate_template(self, template: ProjectStructure) -> Dict[str, Any]:
        """Validate template structure and configuration."""
        issues = []
        
        # Check for empty directories list
        if not template.directories:
            issues.append("Template has no directories defined")
        
        # Check for duplicate directories
        if len(template.directories) != len(set(template.directories)):
            issues.append("Template has duplicate directories")
        
        # Check for invalid directory names
        for directory in template.directories:
            if not directory or ".." in directory:
                issues.append(f"Invalid directory name: {directory}")
        
        # Check gitignore patterns
        if not template.gitignore_patterns:
            issues.append("Template has no gitignore patterns (consider adding some)")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "template_name": template.name,
            "validation_timestamp": datetime.now().isoformat()
        }