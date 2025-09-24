"""
Directory Structure Manager MCP Tool for MCP Swarm Intelligence Server

This tool manages project directory structure, particularly focusing on hidden
directories and proper organization for MCP server projects with agent-config integration.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from ..memory.manager import MemoryManager

logger = logging.getLogger(__name__)


def mcp_tool(name: str):
    """Decorator for MCP tools (placeholder implementation)."""
    def decorator(func):
        func.mcp_tool_name = name
        return func
    return decorator


@mcp_tool("directory_structure_manager")
async def directory_structure_manager_tool(
    action: str,
    target_directory: str = ".",
    structure_type: str = "full_mcp_project",
    hidden_prefix: bool = True,
    backup_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    MCP tool for managing project directory structure.
    
    Args:
        action: Action to perform (create/validate/update/backup/restore)
        target_directory: Target directory for structure creation
        structure_type: Type of structure (full_mcp_project/agent_config_only/minimal)
        hidden_prefix: Whether to use hidden folder prefixes (.agent-config)
        backup_config: Configuration for backup operations
        
    Returns:
        Directory structure management results
    """
    
    try:
        manager = DirectoryStructureManager(
            target_directory=Path(target_directory),
            hidden_prefix=hidden_prefix
        )
        
        if action == "create":
            result = await manager.create_directory_structure(structure_type)
            
        elif action == "validate":
            result = await manager.validate_directory_structure(structure_type)
            
        elif action == "update":
            result = await manager.update_directory_structure(structure_type)
            
        elif action == "backup":
            result = await manager.backup_directory_structure(backup_config or {})
            
        elif action == "restore":
            if not backup_config or "backup_path" not in backup_config:
                raise ValueError("backup_config with backup_path required for restore action")
            result = await manager.restore_directory_structure(backup_config["backup_path"])
            
        elif action == "list_templates":
            result = await manager.list_available_templates()
            
        else:
            raise ValueError(f"Unknown action: {action}")
            
        return {
            "success": True,
            "action": action,
            "structure_type": structure_type,
            "target_directory": str(manager.target_directory),
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except (OSError, PermissionError, ValueError) as e:
        return {
            "success": False,
            "error": str(e),
            "action": action,
            "structure_type": structure_type,
            "target_directory": target_directory,
            "timestamp": datetime.now().isoformat()
        }


class DirectoryStructureManager:
    """Manages directory structure for MCP server projects."""
    
    def __init__(
        self, 
        target_directory: Path = Path("."),
        hidden_prefix: bool = True
    ):
        self.target_directory = target_directory.resolve()
        self.hidden_prefix = hidden_prefix
        self.memory_manager = MemoryManager()
        self.logger = logging.getLogger(__name__)
        
        # Define directory templates
        self.templates = {
            "full_mcp_project": self._get_full_mcp_template(),
            "agent_config_only": self._get_agent_config_template(),
            "minimal": self._get_minimal_template()
        }
    
    def _get_full_mcp_template(self) -> Dict[str, Any]:
        """Get full MCP project directory structure template."""
        prefix = "." if self.hidden_prefix else ""
        
        return {
            "directories": [
                # Agent configuration
                f"{prefix}agent-config",
                f"{prefix}agent-config/specialists",
                
                # MCP server cache and data
                f"{prefix}mcp-cache",
                f"{prefix}mcp-cache/tools",
                f"{prefix}mcp-cache/resources",
                
                # Swarm intelligence data
                f"{prefix}swarm-data",
                f"{prefix}swarm-data/coordination",
                f"{prefix}swarm-data/algorithms", 
                
                # Hive mind memory
                f"{prefix}hive-memory",
                f"{prefix}hive-memory/knowledge",
                f"{prefix}hive-memory/patterns",
                
                # Configuration backups
                f"{prefix}config-backups",
                
                # Quality gates and validation
                f"{prefix}quality-gates",
                f"{prefix}quality-gates/validation-reports"
            ],
            "files": [
                f"{prefix}agent-config/.gitignore",
                f"{prefix}mcp-cache/.gitignore", 
                f"{prefix}swarm-data/.gitignore",
                f"{prefix}hive-memory/.gitignore",
                f"{prefix}config-backups/.gitignore"
            ],
            "gitignore_patterns": [
                "# MCP Swarm Intelligence Server Generated Files",
                f"{prefix}mcp-cache/",
                f"{prefix}swarm-data/runtime/",
                f"{prefix}hive-memory/sessions/",
                f"{prefix}config-backups/*.backup",
                "*.swarm-temp",
                "*.mcp-temp"
            ]
        }
    
    def _get_agent_config_template(self) -> Dict[str, Any]:
        """Get agent configuration only template."""
        prefix = "." if self.hidden_prefix else ""
        
        return {
            "directories": [
                f"{prefix}agent-config",
                f"{prefix}agent-config/specialists",
                f"{prefix}config-backups"
            ],
            "files": [
                f"{prefix}agent-config/.gitignore"
            ],
            "gitignore_patterns": [
                "# Agent Configuration",
                f"{prefix}config-backups/*.backup"
            ]
        }
    
    def _get_minimal_template(self) -> Dict[str, Any]:
        """Get minimal directory structure template."""
        prefix = "." if self.hidden_prefix else ""
        
        return {
            "directories": [
                f"{prefix}agent-config"
            ],
            "files": [],
            "gitignore_patterns": []
        }
    
    async def create_directory_structure(self, structure_type: str) -> Dict[str, Any]:
        """Create directory structure based on template."""
        if structure_type not in self.templates:
            raise ValueError(f"Unknown structure type: {structure_type}")
        
        template = self.templates[structure_type]
        created_dirs = []
        created_files = []
        
        try:
            # Create directories
            for dir_path in template["directories"]:
                full_path = self.target_directory / dir_path
                if not full_path.exists():
                    full_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(full_path))
            
            # Create gitignore files
            for file_path in template["files"]:
                full_path = self.target_directory / file_path
                if not full_path.exists():
                    content = self._get_gitignore_content(file_path, template["gitignore_patterns"])
                    full_path.write_text(content)
                    created_files.append(str(full_path))
            
            # Update main project gitignore
            await self._update_main_gitignore(template["gitignore_patterns"])
            
            # Log structure creation
            await self.memory_manager.log_request({
                "action": "directory_structure_created",
                "structure_type": structure_type,
                "created_dirs": created_dirs,
                "created_files": created_files,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "status": "created",
                "created_directories": created_dirs,
                "created_files": created_files,
                "template_used": structure_type
            }
            
        except (OSError, PermissionError, ValueError) as e:
            # Rollback on failure
            await self._rollback_creation(created_dirs, created_files)
            raise RuntimeError(f"Failed to create directory structure: {e}") from e
    
    async def validate_directory_structure(self, structure_type: str) -> Dict[str, Any]:
        """Validate existing directory structure against template."""
        if structure_type not in self.templates:
            raise ValueError(f"Unknown structure type: {structure_type}")
        
        template = self.templates[structure_type]
        missing_dirs = []
        missing_files = []
        existing_dirs = []
        existing_files = []
        
        # Check directories
        for dir_path in template["directories"]:
            full_path = self.target_directory / dir_path
            if full_path.exists() and full_path.is_dir():
                existing_dirs.append(str(full_path))
            else:
                missing_dirs.append(str(full_path))
        
        # Check files
        for file_path in template["files"]:
            full_path = self.target_directory / file_path
            if full_path.exists() and full_path.is_file():
                existing_files.append(str(full_path))
            else:
                missing_files.append(str(full_path))
        
        is_valid = len(missing_dirs) == 0 and len(missing_files) == 0
        
        return {
            "is_valid": is_valid,
            "structure_type": structure_type,
            "existing_directories": existing_dirs,
            "existing_files": existing_files,
            "missing_directories": missing_dirs,
            "missing_files": missing_files,
            "completeness": len(existing_dirs + existing_files) / len(template["directories"] + template["files"]) if (template["directories"] + template["files"]) else 1.0
        }
    
    async def update_directory_structure(self, structure_type: str) -> Dict[str, Any]:
        """Update directory structure to match template."""
        validation = await self.validate_directory_structure(structure_type)
        
        if validation["is_valid"]:
            return {
                "status": "up_to_date",
                "message": "Directory structure is already valid"
            }
        
        # Create missing components
        template = self.templates[structure_type]
        created_dirs = []
        created_files = []
        
        # Create missing directories
        for dir_path in validation["missing_directories"]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            created_dirs.append(dir_path)
        
        # Create missing files
        for file_path in validation["missing_files"]:
            file_path_obj = Path(file_path)
            content = self._get_gitignore_content(
                file_path_obj.name, 
                template["gitignore_patterns"]
            )
            file_path_obj.write_text(content, encoding='utf-8')
            created_files.append(file_path)
        
        return {
            "status": "updated",
            "created_directories": created_dirs,
            "created_files": created_files,
            "structure_type": structure_type
        }
    
    async def backup_directory_structure(self, backup_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create backup of current directory structure."""
        backup_name = backup_config.get("name", f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        include_content = backup_config.get("include_content", True)
        
        prefix = "." if self.hidden_prefix else ""
        backup_dir = self.target_directory / f"{prefix}config-backups" / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backed_up_items = []
        
        # Backup configuration directories
        config_dirs = [
            f"{prefix}agent-config",
            f"{prefix}mcp-cache", 
            f"{prefix}swarm-data",
            f"{prefix}hive-memory"
        ]
        
        for dir_name in config_dirs:
            source_dir = self.target_directory / dir_name
            if source_dir.exists():
                target_dir = backup_dir / dir_name
                if include_content:
                    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
                else:
                    target_dir.mkdir(parents=True, exist_ok=True)
                backed_up_items.append(str(source_dir))
        
        # Create backup metadata
        metadata = {
            "backup_name": backup_name,
            "timestamp": datetime.now().isoformat(),
            "target_directory": str(self.target_directory),
            "include_content": include_content,
            "backed_up_items": backed_up_items,
            "structure_template": "full_mcp_project"  # Default for backups
        }
        
        metadata_file = backup_dir / "backup_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
        
        return {
            "status": "backup_created",
            "backup_path": str(backup_dir),
            "backup_name": backup_name,
            "backed_up_items": backed_up_items,
            "metadata": metadata
        }
    
    async def restore_directory_structure(self, backup_path: str) -> Dict[str, Any]:
        """Restore directory structure from backup."""
        backup_dir = Path(backup_path)
        if not backup_dir.exists():
            raise ValueError(f"Backup directory does not exist: {backup_path}")
        
        metadata_file = backup_dir / "backup_metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"Backup metadata not found: {metadata_file}")
        
        metadata = json.loads(metadata_file.read_text())
        restored_items = []
        
        # Restore directories
        for item_name in backup_dir.iterdir():
            if item_name.is_dir() and item_name.name != "backup_metadata.json":
                source_dir = backup_dir / item_name.name
                target_dir = self.target_directory / item_name.name
                
                # Create backup of existing before restore
                if target_dir.exists():
                    temp_backup = target_dir.with_suffix('.pre_restore_backup')
                    if temp_backup.exists():
                        shutil.rmtree(temp_backup)
                    shutil.move(str(target_dir), str(temp_backup))
                
                shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
                restored_items.append(str(target_dir))
        
        return {
            "status": "restored",
            "backup_path": backup_path,
            "restored_items": restored_items,
            "backup_metadata": metadata
        }
    
    async def list_available_templates(self) -> Dict[str, Any]:
        """List all available directory structure templates."""
        templates_info = {}
        
        for template_name, template_data in self.templates.items():
            templates_info[template_name] = {
                "directories_count": len(template_data["directories"]),
                "files_count": len(template_data["files"]),
                "gitignore_patterns_count": len(template_data["gitignore_patterns"]),
                "directories": template_data["directories"],
                "description": self._get_template_description(template_name)
            }
        
        return {
            "available_templates": templates_info,
            "default_template": "full_mcp_project"
        }
    
    def _get_template_description(self, template_name: str) -> str:
        """Get description for a template."""
        descriptions = {
            "full_mcp_project": "Complete MCP server project with all swarm intelligence components",
            "agent_config_only": "Agent configuration directories only",
            "minimal": "Minimal directory structure for basic MCP functionality"
        }
        return descriptions.get(template_name, "Custom template")
    
    def _get_gitignore_content(self, file_path: str, patterns: List[str]) -> str:
        """Generate gitignore content for specific files."""
        if ".gitignore" not in file_path:
            return ""
        
        content = [
            "# Generated by MCP Swarm Intelligence Server",
            "# Directory Structure Manager",
            f"# Generated at: {datetime.now().isoformat()}",
            "",
            "# Temporary files",
            "*.tmp",
            "*.temp", 
            "*.log",
            "",
            "# Runtime data",
            "runtime/",
            "cache/",
            "sessions/",
            "",
            "# Backup files", 
            "*.backup",
            "*.bak",
            ""
        ]
        
        if patterns:
            content.extend(["# Specific patterns"] + patterns)
        
        return "\n".join(content)
    
    async def _update_main_gitignore(self, patterns: List[str]) -> None:
        """Update main project gitignore with MCP-specific patterns."""
        gitignore_path = self.target_directory / ".gitignore"
        
        if not patterns:
            return
        
        existing_content = ""
        if gitignore_path.exists():
            existing_content = gitignore_path.read_text()
        
        # Check if patterns already exist
        mcp_section_marker = "# MCP Swarm Intelligence Server Generated Files"
        if mcp_section_marker in existing_content:
            return  # Already updated
        
        # Add MCP-specific patterns
        new_content = existing_content.rstrip() + "\n\n" + "\n".join(patterns) + "\n"
        gitignore_path.write_text(new_content)
    
    async def _rollback_creation(self, created_dirs: List[str], created_files: List[str]) -> None:
        """Rollback directory/file creation on failure."""
        # Remove created files
        for file_path in created_files:
            try:
                Path(file_path).unlink()
            except (OSError, FileNotFoundError):
                pass  # Best effort cleanup
        
        # Remove created directories (in reverse order)
        for dir_path in reversed(created_dirs):
            try:
                Path(dir_path).rmdir()  # Only removes empty directories
            except (OSError, FileNotFoundError):
                pass  # Best effort cleanup