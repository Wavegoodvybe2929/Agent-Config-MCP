"""
Agent Configuration Operations

This module provides comprehensive operations for managing agent configuration files
including directory creation, file management, YAML frontmatter parsing, and 
orchestrator routing pattern enforcement. Integrates with existing agent discovery system.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .config_scanner import AgentConfig, AgentConfigScanner, ValidationResult
from .config_reloader import AgentRegistry, DynamicConfigurationReloader

logger = logging.getLogger(__name__)


class AgentConfigOperations:
    """Operations for agent configuration file management"""
    
    def __init__(self, config_dir: Path = Path(".agent-config")):
        self.config_dir = Path(config_dir)
        self.specialists_dir = self.config_dir / "specialists"
        
        # Initialize existing components for integration
        self.scanner = AgentConfigScanner(self.config_dir)
        self.registry = AgentRegistry()
        self.reloader = DynamicConfigurationReloader(self.registry, self.config_dir)
        
        # Template for orchestrator routing directive
        self.orchestrator_routing_directive = """⚠️ MANDATORY ORCHESTRATOR ROUTING: Before executing any work from this specialist config, 
ALWAYS consult agent-config/orchestrator.md FIRST for:
- Task routing and agent selection validation
- Workflow coordination and quality gate requirements  
- Multi-agent coordination needs and handoff procedures
- Current project context and priority alignment
- Agent hooks integration and lifecycle management

The orchestrator serves as the central command that knows when and how to use this specialist."""

    async def create_agent_config_directory(self) -> bool:
        """Create .agent-config directory structure"""
        try:
            # Create main config directory
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Create specialists subdirectory
            self.specialists_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Created agent config directory structure at {self.config_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create agent config directory: {e}")
            return False

    async def create_agent_config_file(
        self, 
        agent_name: str, 
        config_data: Dict[str, Any]
    ) -> Path:
        """Create new agent configuration file"""
        try:
            # Determine file path - specialists go in specialists/ subdirectory
            if config_data.get('agent_type') == 'specialist':
                file_path = self.specialists_dir / f"{agent_name}.md"
            else:
                file_path = self.config_dir / f"{agent_name}.md"
                
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract frontmatter and content
            frontmatter = self._extract_frontmatter(config_data)
            content = config_data.get('content', '')
            
            # Add orchestrator routing directive if not present
            if self.orchestrator_routing_directive not in content:
                # Find where to insert the routing directive
                lines = content.split('\n')
                insert_index = 1  # After title
                
                # Look for role overview or similar section
                for i, line in enumerate(lines):
                    if any(keyword in line.lower() for keyword in ['role overview', '## role', 'expertise', 'responsibilities']):
                        insert_index = i
                        break
                
                lines.insert(insert_index, f"\n{self.orchestrator_routing_directive}\n")
                content = '\n'.join(lines)
            
            # Write file with frontmatter
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("---\n")
                yaml.dump(frontmatter, f, default_flow_style=False)
                f.write("---\n\n")
                f.write(content)
            
            # Trigger agent discovery update
            await self._trigger_discovery_update()
            
            logger.info("Created agent config file: %s", file_path)
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to create agent config file for {agent_name}: {e}")
            raise

    async def update_agent_config_file(
        self, 
        agent_name: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """Update existing agent configuration file"""
        try:
            # Find existing file
            file_path = await self._find_agent_config_file(agent_name)
            if not file_path:
                logger.error(f"Agent config file not found for {agent_name}")
                return False
            
            # Read existing file
            existing_data = await self.read_agent_config_file(agent_name)
            if not existing_data:
                return False
            
            # Merge updates
            frontmatter = existing_data.get('frontmatter', {})
            content = existing_data.get('content', '')
            
            # Update frontmatter if provided
            if 'frontmatter' in updates:
                frontmatter.update(updates['frontmatter'])
            
            # Update content if provided
            if 'content' in updates:
                content = updates['content']
            elif 'content_append' in updates:
                content += f"\n\n{updates['content_append']}"
            
            # Ensure orchestrator routing directive is present
            if self.orchestrator_routing_directive not in content:
                lines = content.split('\n')
                insert_index = 1
                
                for i, line in enumerate(lines):
                    if any(keyword in line.lower() for keyword in ['role overview', '## role', 'expertise', 'responsibilities']):
                        insert_index = i
                        break
                
                lines.insert(insert_index, f"\n{self.orchestrator_routing_directive}\n")
                content = '\n'.join(lines)
            
            # Write updated file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("---\n")
                yaml.dump(frontmatter, f, default_flow_style=False)
                f.write("---\n\n")
                f.write(content)
            
            # Trigger agent discovery update
            await self._trigger_discovery_update()
            
            logger.info("Updated agent config file: %s", file_path)
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent config file for {agent_name}: {e}")
            return False

    async def read_agent_config_file(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Read agent configuration file"""
        try:
            file_path = await self._find_agent_config_file(agent_name)
            if not file_path or not file_path.exists():
                logger.warning(f"Agent config file not found for {agent_name}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse frontmatter and content
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter_text = parts[1]
                    content_text = parts[2].strip()
                    
                    try:
                        frontmatter = yaml.safe_load(frontmatter_text)
                    except yaml.YAMLError as e:
                        logger.error(f"Failed to parse YAML frontmatter: {e}")
                        frontmatter = {}
                else:
                    frontmatter = {}
                    content_text = content
            else:
                frontmatter = {}
                content_text = content
            
            return {
                'agent_name': agent_name,
                'file_path': str(file_path),
                'frontmatter': frontmatter,
                'content': content_text,
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime)
            }
            
        except Exception as e:
            logger.error(f"Failed to read agent config file for {agent_name}: {e}")
            return None

    async def list_agent_config_files(self) -> List[Dict[str, Any]]:
        """List all agent configuration files"""
        try:
            configs = []
            
            # List files in main config directory
            if self.config_dir.exists():
                for file_path in self.config_dir.glob("*.md"):
                    if file_path.name != "README.md":  # Skip README
                        agent_name = file_path.stem
                        config_data = await self.read_agent_config_file(agent_name)
                        if config_data:
                            configs.append(config_data)
            
            # List files in specialists directory
            if self.specialists_dir.exists():
                for file_path in self.specialists_dir.glob("*.md"):
                    agent_name = file_path.stem
                    config_data = await self.read_agent_config_file(agent_name)
                    if config_data:
                        configs.append(config_data)
            
            return configs
            
        except Exception as e:
            logger.error(f"Failed to list agent config files: {e}")
            return []

    async def validate_agent_config_file(self, agent_name: str) -> Dict[str, Any]:
        """Validate agent configuration file"""
        try:
            config_data = await self.read_agent_config_file(agent_name)
            if not config_data:
                return {
                    'valid': False,
                    'errors': [f'Config file not found for agent: {agent_name}']
                }
            
            errors = []
            warnings = []
            
            frontmatter = config_data.get('frontmatter', {})
            content = config_data.get('content', '')
            
            # Validate frontmatter structure
            required_frontmatter_fields = ['agent_type', 'domain']
            for field in required_frontmatter_fields:
                if field not in frontmatter:
                    errors.append(f'Missing required frontmatter field: {field}')
            
            # Validate orchestrator routing directive
            if self.orchestrator_routing_directive not in content:
                warnings.append('Missing mandatory orchestrator routing directive')
            
            # Validate agent type and placement
            agent_type = frontmatter.get('agent_type')
            file_path = Path(config_data['file_path'])
            
            if agent_type == 'specialist' and 'specialists' not in str(file_path):
                warnings.append('Specialist agent should be in specialists/ directory')
            elif agent_type != 'specialist' and 'specialists' in str(file_path):
                warnings.append('Non-specialist agent should not be in specialists/ directory')
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'agent_name': agent_name,
                'file_path': config_data['file_path']
            }
            
        except Exception as e:
            logger.error(f"Failed to validate agent config file for {agent_name}: {e}")
            return {
                'valid': False,
                'errors': [f'Validation error: {str(e)}']
            }

    async def _find_agent_config_file(self, agent_name: str) -> Optional[Path]:
        """Find agent configuration file by name"""
        # Check specialists directory first
        specialist_path = self.specialists_dir / f"{agent_name}.md"
        if specialist_path.exists():
            return specialist_path
        
        # Check main config directory
        main_path = self.config_dir / f"{agent_name}.md"
        if main_path.exists():
            return main_path
        
        return None

    def _extract_frontmatter(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract frontmatter from config data"""
        frontmatter = {}
        
        # Standard frontmatter fields
        standard_fields = [
            'agent_type', 'domain', 'capabilities', 'intersections',
            'memory_enabled', 'coordination_style', 'priority', 'status'
        ]
        
        for field in standard_fields:
            if field in config_data:
                frontmatter[field] = config_data[field]
        
        # Include any additional frontmatter
        if 'frontmatter' in config_data:
            frontmatter.update(config_data['frontmatter'])
        
        return frontmatter

    async def _trigger_discovery_update(self):
        """Trigger agent discovery system update after config changes"""
        try:
            # Rescan configurations to update the discovery system
            await self.scanner.scan_agent_configurations(force_rescan=True)
            logger.info("Triggered agent discovery update after config change")
        except Exception as e:
            logger.warning("Failed to trigger discovery update: %s", e)

    async def refresh_agent_discovery(self) -> Dict[str, Any]:
        """Refresh the agent discovery system and return current state"""
        try:
            configs = await self.scanner.scan_agent_configurations(force_rescan=True)
            return {
                "success": True,
                "agent_count": len(configs),
                "agents_discovered": list(configs.keys()),
                "message": f"Successfully refreshed agent discovery. Found {len(configs)} agents."
            }
        except Exception as e:
            logger.error("Failed to refresh agent discovery: %s", e)
            return {
                "success": False,
                "error": f"Failed to refresh agent discovery: {str(e)}"
            }