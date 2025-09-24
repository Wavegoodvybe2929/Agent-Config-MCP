"""
MCP Swarm Intelligence Server Documentation Generation Module

This module provides infrastructure for generating comprehensive documentation,
particularly copilot instructions that integrate MCP server capabilities with
agent-config workflow management.
"""

from typing import Dict, Any, List, Optional
import json
import yaml
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemplateContext:
    """Context data for template generation"""
    mcp_server_config: Dict[str, Any]
    agent_configs: Dict[str, Dict[str, Any]]
    project_phase: str
    project_context: Dict[str, Any]
    workflow_patterns: Dict[str, Any]


class TemplateEngine:
    """Template engine for generating documentation from configurations"""
    
    def __init__(self):
        self.template_dir = Path(__file__).parent / "templates"
        self.ensure_template_directory()
    
    def ensure_template_directory(self):
        """Ensure template directory exists with default templates"""
        self.template_dir.mkdir(exist_ok=True)
        
        # Create default copilot instruction template if it doesn't exist
        copilot_template = self.template_dir / "copilot_instructions.md.template"
        if not copilot_template.exists():
            self._create_default_copilot_template(copilot_template)
    
    def _create_default_copilot_template(self, template_path: Path):
        """Create default copilot instructions template"""
        template_content = '''# {project_name} Copilot Instructions

## Project Overview

{project_name} is a {project_description} featuring {key_features}.

## Agent Configuration System - Orchestrator-Driven Multi-Agent Workflow

This project uses the EXACT SAME agent configuration system as proven in BitNet-Rust, adapted for MCP development. **THE ORCHESTRATOR IS THE CENTRAL COMMAND** that routes all work and manages all specialist coordination with enhanced swarm intelligence and persistent memory capabilities.

### 🎯 MANDATORY ORCHESTRATOR-FIRST WORKFLOW

**ALWAYS START WITH THE ORCHESTRATOR** - This is non-negotiable for any development work:

#### **Step 1: ORCHESTRATOR CONSULTATION (REQUIRED)**
Before doing ANY work, **ALWAYS read `agent-config/orchestrator.md` FIRST** to:
- **Understand current project context** and MCP development priorities
- **Get proper task routing** to appropriate MCP specialist agents
- **Identify multi-agent coordination needs** for complex MCP features
- **Access workflow management** and quality gate requirements
- **Integrate with agent hooks system** for automated lifecycle management

#### **Step 2: ORCHESTRATOR ROUTING DECISION**
The orchestrator will route you to appropriate specialists using this framework:
- **Primary Agent Selection**: Based on task domain and complexity
- **Secondary Agent Coordination**: For cross-domain or complex requirements
- **Quality Gate Assignment**: Validation and review requirements
- **Workflow Coordination**: Timeline and dependency management

#### **Step 3: SPECIALIST CONSULTATION (ORCHESTRATOR-GUIDED)**
After orchestrator routing, consult the specific specialist agents identified:
- **Read specialist agent configs** for domain-specific context and expertise
- **Understand agent intersections** and collaboration patterns
- **Follow established workflows** and handoff procedures
- **Maintain orchestrator coordination** throughout the work

## MCP Server Integration

### Available MCP Tools

{mcp_tools_list}

### MCP Server Configuration

{mcp_server_config}

### Agent-Config Integration

{agent_config_integration}

## Agent Configuration Hierarchy & Orchestrator Authority

{agent_hierarchy}

## Multi-Agent Coordination Patterns (Orchestrator-Managed)

{coordination_patterns}

## Current Priority ({current_phase})

{current_priority}

## Workflow Rules - Orchestrator-Driven

{workflow_rules}

## When to Stop - Orchestrator-Defined Criteria

{stop_criteria}
'''
        template_path.write_text(template_content)
    
    async def render_template(self, template_name: str, context: TemplateContext) -> str:
        """Render a template with the provided context"""
        template_path = self.template_dir / f"{template_name}.template"
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template {template_name} not found at {template_path}")
        
        template_content = template_path.read_text()
        
        # Basic template variable substitution
        rendered = template_content.format(
            project_name=context.project_context.get('name', 'MCP Swarm Intelligence Server'),
            project_description=context.project_context.get('description', 'high-performance implementation of collective intelligence for multi-agent coordination'),
            key_features=', '.join(context.project_context.get('features', ['agent ecosystem management', 'hive mind knowledge bases', 'persistent memory systems'])),
            mcp_tools_list=self._generate_mcp_tools_list(context.mcp_server_config),
            mcp_server_config=self._format_mcp_config(context.mcp_server_config),
            agent_config_integration=self._generate_agent_integration_docs(context.agent_configs),
            agent_hierarchy=self._generate_agent_hierarchy(context.agent_configs),
            coordination_patterns=self._generate_coordination_patterns(context.workflow_patterns),
            current_phase=context.project_phase,
            current_priority=self._generate_current_priority(context.project_context, context.project_phase),
            workflow_rules=self._generate_workflow_rules(),
            stop_criteria=self._generate_stop_criteria()
        )
        
        return rendered
    
    def _generate_mcp_tools_list(self, mcp_config: Dict[str, Any]) -> str:
        """Generate formatted list of available MCP tools"""
        tools = mcp_config.get('tools', [])
        if not tools:
            return "- *Tools will be listed here as they are discovered*"
        
        tool_list = []
        for tool in tools:
            tool_list.append(f"- **{tool.get('name', 'Unknown')}**: {tool.get('description', 'No description')}")
        
        return '\n'.join(tool_list)
    
    def _format_mcp_config(self, mcp_config: Dict[str, Any]) -> str:
        """Format MCP server configuration for documentation"""
        config_str = "```json\n"
        config_str += json.dumps(mcp_config, indent=2)
        config_str += "\n```"
        return config_str
    
    def _generate_agent_integration_docs(self, agent_configs: Dict[str, Dict[str, Any]]) -> str:
        """Generate agent-config integration documentation"""
        agent_count = len(agent_configs)
        return f"""The MCP server integrates seamlessly with the agent-config system:

- **Automatic Tool Discovery**: MCP tools are automatically discovered from {agent_count} agent configurations
- **Orchestrator Coordination**: All MCP tool execution routes through orchestrator workflow management
- **Quality Gates**: Agent-defined quality standards apply to all MCP tool operations
- **Multi-Agent Workflows**: Complex MCP operations coordinate multiple specialist agents"""
    
    def _generate_agent_hierarchy(self, agent_configs: Dict[str, Dict[str, Any]]) -> str:
        """Generate agent hierarchy documentation"""
        hierarchy = """#### 🎯 **Central Command (ALWAYS START HERE)**
- **`orchestrator.md`** - **MANDATORY FIRST STOP** - Central coordination, agent routing, workflow management, project context

#### Core Technical Specialists (Orchestrator-Routed)
"""
        
        specialists = [name for name in agent_configs.keys() if name != 'orchestrator' and not name.endswith('_config')]
        for specialist in sorted(specialists):
            config = agent_configs.get(specialist, {})
            description = config.get('description', 'Specialist agent')
            intersections = config.get('intersections', [])
            hierarchy += f"- **`{specialist}.md`** - {description}"
            if intersections:
                hierarchy += f" (intersects with: {', '.join(intersections)})"
            hierarchy += "\n"
        
        return hierarchy
    
    def _generate_coordination_patterns(self, workflow_patterns: Dict[str, Any]) -> str:
        """Generate coordination patterns documentation"""
        single_pattern = workflow_patterns.get('single_agent', {}).get('pattern', 'Primary specialist + orchestrator coordination')
        multi_pattern = workflow_patterns.get('multi_agent', {}).get('pattern', 'Primary + Secondary specialists + orchestrator management')
        emergency_pattern = workflow_patterns.get('emergency', {}).get('pattern', 'Immediate escalation + orchestrator resource coordination')
        
        return f"""The orchestrator manages several coordination patterns for different task types:

#### **Single-Agent Tasks (Orchestrator Oversight)**
```
Simple tasks → {single_pattern}
Quality validation → truth_validator.md review
Documentation → documentation_writer.md if user-facing
```

#### **Multi-Agent Collaboration (Orchestrator Coordination)**
```
Complex features → {multi_pattern}
Cross-domain tasks → Multiple specialists + daily sync + orchestrator coordination
Critical changes → Full review chain + architect + security + orchestrator validation
```

#### **Emergency Response (Orchestrator Escalation)**
```
Critical issues → {emergency_pattern}
```"""
    
    def _generate_current_priority(self, project_context: Dict[str, Any], phase: str) -> str:
        """Generate current priority documentation"""
        return f"""**🎯 {phase}**: {project_context.get('current_focus', 'MCP Server Development')}
- **Orchestrator Routing**: As defined in orchestrator.md workflow matrix
- **Goal**: {project_context.get('current_goal', 'Complete MCP server implementation')}
- **Key Tasks**: {project_context.get('key_tasks', 'As defined in comprehensive TODO')}
- **Timeline**: {project_context.get('timeline', 'Current phase active')}"""
    
    def _generate_workflow_rules(self) -> str:
        """Generate workflow rules documentation"""
        return """1. **🎯 ALWAYS START WITH ORCHESTRATOR** - Read `orchestrator.md` first for every task
2. **Follow orchestrator routing** - Use the orchestrator's agent selection matrix
3. **Maintain orchestrator coordination** - Keep orchestrator informed of progress and handoffs
4. **Respect agent intersections** - Follow established collaboration patterns between agents
5. **Use quality gates** - Apply orchestrator-defined validation requirements
6. **Follow current phase** - Align with COMPREHENSIVE_TODO.md priorities as managed by orchestrator
7. **Execute user requests exactly** - Within the orchestrator's workflow framework
8. **Stop when complete** - When orchestrator-defined success criteria are met
9. **Be direct and clear** - Provide straightforward responses following orchestrator guidance
10. **Use available tools** - Leverage tools efficiently within orchestrator's workflow framework"""
    
    def _generate_stop_criteria(self) -> str:
        """Generate stop criteria documentation"""
        return """- Task completed successfully according to orchestrator quality gates
- User request fulfilled within orchestrator workflow context
- No further action required as determined by orchestrator coordination
- Clear completion criteria from orchestrator workflow met
- Current phase priorities defined by orchestrator respected"""


class AgentConfigScanner:
    """Scanner for agent configuration files and metadata"""
    
    def __init__(self, config_dir: Path = Path(".agent-config")):
        self.config_dir = Path(config_dir)
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        self._last_scan_time: Optional[datetime] = None
    
    async def scan_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """Scan and parse all agent configuration files"""
        if not self.config_dir.exists():
            logger.warning("Agent config directory %s does not exist", self.config_dir)
            return {}
        
        configs = {}
        
        # Scan main config files
        for config_file in self.config_dir.glob("*.md"):
            try:
                config_data = await self._parse_agent_config(config_file)
                configs[config_file.stem] = config_data
            except (IOError, yaml.YAMLError) as e:
                logger.error("Error parsing %s: %s", config_file, e)
        
        # Scan specialists directory
        specialists_dir = self.config_dir / "specialists"
        if specialists_dir.exists():
            for config_file in specialists_dir.glob("*.md"):
                try:
                    config_data = await self._parse_agent_config(config_file)
                    configs[f"specialists/{config_file.stem}"] = config_data
                except (IOError, yaml.YAMLError) as e:
                    logger.error("Error parsing %s: %s", config_file, e)
        
        self.agent_configs = configs
        self._last_scan_time = datetime.now()
        
        return configs
    
    async def _parse_agent_config(self, config_file: Path) -> Dict[str, Any]:
        """Parse an individual agent configuration file"""
        content = config_file.read_text()
        
        # Extract YAML frontmatter if present
        frontmatter = {}
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1])
                except yaml.YAMLError as e:
                    logger.warning("Invalid YAML frontmatter in %s: %s", config_file, e)
                content = parts[2]
        
        # Extract basic metadata from content
        metadata = {
            'name': config_file.stem,
            'path': str(config_file),
            'frontmatter': frontmatter or {},
            'has_orchestrator_routing': 'MANDATORY ORCHESTRATOR ROUTING' in content,
            'intersections': self._extract_intersections(content),
            'description': self._extract_description(content),
            'last_modified': datetime.fromtimestamp(config_file.stat().st_mtime)
        }
        
        return metadata
    
    def _extract_intersections(self, content: str) -> List[str]:
        """Extract intersection patterns from agent config content"""
        intersections = []
        lines = content.split('\n')
        
        for line in lines:
            if 'intersects with' in line.lower():
                # Extract agent names from intersection descriptions
                import re
                matches = re.findall(r'(\w+\.md|\w+_\w+\.md)', line)
                intersections.extend([match.replace('.md', '') for match in matches])
        
        return list(set(intersections))
    
    def _extract_description(self, content: str) -> str:
        """Extract description from agent config content"""
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith('## Role Overview') or line.startswith('## Overview'):
                # Find the next paragraph
                idx = lines.index(line)
                for i in range(idx + 1, len(lines)):
                    if lines[i].strip() and not lines[i].startswith('#'):
                        return lines[i].strip()
        
        return "Specialist agent for project coordination"
    
    async def get_config_changes_since(self, timestamp: datetime) -> List[str]:
        """Get list of agent configs that have changed since the given timestamp"""
        if not self.agent_configs:
            await self.scan_agent_configs()
        
        changed_configs = []
        for name, config in self.agent_configs.items():
            if config['last_modified'] > timestamp:
                changed_configs.append(name)
        
        return changed_configs
    
    def get_orchestrator_first_agents(self) -> List[str]:
        """Get list of agents that properly implement orchestrator-first routing"""
        return [
            name for name, config in self.agent_configs.items()
            if config['has_orchestrator_routing']
        ]


class CopilotInstructionGenerator:
    """Generate copilot instructions with MCP server integration"""
    
    def __init__(self, config_dir: str = "agent-config"):
        self.template_engine = TemplateEngine()
        self.agent_config_scanner = AgentConfigScanner(Path(config_dir))
        self.project_root = Path.cwd()
    
    async def generate_mcp_integration_instructions(
        self, 
        mcp_server_config: Dict[str, Any]
    ) -> str:
        """Generate MCP server integration instructions"""
        
        # Scan current agent configurations
        agent_configs = await self.agent_config_scanner.scan_agent_configs()
        
        # Determine current project phase
        project_phase = await self._detect_project_phase()
        
        # Build template context
        context = TemplateContext(
            mcp_server_config=mcp_server_config,
            agent_configs=agent_configs,
            project_phase=project_phase,
            project_context=await self._get_project_context(),
            workflow_patterns=await self._get_workflow_patterns()
        )
        
        # Generate instructions using template
        instructions = await self.template_engine.render_template(
            "copilot_instructions.md",
            context
        )
        
        return instructions
    
    async def generate_agent_workflow_instructions(self) -> str:
        """Generate agent workflow instructions from configs"""
        agent_configs = await self.agent_config_scanner.scan_agent_configs()
        
        instructions = """## Agent Workflow Instructions

### Orchestrator-Driven Multi-Agent Workflow

This project uses an orchestrator-driven approach where all work routes through the central orchestrator:

"""
        
        # List all agents with orchestrator routing
        orchestrator_agents = self.agent_config_scanner.get_orchestrator_first_agents()
        instructions += f"#### Agents with Proper Orchestrator Routing ({len(orchestrator_agents)} total):\n\n"
        
        for agent_name in sorted(orchestrator_agents):
            config = agent_configs.get(agent_name, {})
            instructions += f"- **{agent_name}**: {config.get('description', 'Agent description')}\n"
        
        # Add workflow coordination patterns
        instructions += """
### Multi-Agent Coordination Patterns

1. **Start with Orchestrator**: Always read `agent-config/orchestrator.md` first
2. **Follow Routing**: Use orchestrator's agent selection matrix
3. **Maintain Coordination**: Keep orchestrator informed of progress
4. **Respect Intersections**: Follow established collaboration patterns
5. **Apply Quality Gates**: Use orchestrator-defined validation requirements
"""
        
        return instructions
    
    async def _detect_project_phase(self) -> str:
        """Detect current project phase from comprehensive tasks"""
        try:
            tasks_file = self.project_root / "comprehensive_tasks.md"
            if tasks_file.exists():
                content = tasks_file.read_text()
                if "Phase 1" in content and "CURRENT" in content:
                    return "Phase 1: Enhanced Foundation Setup"
                elif "Phase 2" in content and "CURRENT" in content:
                    return "Phase 2: MCP Tools Implementation"
                elif "Phase 3" in content and "CURRENT" in content:
                    return "Phase 3: Integration Stack"
                elif "Phase 4" in content and "CURRENT" in content:
                    return "Phase 4: Complete Automation Integration"
        except (IOError, FileNotFoundError) as e:
            logger.error("Error detecting project phase: %s", e)
        
        return "Development Phase"
    
    async def _get_project_context(self) -> Dict[str, Any]:
        """Get current project context information"""
        return {
            'name': 'MCP Swarm Intelligence Server',
            'description': 'high-performance implementation of collective intelligence for multi-agent coordination',
            'features': [
                'agent ecosystem management',
                'hive mind knowledge bases',
                'persistent memory systems',
                'automated workflow orchestration'
            ],
            'current_focus': 'MCP server development with swarm intelligence capabilities',
            'current_goal': 'Complete automated project scaffolding with memory/swarm components',
            'key_tasks': 'Project structure, agent config deployment, CI/CD pipeline',
            'timeline': '30 hours across current phase'
        }
    
    async def _get_workflow_patterns(self) -> Dict[str, Any]:
        """Get workflow patterns information"""
        return {
            'single_agent': {
                'description': 'Simple tasks with orchestrator oversight',
                'pattern': 'Primary specialist + orchestrator coordination'
            },
            'multi_agent': {
                'description': 'Complex features with orchestrator coordination',
                'pattern': 'Primary + Secondary specialists + orchestrator management'
            },
            'emergency': {
                'description': 'Critical issues with orchestrator escalation',
                'pattern': 'Immediate escalation + orchestrator resource coordination'
            }
        }