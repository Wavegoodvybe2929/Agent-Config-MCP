"""
Copilot Instructions Manager MCP Tool

This MCP tool manages copilot instructions with integrated MCP server and
agent-config system support, providing automatic generation and updates
of comprehensive workflow documentation.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# MCP availability check
try:
    import mcp  # noqa: F401
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from ..docs import (
    AgentConfigScanner,
    CopilotInstructionGenerator,
    TemplateEngine,
)

logger = logging.getLogger(__name__)


def mcp_tool(name: str):
    """Decorator for MCP tools (placeholder implementation)."""
    def decorator(func):
        func.mcp_tool_name = name
        return func
    return decorator

# MCP Tool Definition (schema only for documentation)
copilot_instructions_manager_schema = {
    "name": "copilot_instructions_manager",
    "description": "Manage copilot instructions with MCP server integration and agent-config workflow support",
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "update", "validate", "generate_template"],
                "description": "Action to perform on copilot instructions"
            },
            "instruction_type": {
                "type": "string",
                "enum": ["full", "agent_only", "mcp_only"],
                "default": "full",
                "description": "Type of instructions to generate"
            },
            "mcp_server_config": {
                "type": "object",
                "description": "MCP server configuration details",
                "properties": {
                    "tools": {
                        "type": "array",
                        "items": {"type": "object"}
                    },
                    "resources": {
                        "type": "array", 
                        "items": {"type": "object"}
                    },
                    "server_info": {
                        "type": "object"
                    }
                }
            },
            "agent_workflow_config": {
                "type": "object",
                "description": "Agent workflow configuration overrides",
                "properties": {
                    "current_phase": {"type": "string"},
                    "priority_agents": {"type": "array", "items": {"type": "string"}},
                    "workflow_patterns": {"type": "object"}
                }
            },
            "output_path": {
                "type": "string",
                "default": ".github/copilot-instructions.md",
                "description": "Path where instructions should be written"
            },
            "auto_update_enabled": {
                "type": "boolean",
                "default": True,
                "description": "Enable automatic updates when agent configs change"
            }
        },
        "required": ["action"]
    }
}


class CopilotInstructionsManager:
    """Manages copilot instructions with MCP server and agent-config integration"""
    
    def __init__(self, config_dir: str = "agent-config"):
        self.instruction_generator = CopilotInstructionGenerator(config_dir)
        self.agent_scanner = AgentConfigScanner(Path(config_dir))
        self.template_engine = TemplateEngine()
        self.project_root = Path.cwd()
        self._last_update_time: Optional[datetime] = None
        self._watch_task: Optional[asyncio.Task] = None
    
    async def handle_copilot_instructions_manager(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle copilot instructions manager MCP tool requests"""
        
        action = arguments.get("action")
        instruction_type = arguments.get("instruction_type", "full")
        mcp_server_config = arguments.get("mcp_server_config", {})
        agent_workflow_config = arguments.get("agent_workflow_config", {})
        output_path = arguments.get("output_path", ".github/copilot-instructions.md")
        auto_update_enabled = arguments.get("auto_update_enabled", True)
        
        try:
            if action == "create":
                result = await self._create_instructions(
                    instruction_type, mcp_server_config, agent_workflow_config, output_path
                )
                if auto_update_enabled:
                    await self._start_auto_update_watch(output_path, mcp_server_config, agent_workflow_config)
                
            elif action == "update":
                result = await self._update_instructions(
                    instruction_type, mcp_server_config, agent_workflow_config, output_path
                )
                
            elif action == "validate":
                result = await self._validate_instructions(output_path)
                
            elif action == "generate_template":
                result = await self._generate_template(instruction_type)
                
            else:
                result = {"error": f"Unknown action: {action}"}
            
            return result
            
        except (IOError, ValueError, RuntimeError) as e:
            logger.error("Error in copilot instructions manager: %s", e)
            error_result = {
                "error": str(e),
                "action": action,
                "timestamp": datetime.now().isoformat()
            }
            return error_result
    
    async def _create_instructions(
        self,
        instruction_type: str,
        mcp_server_config: Dict[str, Any],
        _agent_workflow_config: Dict[str, Any],
        output_path: str
    ) -> Dict[str, Any]:
        """Create new copilot instructions"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate instructions based on type
        if instruction_type == "full":
            instructions = await self.instruction_generator.generate_mcp_integration_instructions(
                mcp_server_config
            )
        elif instruction_type == "agent_only":
            instructions = await self.instruction_generator.generate_agent_workflow_instructions()
        elif instruction_type == "mcp_only":
            instructions = await self._generate_mcp_only_instructions(mcp_server_config)
        else:
            raise ValueError(f"Unknown instruction type: {instruction_type}")
        
        # Write instructions to file
        output_file.write_text(instructions, encoding='utf-8')
        
        self._last_update_time = datetime.now()
        
        return {
            "action": "create",
            "instruction_type": instruction_type,
            "output_path": str(output_file),
            "file_size": len(instructions),
            "timestamp": self._last_update_time.isoformat(),
            "agent_configs_scanned": len(await self.agent_scanner.scan_agent_configs()),
            "mcp_tools_included": len(mcp_server_config.get("tools", [])),
            "success": True
        }
    
    async def _update_instructions(
        self,
        instruction_type: str,
        mcp_server_config: Dict[str, Any],
        agent_workflow_config: Dict[str, Any],
        output_path: str
    ) -> Dict[str, Any]:
        """Update existing copilot instructions"""
        
        output_file = Path(output_path)
        
        if not output_file.exists():
            return await self._create_instructions(
                instruction_type, mcp_server_config, agent_workflow_config, output_path
            )
        
        # Check if update is needed
        changes = await self._detect_changes_since_last_update()
        
        if not changes["needs_update"]:
            return {
                "action": "update",
                "result": "no_changes_needed",
                "last_update": self._last_update_time.isoformat() if self._last_update_time else None,
                "changes_checked": changes,
                "success": True
            }
        
        # Perform update
        return await self._create_instructions(
            instruction_type, mcp_server_config, agent_workflow_config, output_path
        )
    
    async def _validate_instructions(self, output_path: str) -> Dict[str, Any]:
        """Validate existing copilot instructions"""
        
        output_file = Path(output_path)
        
        if not output_file.exists():
            return {
                "action": "validate",
                "valid": False,
                "error": f"Instructions file does not exist at {output_path}",
                "recommendations": ["Run 'create' action to generate instructions"]
            }
        
        instructions_content = output_file.read_text(encoding='utf-8')
        validation_results = {
            "action": "validate",
            "file_exists": True,
            "file_size": len(instructions_content),
            "last_modified": datetime.fromtimestamp(output_file.stat().st_mtime).isoformat(),
            "valid": True,
            "checks": {},
            "recommendations": []
        }
        
        # Check for required sections
        required_sections = [
            "MANDATORY ORCHESTRATOR-FIRST WORKFLOW",
            "Agent Configuration System",
            "MCP Server Integration",
            "Workflow Rules",
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in instructions_content:
                missing_sections.append(section)
        
        validation_results["checks"]["required_sections"] = {
            "total": len(required_sections),
            "found": len(required_sections) - len(missing_sections),
            "missing": missing_sections
        }
        
        if missing_sections:
            validation_results["valid"] = False
            validation_results["recommendations"].append(
                f"Add missing required sections: {', '.join(missing_sections)}"
            )
        
        # Check for orchestrator routing
        has_orchestrator_routing = "ALWAYS START WITH ORCHESTRATOR" in instructions_content
        validation_results["checks"]["orchestrator_routing"] = has_orchestrator_routing
        
        if not has_orchestrator_routing:
            validation_results["valid"] = False
            validation_results["recommendations"].append(
                "Add clear orchestrator-first workflow instructions"
            )
        
        # Check for MCP integration
        has_mcp_integration = "MCP Server Integration" in instructions_content
        validation_results["checks"]["mcp_integration"] = has_mcp_integration
        
        if not has_mcp_integration:
            validation_results["recommendations"].append(
                "Add MCP server integration documentation"
            )
        
        # Check if agent configs have changed since last update
        changes = await self._detect_changes_since_last_update()
        validation_results["checks"]["config_changes"] = changes
        
        if changes["needs_update"]:
            validation_results["recommendations"].append(
                "Agent configurations have changed, consider running 'update' action"
            )
        
        return validation_results
    
    async def _generate_template(self, instruction_type: str) -> Dict[str, Any]:
        """Generate template for copilot instructions"""
        
        if instruction_type == "full":
            template_name = "copilot_instructions.md"
        elif instruction_type == "agent_only":
            template_name = "agent_workflow_instructions.md"
        elif instruction_type == "mcp_only":
            template_name = "mcp_integration_instructions.md"
        else:
            raise ValueError(f"Unknown instruction type: {instruction_type}")
        
        # Create template if it doesn't exist
        template_path = self.template_engine.template_dir / f"{template_name}.template"
        
        if not template_path.exists():
            await self._create_template_file(template_name, instruction_type)
        
        template_content = template_path.read_text()
        
        return {
            "action": "generate_template",
            "template_type": instruction_type,
            "template_path": str(template_path),
            "template_size": len(template_content),
            "template_content": template_content,
            "success": True
        }
    
    async def _generate_mcp_only_instructions(self, mcp_server_config: Dict[str, Any]) -> str:
        """Generate MCP-only copilot instructions"""
        
        tools = mcp_server_config.get("tools", [])
        resources = mcp_server_config.get("resources", [])
        server_info = mcp_server_config.get("server_info", {})
        
        instructions = f"""# MCP Server Integration Instructions

## Server Information

{json.dumps(server_info, indent=2)}

## Available MCP Tools ({len(tools)} total)

"""
        
        for tool in tools:
            name = tool.get("name", "unknown_tool")
            description = tool.get("description", "No description available")
            instructions += f"- **{name}**: {description}\n"
        
        instructions += f"""
## Available Resources ({len(resources)} total)

"""
        
        for resource in resources:
            name = resource.get("name", "unknown_resource")
            description = resource.get("description", "No description available")
            instructions += f"- **{name}**: {description}\n"
        
        instructions += """
## Usage Patterns

### Tool Execution
- All MCP tools are executed through the server's tool handling system
- Tools integrate with the agent-config workflow management
- Quality gates and orchestrator coordination apply to all tool operations

### Resource Access
- Resources are accessed through the MCP resource protocol
- Resource content is validated and cached for performance
- Agent-config system manages resource access permissions

### Error Handling
- MCP protocol errors are handled gracefully with appropriate fallbacks
- Agent coordination ensures error recovery through alternative workflows
- All errors are logged and reported through the orchestrator system
"""
        
        return instructions
    
    async def _create_template_file(self, template_name: str, instruction_type: str):
        """Create a template file for the given instruction type"""
        
        template_path = self.template_engine.template_dir / f"{template_name}.template"
        
        if instruction_type == "agent_only":
            template_content = """# Agent Workflow Instructions

## Orchestrator-Driven Multi-Agent Workflow

{agent_workflow_instructions}

## Agent Configuration Hierarchy

{agent_hierarchy}

## Multi-Agent Coordination Patterns

{coordination_patterns}
"""
        elif instruction_type == "mcp_only":
            template_content = """# MCP Server Integration

## Server Configuration

{mcp_server_config}

## Available Tools

{mcp_tools_list}

## Integration Patterns

{mcp_integration_patterns}
"""
        else:
            # Default full template already created in TemplateEngine
            return
        
        template_path.write_text(template_content, encoding='utf-8')
    
    async def _detect_changes_since_last_update(self) -> Dict[str, Any]:
        """Detect if agent configs or MCP server state has changed since last update"""
        
        if self._last_update_time is None:
            return {"needs_update": True, "reason": "no_previous_update"}
        
        # Check for agent config changes
        changed_configs = await self.agent_scanner.get_config_changes_since(
            self._last_update_time
        )
        
        # Check comprehensive tasks for phase changes
        phase_changed = await self._check_project_phase_change()
        
        changes = {
            "needs_update": bool(changed_configs) or phase_changed,
            "changed_agent_configs": changed_configs,
            "phase_changed": phase_changed,
            "last_update": self._last_update_time.isoformat(),
            "check_time": datetime.now().isoformat()
        }
        
        if changed_configs:
            changes["reason"] = f"agent_configs_changed: {', '.join(changed_configs)}"
        elif phase_changed:
            changes["reason"] = "project_phase_changed"
        
        return changes
    
    async def _check_project_phase_change(self) -> bool:
        """Check if project phase has changed since last update"""
        
        if self._last_update_time is None:
            return True
        
        try:
            tasks_file = self.project_root / "comprehensive_tasks.md"
            if not tasks_file.exists():
                return False
            
            # Check if file was modified since last update
            last_modified = datetime.fromtimestamp(tasks_file.stat().st_mtime)
            return last_modified > self._last_update_time
            
        except (IOError, OSError) as e:
            logger.warning("Error checking project phase change: %s", e)
            return False
    
    async def _start_auto_update_watch(
        self,
        output_path: str,
        mcp_server_config: Dict[str, Any],
        agent_workflow_config: Dict[str, Any]
    ):
        """Start background task to watch for changes and auto-update instructions"""
        
        if self._watch_task and not self._watch_task.done():
            self._watch_task.cancel()
        
        self._watch_task = asyncio.create_task(
            self._auto_update_watcher(output_path, mcp_server_config, agent_workflow_config)
        )
    
    async def _auto_update_watcher(
        self,
        output_path: str,
        mcp_server_config: Dict[str, Any],
        agent_workflow_config: Dict[str, Any]
    ):
        """Background watcher for automatic updates"""
        
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                changes = await self._detect_changes_since_last_update()
                
                if changes["needs_update"]:
                    logger.info("Auto-updating copilot instructions due to: %s", 
                              changes.get("reason", "unknown changes"))
                    
                    await self._update_instructions(
                        "full", mcp_server_config, agent_workflow_config, output_path
                    )
                    
        except asyncio.CancelledError:
            logger.info("Auto-update watcher cancelled")
        except (IOError, RuntimeError) as e:
            logger.error("Error in auto-update watcher: %s", e)


# Global instance for MCP server integration
copilot_instructions_manager = CopilotInstructionsManager(config_dir="agent-config")


async def handle_copilot_instructions_manager_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle copilot instructions manager tool requests"""
    return await copilot_instructions_manager.handle_copilot_instructions_manager(arguments)