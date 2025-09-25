---
up: 
related: 
created: 2025-08-26 12:07
daily_note: '[[4 - Archives/1 - Daily Notes/2025-08-26|2025-08-26]]'
aliases: 
tags:
---

  

# NetworkChuck's MCP Server Builder Prompt

  

## INITIAL CLARIFICATIONS

### 1. **Service/Tool Name**: 
**MCP Swarm Intelligence Server**

### 2. **API Documentation**: 
This server integrates with:
- Model Context Protocol (MCP) specification: https://spec.modelcontextprotocol.io/
- Internal SQLite database for persistent memory and hive mind knowledge
- Agent configuration file system monitoring
- No external APIs required - fully self-contained

### 3. **Required Features**:

#### **Core Swarm Intelligence Features:**
- **Agent Assignment**: Optimal task assignment using ACO (Ant Colony Optimization) algorithms
- **Hive Mind Knowledge Management**: Collective knowledge base with semantic search
- **Swarm Consensus**: Democratic decision-making across multiple agents
- **Adaptive Coordination**: Dynamic strategy selection and real-time coordination
- **Memory Persistence**: Cross-session state management with SQLite backend

#### **MCP Tools to Implement (37+ tools):**
1. **agent_assignment** - Assign tasks to optimal agents using swarm intelligence
2. **hive_mind_query** - Query collective knowledge base with semantic search
3. **swarm_consensus** - Reach consensus on decisions using swarm algorithms
4. **adaptive_coordination** - Dynamically coordinate multiple agents
5. **agent_config_manager** - Manage agent configuration files
6. **copilot_instructions_manager** - Manage copilot instructions with MCP integration
7. **agent_discovery** - Automatically discover and register agent capabilities
8. **ecosystem_management** - Monitor and manage agent ecosystem health
9. **dynamic_coordination** - Real-time coordination strategy selection
10. **complete_pipeline** - Execute complete multi-agent workflows
11. **knowledge_contribution** - Contribute knowledge to hive mind
12. **knowledge_extraction** - Extract structured knowledge from data
13. **knowledge_synthesis** - Synthesize knowledge from multiple sources
14. **knowledge_validation** - Validate knowledge quality and consistency
15. **automation_validation** - Validate automated processes and workflows
16. **self_monitoring** - Monitor server and agent health metrics
17. **decision_audit** - Audit decision-making processes
18. **risk_assessment** - Assess risks in agent coordination
19. **directory_manager** - Manage project directory structures
20. **agent_hooks** - Execute agent lifecycle hooks
21. **confidence_aggregation** - Aggregate confidence scores across agents
22. **consensus_algorithms** - Apply various consensus mechanisms
23. **coordination_strategies** - Select optimal coordination strategies
24. **decision_confidence** - Calculate decision confidence metrics
25. **explanation** - Provide explanations for agent decisions
26. **fuzzy_matcher** - Match capabilities using fuzzy logic
27. **knowledge_classifier** - Classify knowledge by domain and type
28. **knowledge_quality** - Assess knowledge quality metrics
29. **load_balancer** - Balance workload across agents
30. **mcda** - Multi-criteria decision analysis
31. **minority_opinion** - Capture and analyze minority opinions
32. **semantic_search** - Semantic search across knowledge base
33. **strategy_selector** - Select optimal coordination strategies
34. **mcp_server_manager** - Manage MCP server lifecycle
35. **adaptive_learning** - Machine learning for agent optimization
36. **knowledge_updater** - Update knowledge base with versioning
37. **coordination_pattern_learning** - Learn from successful coordination patterns

#### **Advanced Features:**
- **Persistent Memory System**: SQLite-based cross-session memory
- **Real-time Agent Monitoring**: File system monitoring for configuration changes
- **Swarm Optimization Algorithms**: ACO, PSO (Particle Swarm Optimization)
- **Multi-criteria Decision Making**: MCDA algorithms for complex decisions
- **Quality Gates**: Automated validation and quality assurance
- **Performance Metrics**: Comprehensive monitoring and metrics collection

### 4. **Authentication**: 
- No external authentication required
- Internal access control based on agent configurations
- Optional environment variables for advanced security:
  - `SWARM_DB_ENCRYPTION_KEY` (optional database encryption)
  - `SWARM_ADMIN_TOKEN` (optional admin access)

### 5. **Data Sources**: 
- **SQLite Database**: Primary persistent storage (`data/memory.db`)
- **Agent Configuration Files**: YAML/Markdown files in `agent-config/` directory
- **File System Monitoring**: Real-time monitoring of configuration changes
- **Internal Knowledge Base**: Structured knowledge storage with semantic indexing
- **Metrics Database**: Performance and usage analytics

### 6. **Additional Technical Requirements**:

#### **Dependencies:**
- **Core**: mcp>=1.0.0, asyncio, uvicorn, fastapi
- **ML/AI**: numpy, scipy, scikit-learn, networkx
- **Database**: aiosqlite, SQLite with JSON1 and FTS5 extensions
- **Monitoring**: prometheus-client, psutil, watchdog
- **Utilities**: pydantic, click, python-dotenv, structlog, PyYAML

#### **Database Schema:**
- **agents** - Agent registry and capabilities
- **tasks** - Task assignment history
- **hive_knowledge** - Collective knowledge base
- **consensus_decisions** - Decision history with confidence scores
- **coordination_patterns** - Successful coordination patterns
- **performance_metrics** - System performance data
- **agent_interactions** - Inter-agent communication logs

#### **Special Configuration:**
- Supports both stdio transport (for Claude Desktop) and HTTP transport
- Automatic tool discovery and registration
- Configuration hot-reloading
- Graceful shutdown with state persistence
- Comprehensive logging with structured output

### 7. **VS Code Extension Integration**: 
- **VS Code API Integration**: Full support for VS Code Extension API including:
  - `vscode.commands` - Command registration and execution
  - `vscode.workspace` - Workspace folder and file management
  - `vscode.window` - UI integration (status bar, notifications, webviews)
  - `vscode.languages` - Language service provider registration
  - `vscode.extensions` - Extension lifecycle and inter-extension communication
  - `vscode.lm` - Language model integration for MCP server registration
- **MCP Server Definition Provider**: Register with `lm.registerMcpServerDefinitionProvider()`
- **Chat Participant Integration**: Create chat participants with `chat.createChatParticipant()`
- **Agent Configuration File Integration**: Direct file system access to agent-config directory
- **VS Code Settings Integration**: Configuration via `workspace.getConfiguration()`

### 8. **Docker MCP Toolkit Integration**: 
- **Container-based Deployment**: Full Docker containerization with MCP Toolkit support
- **MCP Gateway Compatibility**: Works with Docker MCP Gateway for client aggregation
- **Security Features**: 
  - CPU/Memory limits (1 CPU, 2GB RAM)
  - Filesystem isolation with selective mounts
  - Request interception for sensitive data
- **Client Support**: Compatible with Claude Desktop, VS Code, Cursor, Continue.dev
- **OAuth Integration**: GitHub OAuth for agent-config repository access
- **One-click Setup**: Zero manual configuration deployment

### 9. **GitHub Copilot REST API Integration**: 
- **Copilot Metrics API**: Integration with GitHub Copilot metrics endpoints
- **User Management API**: Copilot seat management for team coordination
- **Agent Configuration Repository**: Direct integration with GitHub repositories
- **Team Collaboration**: Multi-user agent configuration management
- **Usage Analytics**: Track agent performance and coordination metrics

**Architecture Overview:**
This server implements a complete swarm intelligence system for multi-agent coordination with full VS Code extension integration and GitHub Copilot API support. It features Docker MCP Toolkit deployment, making it ideal for complex development workflows requiring intelligent task distribution, collective decision-making, and persistent learning across sessions. The system combines queen-led hierarchical coordination with worker agent specialization, featuring a persistent hive mind knowledge base, VS Code extension APIs for deep IDE integration, and GitHub Copilot REST API connectivity for team collaboration and metrics tracking.


  

---

  

# INSTRUCTIONS FOR THE LLM

  

## YOUR ROLE

You are an expert MCP (Model Context Protocol) server developer. You will create a complete, working MCP server based on the user's requirements.

  

## CLARIFICATION PROCESS

Before generating the server, ensure you have:

1. **Service name and description** - Clear understanding of what the server does

2. **API documentation** - If integrating with external services, fetch and review API docs

3. **Tool requirements** - Specific list of tools/functions needed

4. **Authentication needs** - API keys, OAuth tokens, or other auth requirements

5. **Output preferences** - Any specific formatting or response requirements

  

  

## YOUR OUTPUT STRUCTURE

You must organize your response in TWO distinct sections:

  

### SECTION 1: FILES TO CREATE

Generate EXACTLY these 5 files with complete content that the user can copy and save.

**DO NOT** create duplicate files or variations. Each file should appear ONCE with its complete content.

  

### SECTION 2: INSTALLATION INSTRUCTIONS FOR THE USER

Provide step-by-step commands the user needs to run on their computer.

Present these as a clean, numbered list without creating duplicate instruction sets.

  

## CRITICAL RULES FOR CODE GENERATION

1. **NO `@mcp.prompt()` decorators** - They break Claude Desktop

2. **NO `prompt` parameter to FastMCP()** - It breaks Claude Desktop

3. **NO type hints from typing module** - No `Optional`, `Union`, `List[str]`, etc.

4. **NO complex parameter types** - Use `param: str = ""` not `param: str = None`

5. **SINGLE-LINE DOCSTRINGS ONLY** - Multi-line docstrings cause gateway panic errors

6. **DEFAULT TO EMPTY STRINGS** - Use `param: str = ""` never `param: str = None`

7. **ALWAYS return strings from tools** - All tools must return formatted strings

8. **ALWAYS use Docker** - The server must run in a Docker container

9. **ALWAYS log to stderr** - Use the logging configuration provided

10. **ALWAYS handle errors gracefully** - Return user-friendly error messages

  

---

  

# SECTION 1: FILES TO CREATE

  

## File 1: Dockerfile

```dockerfile

# Use Python slim image

FROM python:3.11-slim

  

# Set working directory

WORKDIR /app

  

# Set Python unbuffered mode

ENV PYTHONUNBUFFERED=1

  

# Copy requirements first for better caching

COPY requirements.txt .

  

# Install dependencies

RUN pip install --no-cache-dir -r requirements.txt

  

# Copy the server code

COPY [SERVER_NAME]_server.py .

  

# Create non-root user

RUN useradd -m -u 1000 mcpuser && \

&nbsp;&nbsp;&nbsp;&nbsp;chown -R mcpuser:mcpuser /app

  

# Switch to non-root user

USER mcpuser

  

# Run the server

CMD ["python", "[SERVER_NAME]_server.py"]

```

  

## File 2: requirements.txt

```

mcp[cli]>=1.2.0

httpx

# Add any other required libraries based on the user's needs

```

  

## File 3: [SERVER_NAME]_server.py

```python

#!/usr/bin/env python3

"""

Simple [SERVICE_NAME] MCP Server - [DESCRIPTION]

"""

import os

import sys

import logging

from datetime import datetime, timezone

import httpx

from mcp.server.fastmcp import FastMCP

  

# Configure logging to stderr

logging.basicConfig(

&nbsp;&nbsp;&nbsp;&nbsp;level=logging.INFO,

&nbsp;&nbsp;&nbsp;&nbsp;format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',

&nbsp;&nbsp;&nbsp;&nbsp;stream=sys.stderr

)

logger = logging.getLogger("[SERVER_NAME]-server")

  

# Initialize MCP server - NO PROMPT PARAMETER!

mcp = FastMCP("[SERVER_NAME]")

  

# Configuration

# Add any API keys, URLs, or configuration here

# API_TOKEN = os.environ.get("[SERVER_NAME_UPPER]_API_TOKEN", "")

  

# === UTILITY FUNCTIONS ===

# Add utility functions as needed

  

# === MCP TOOLS ===

# Create tools based on user requirements

# Each tool must:

# - Use @mcp.tool() decorator

# - Have SINGLE-LINE docstrings only

# - Use empty string defaults (param: str = "") NOT None

# - Have simple parameter types

# - Return a formatted string

# - Include proper error handling

# WARNING: Multi-line docstrings will cause gateway panic errors!

  

@mcp.tool()

async def example_tool(param: str = "") -> str:

&nbsp;&nbsp;&nbsp;&nbsp;"""Single-line description of what this tool does - MUST BE ONE LINE."""

&nbsp;&nbsp;&nbsp;&nbsp;logger.info(f"Executing example_tool with {param}")

&nbsp;&nbsp;&nbsp;&nbsp;

&nbsp;&nbsp;&nbsp;&nbsp;try:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Implementation here

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;result = "example"

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return f"✅ Success: {result}"

&nbsp;&nbsp;&nbsp;&nbsp;except Exception as e:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.error(f"Error: {e}")

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return f"❌ Error: {str(e)}"

  

# === SERVER STARTUP ===

if __name__ == "__main__":

&nbsp;&nbsp;&nbsp;&nbsp;logger.info("Starting [SERVICE_NAME] MCP server...")

&nbsp;&nbsp;&nbsp;&nbsp;

&nbsp;&nbsp;&nbsp;&nbsp;# Add any startup checks

&nbsp;&nbsp;&nbsp;&nbsp;# if not API_TOKEN:

&nbsp;&nbsp;&nbsp;&nbsp;# logger.warning("[SERVER_NAME_UPPER]_API_TOKEN not set")

&nbsp;&nbsp;&nbsp;&nbsp;

&nbsp;&nbsp;&nbsp;&nbsp;try:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mcp.run(transport='stdio')

&nbsp;&nbsp;&nbsp;&nbsp;except Exception as e:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logger.error(f"Server error: {e}", exc_info=True)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sys.exit(1)

```

  

## File 4: readme.txt

Create a comprehensive readme with all sections filled in based on the implementation.

  

## File 5: CLAUDE.md

Create a CLAUDE.md file with implementation details and guidelines.

  

---

  

# SECTION 2: INSTALLATION INSTRUCTIONS FOR THE USER

  

After creating the files above, provide these instructions for the user to run:

  

## Step 1: Save the Files

```bash

# Create project directory

mkdir [SERVER_NAME]-mcp-server

cd [SERVER_NAME]-mcp-server

  

# Save all 5 files in this directory

```

  

## Step 2: Build Docker Image

```bash

docker build -t [SERVER_NAME]-mcp-server .

```

  

## Step 3: Set Up Secrets (if needed)

```bash

# Only include if the server needs API keys or secrets

docker mcp secret set [SECRET_NAME]="your-secret-value"

  

# Verify secrets

docker mcp secret list

```

  

## Step 4: Create Custom Catalog

```bash

# Create catalogs directory if it doesn't exist

mkdir -p ~/.docker/mcp/catalogs

  

# Create or edit custom.yaml

nano ~/.docker/mcp/catalogs/custom.yaml

```

  

Add this entry to custom.yaml:

```yaml

version: 2

name: custom

displayName: Custom MCP Servers

registry:

&nbsp;&nbsp;[SERVER_NAME]:

&nbsp;&nbsp;&nbsp;&nbsp;description: "[DESCRIPTION]"

&nbsp;&nbsp;&nbsp;&nbsp;title: "[SERVICE_NAME]"

&nbsp;&nbsp;&nbsp;&nbsp;type: server

&nbsp;&nbsp;&nbsp;&nbsp;dateAdded: "[CURRENT_DATE]" # Format: 2025-01-01T00:00:00Z

&nbsp;&nbsp;&nbsp;&nbsp;image: [SERVER_NAME]-mcp-server:latest

&nbsp;&nbsp;&nbsp;&nbsp;ref: ""

&nbsp;&nbsp;&nbsp;&nbsp;readme: ""

&nbsp;&nbsp;&nbsp;&nbsp;toolsUrl: ""

&nbsp;&nbsp;&nbsp;&nbsp;source: ""

&nbsp;&nbsp;&nbsp;&nbsp;upstream: ""

&nbsp;&nbsp;&nbsp;&nbsp;icon: ""

&nbsp;&nbsp;&nbsp;&nbsp;tools:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- name: [tool_name_1]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- name: [tool_name_2]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# List all tools

&nbsp;&nbsp;&nbsp;&nbsp;secrets:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- name: [SECRET_NAME]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;env: [ENV_VAR_NAME]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;example: [EXAMPLE_VALUE]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Only include if using secrets

&nbsp;&nbsp;&nbsp;&nbsp;metadata:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;category: [Choose: productivity|monitoring|automation|integration]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tags:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [relevant_tag_1]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [relevant_tag_2]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;license: MIT

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;owner: local

```

  

## Step 5: Update Registry

```bash

# Edit registry file

nano ~/.docker/mcp/registry.yaml

```

  

Add this entry under the existing `registry:` key:

```yaml

registry:

&nbsp;&nbsp;# ... existing servers ...

&nbsp;&nbsp;[SERVER_NAME]:

&nbsp;&nbsp;&nbsp;&nbsp;ref: ""

```

  

**IMPORTANT**: The entry must be under the `registry:` key, not at the root level.

  

## Step 6: Configure Claude Desktop

  

Find your Claude Desktop config file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

- **Linux**: `~/.config/Claude/claude_desktop_config.json`

  

Edit the file and add your custom catalog to the args array:

```json

{

&nbsp;&nbsp;"mcpServers": {

&nbsp;&nbsp;&nbsp;&nbsp;"mcp-toolkit-gateway": {

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"command": "docker",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"args": [

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"run",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"-i",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"--rm",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"-v", "/var/run/docker.sock:/var/run/docker.sock",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"-v", "[YOUR_HOME]/.docker/mcp:/mcp",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"docker/mcp-gateway",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"--catalog=/mcp/catalogs/docker-mcp.yaml",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"--catalog=/mcp/catalogs/custom.yaml",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"--config=/mcp/config.yaml",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"--registry=/mcp/registry.yaml",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"--tools-config=/mcp/tools.yaml",

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"--transport=stdio"

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;]

&nbsp;&nbsp;&nbsp;&nbsp;}

&nbsp;&nbsp;}

}

```

  

**NOTE**: JSON does not support comments. The custom.yaml catalog line should be added without any comment.

  

Replace `[YOUR_HOME]` with:

- **macOS**: `/Users/your_username`

- **Windows**: `C:\\Users\\your_username` (use double backslashes)

- **Linux**: `/home/your_username`

  

## Step 7: Restart Claude Desktop

1. Quit Claude Desktop completely

2. Start Claude Desktop again

3. Your new tools should appear!

  

## Step 8: Test Your Server

```bash

# Verify it appears in the list

docker mcp server list

  

# If you don't see your server, check logs:

docker logs [container_name]

```

  

---

  

# IMPLEMENTATION PATTERNS FOR THE LLM

  

## CORRECT Tool Implementation:

```python

@mcp.tool()

async def fetch_data(endpoint: str = "", limit: str = "10") -> str:

&nbsp;&nbsp;&nbsp;&nbsp;"""Fetch data from API endpoint with optional limit."""

&nbsp;&nbsp;&nbsp;&nbsp;# Check for empty strings, not just truthiness

&nbsp;&nbsp;&nbsp;&nbsp;if not endpoint.strip():

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return "❌ Error: Endpoint is required"

&nbsp;&nbsp;&nbsp;&nbsp;

&nbsp;&nbsp;&nbsp;&nbsp;try:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Convert string parameters as needed

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;limit_int = int(limit) if limit.strip() else 10

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Implementation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return f"✅ Fetched {limit_int} items"

&nbsp;&nbsp;&nbsp;&nbsp;except ValueError:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return f"❌ Error: Invalid limit value: {limit}"

&nbsp;&nbsp;&nbsp;&nbsp;except Exception as e:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return f"❌ Error: {str(e)}"

```

  

## For API Integration:

```python

async with httpx.AsyncClient() as client:

&nbsp;&nbsp;&nbsp;&nbsp;try:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;response = await client.get(url, headers=headers, timeout=10)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;response.raise_for_status()

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;data = response.json()

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Process and format data

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return f"✅ Result: {formatted_data}"

&nbsp;&nbsp;&nbsp;&nbsp;except httpx.HTTPStatusError as e:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return f"❌ API Error: {e.response.status_code}"

&nbsp;&nbsp;&nbsp;&nbsp;except Exception as e:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return f"❌ Error: {str(e)}"

```

  

## For System Commands:

```python

import subprocess

try:

&nbsp;&nbsp;&nbsp;&nbsp;result = subprocess.run(

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;command,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;capture_output=True,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;text=True,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;timeout=10,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;shell=True # Only if needed

&nbsp;&nbsp;&nbsp;&nbsp;)

&nbsp;&nbsp;&nbsp;&nbsp;if result.returncode == 0:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return f"✅ Output:\n{result.stdout}"

&nbsp;&nbsp;&nbsp;&nbsp;else:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return f"❌ Error:\n{result.stderr}"

except subprocess.TimeoutExpired:

&nbsp;&nbsp;&nbsp;&nbsp;return "⏱️ Command timed out"

```

  

## For File Operations:

```python

try:

&nbsp;&nbsp;&nbsp;&nbsp;with open(filename, 'r') as f:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;content = f.read()

&nbsp;&nbsp;&nbsp;&nbsp;return f"✅ File content:\n{content}"

except FileNotFoundError:

&nbsp;&nbsp;&nbsp;&nbsp;return f"❌ File not found: {filename}"

except Exception as e:

&nbsp;&nbsp;&nbsp;&nbsp;return f"❌ Error reading file: {str(e)}"

```

  

## OUTPUT FORMATTING GUIDELINES

  

Use emojis for visual clarity:

- ✅ Success operations

- ❌ Errors or failures

- ⏱️ Time-related information

- 📊 Data or statistics

- 🔍 Search or lookup operations

- ⚡ Actions or commands

- 🔒 Security-related information

- 📁 File operations

- 🌐 Network operations

- ⚠️ Warnings

  

Format multi-line output clearly:

```python

return f"""📊 Results:

- Field 1: {value1}

- Field 2: {value2}

- Field 3: {value3}

  

Summary: {summary}"""

```

  

## COMPLETE README.TXT TEMPLATE

  

```markdown

# [SERVICE_NAME] MCP Server

  

A Model Context Protocol (MCP) server that [DESCRIPTION].

  

## Purpose

  

This MCP server provides a secure interface for AI assistants to [MAIN_PURPOSE].

  

## Features

  

### Current Implementation

- **`[tool_name_1]`** - [What it does]

- **`[tool_name_2]`** - [What it does]

[LIST ALL TOOLS]

  

## Prerequisites

  

- Docker Desktop with MCP Toolkit enabled

- Docker MCP CLI plugin (`docker mcp` command)

[ADD ANY SERVICE-SPECIFIC REQUIREMENTS]

  

## Installation

  

See the step-by-step instructions provided with the files.

  

## Usage Examples

  

In Claude Desktop, you can ask:

- "[Natural language example 1]"

- "[Natural language example 2]"

[PROVIDE EXAMPLES FOR EACH TOOL]

  

## Architecture

  

```

Claude Desktop → MCP Gateway → [SERVICE_NAME] MCP Server → [SERVICE/API]

↓

Docker Desktop Secrets

([SECRET_NAMES])

```

  

## Development

  

### Local Testing

  

```bash

# Set environment variables for testing

export [SECRET_NAME]="test-value"

  

# Run directly

python [SERVER_NAME]_server.py

  

# Test MCP protocol

echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python [SERVER_NAME]_server.py

```

  

### Adding New Tools

  

1. Add the function to `[SERVER_NAME]_server.py`

2. Decorate with `@mcp.tool()`

3. Update the catalog entry with the new tool name

4. Rebuild the Docker image

  

## Troubleshooting

  

### Tools Not Appearing

- Verify Docker image built successfully

- Check catalog and registry files

- Ensure Claude Desktop config includes custom catalog

- Restart Claude Desktop

  

### Authentication Errors

- Verify secrets with `docker mcp secret list`

- Ensure secret names match in code and catalog

  

## Security Considerations

  

- All secrets stored in Docker Desktop secrets

- Never hardcode credentials

- Running as non-root user

- Sensitive data never logged

  

## License

  

MIT License

```

  

## FINAL GENERATION CHECKLIST FOR THE LLM

  

Before presenting your response, verify:

- [ ] Created all 5 files with proper naming

- [ ] No @mcp.prompt() decorators used

- [ ] No prompt parameter in FastMCP()

- [ ] No complex type hints

- [ ] ALL tool docstrings are SINGLE-LINE only

- [ ] ALL parameters default to empty strings ("") not None

- [ ] All tools return strings

- [ ] Check for empty strings with .strip() not just truthiness

- [ ] Error handling in every tool

- [ ] Clear separation between files and user instructions

- [ ] All placeholders replaced with actual values

- [ ] Usage examples provided

- [ ] Security handled via Docker secrets

- [ ] Catalog includes version: 2, name, displayName, and registry wrapper

- [ ] Registry entries are under registry: key with ref: ""

- [ ] Date format is ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)

- [ ] Claude config JSON has no comments

- [ ] Each file appears exactly once

- [ ] Instructions are clear and numbered