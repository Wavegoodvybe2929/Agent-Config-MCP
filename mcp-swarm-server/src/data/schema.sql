-- MCP Swarm Intelligence Server Database Schema
-- SQLite database for persistent memory and swarm coordination
-- Version: 1.0.0
-- Created: September 17, 2025

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA temp_store = MEMORY;

-- Core agents table for agent registration and management
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    capabilities TEXT, -- JSON array of capability strings
    status TEXT NOT NULL DEFAULT 'inactive', -- active, inactive, error, maintenance
    current_load REAL DEFAULT 0.0,
    success_rate REAL DEFAULT 1.0,
    last_heartbeat TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge entries for hive mind collective intelligence
CREATE TABLE IF NOT EXISTS knowledge_entries (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source TEXT, -- agent_id or external source
    confidence REAL DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    embedding BLOB, -- vector embedding for semantic search
    tags TEXT, -- JSON array of tags
    category TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP -- optional expiration
);

-- Swarm state for coordination and collective decision making
CREATE TABLE IF NOT EXISTS swarm_state (
    id TEXT PRIMARY KEY,
    pheromone_data BLOB, -- serialized pheromone trail data
    consensus_data BLOB, -- current consensus state
    coordination_mode TEXT DEFAULT 'queen_led', -- queen_led, democratic, hierarchical
    active_agents INTEGER DEFAULT 0,
    last_consensus TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Task execution history for learning and optimization
CREATE TABLE IF NOT EXISTS task_history (
    id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    agent_id TEXT,
    task_data TEXT, -- JSON task parameters
    success BOOLEAN,
    execution_time REAL, -- execution time in seconds
    error_message TEXT,
    performance_metrics TEXT, -- JSON metrics data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE SET NULL
);

-- Memory sessions for cross-session persistence
CREATE TABLE IF NOT EXISTS memory_sessions (
    id TEXT PRIMARY KEY,
    session_data BLOB, -- serialized session state
    session_type TEXT DEFAULT 'agent_state', -- agent_state, coordination, knowledge
    agent_id TEXT,
    metadata TEXT, -- JSON metadata
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
);

-- Pheromone trails for swarm intelligence coordination
CREATE TABLE IF NOT EXISTS pheromone_trails (
    id TEXT PRIMARY KEY,
    source_agent TEXT NOT NULL,
    target_agent TEXT NOT NULL,
    trail_type TEXT NOT NULL, -- task_assignment, communication, resource_sharing
    strength REAL NOT NULL DEFAULT 1.0 CHECK (strength >= 0.0),
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    last_reinforcement TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_agent) REFERENCES agents(id) ON DELETE CASCADE,
    FOREIGN KEY (target_agent) REFERENCES agents(id) ON DELETE CASCADE
);

-- MCP tool registry for dynamic tool discovery
CREATE TABLE IF NOT EXISTS mcp_tools (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    parameters_schema TEXT, -- JSON schema for tool parameters
    agent_id TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    usage_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE SET NULL
);

-- Resource management for MCP resources
CREATE TABLE IF NOT EXISTS mcp_resources (
    id TEXT PRIMARY KEY,
    uri TEXT NOT NULL UNIQUE,
    resource_type TEXT NOT NULL, -- text, image, binary
    content_type TEXT,
    size INTEGER,
    checksum TEXT,
    metadata TEXT, -- JSON metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_last_heartbeat ON agents(last_heartbeat);
CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge_entries(source);
CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge_entries(category);
CREATE INDEX IF NOT EXISTS idx_knowledge_created_at ON knowledge_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_task_history_agent_id ON task_history(agent_id);
CREATE INDEX IF NOT EXISTS idx_task_history_task_type ON task_history(task_type);
CREATE INDEX IF NOT EXISTS idx_task_history_created_at ON task_history(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_sessions_agent_id ON memory_sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_sessions_session_type ON memory_sessions(session_type);
CREATE INDEX IF NOT EXISTS idx_memory_sessions_expires_at ON memory_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_pheromone_source_target ON pheromone_trails(source_agent, target_agent);
CREATE INDEX IF NOT EXISTS idx_pheromone_trail_type ON pheromone_trails(trail_type);
CREATE INDEX IF NOT EXISTS idx_pheromone_strength ON pheromone_trails(strength);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_name ON mcp_tools(name);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_agent_id ON mcp_tools(agent_id);
CREATE INDEX IF NOT EXISTS idx_mcp_resources_uri ON mcp_resources(uri);
CREATE INDEX IF NOT EXISTS idx_mcp_resources_type ON mcp_resources(resource_type);

-- Create triggers for automatic timestamp updates
CREATE TRIGGER IF NOT EXISTS update_agents_timestamp 
    AFTER UPDATE ON agents 
    BEGIN 
        UPDATE agents SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS update_knowledge_timestamp 
    AFTER UPDATE ON knowledge_entries 
    BEGIN 
        UPDATE knowledge_entries SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS update_memory_sessions_timestamp 
    AFTER UPDATE ON memory_sessions 
    BEGIN 
        UPDATE memory_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS update_pheromone_timestamp 
    AFTER UPDATE ON pheromone_trails 
    BEGIN 
        UPDATE pheromone_trails SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS update_mcp_tools_timestamp 
    AFTER UPDATE ON mcp_tools 
    BEGIN 
        UPDATE mcp_tools SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS update_mcp_resources_timestamp 
    AFTER UPDATE ON mcp_resources 
    BEGIN 
        UPDATE mcp_resources SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

-- Create views for common queries
CREATE VIEW IF NOT EXISTS active_agents AS
SELECT 
    id, name, capabilities, current_load, success_rate, last_heartbeat
FROM agents 
WHERE status = 'active' 
    AND (last_heartbeat IS NULL OR last_heartbeat > datetime('now', '-5 minutes'));

CREATE VIEW IF NOT EXISTS agent_performance AS
SELECT 
    a.id,
    a.name,
    a.success_rate as agent_success_rate,
    COUNT(th.id) as total_tasks,
    SUM(CASE WHEN th.success = 1 THEN 1 ELSE 0 END) as successful_tasks,
    AVG(th.execution_time) as avg_execution_time,
    MAX(th.created_at) as last_task_time
FROM agents a
LEFT JOIN task_history th ON a.id = th.agent_id
GROUP BY a.id, a.name, a.success_rate;

CREATE VIEW IF NOT EXISTS pheromone_strengths AS
SELECT 
    source_agent,
    target_agent,
    trail_type,
    strength,
    CASE 
        WHEN (success_count + failure_count) > 0 
        THEN CAST(success_count AS REAL) / (success_count + failure_count)
        ELSE 1.0
    END as computed_success_rate,
    last_reinforcement
FROM pheromone_trails
WHERE strength > 0.1  -- Only show significant trails
ORDER BY strength DESC;

-- Insert initial system data
INSERT OR IGNORE INTO swarm_state (id, coordination_mode, active_agents) 
VALUES ('main_swarm', 'queen_led', 0);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO schema_version (version) VALUES ('1.0.0');