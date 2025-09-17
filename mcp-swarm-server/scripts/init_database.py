#!/usr/bin/env python3
"""
Database initialization script for MCP Swarm Intelligence Server.
This script creates the SQLite database and applies the schema.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import aiosqlite


class DatabaseInitializationError(Exception):
    """Raised when database initialization fails."""


class DatabaseVerificationError(Exception):
    """Raised when database verification fails."""


class SchemaValidationError(Exception):
    """Raised when database schema validation fails."""

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Initialize and manage the SQLite database for MCP Swarm Server."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the database initializer.

        Args:
            db_path: Path to the SQLite database file. Defaults to src/data/memory.db
        """
        if db_path is None:
            # Default to src/data/memory.db
            project_root = Path(__file__).parent.parent
            db_path = project_root / "src" / "data" / "memory.db"

        self.db_path = db_path
        self.schema_path = Path(__file__).parent.parent / "src" / "data" / "schema.sql"

    async def initialize_database(self) -> bool:
        """Initialize the database with the schema.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Ensure the data directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Read the schema file
            if not self.schema_path.exists():
                logger.error("Schema file not found: %s", self.schema_path)
                return False

            with open(self.schema_path, "r", encoding="utf-8") as f:
                schema_sql = f.read()

            # Initialize the database
            async with aiosqlite.connect(self.db_path) as db:
                logger.info("Creating database at: %s", self.db_path)

                # Execute the schema
                await db.executescript(schema_sql)
                await db.commit()

                # Verify the database was created correctly
                await self._verify_database(db)

                logger.info("Database initialization completed successfully")
                return True

        except (OSError, aiosqlite.Error, SchemaValidationError, DatabaseVerificationError) as e:
            logger.error("Database initialization failed: %s", e)
            return False

    async def _verify_database(self, db: aiosqlite.Connection) -> None:
        """Verify that the database was created correctly.

        Args:
            db: Database connection to verify
        """
        # Check that all required tables exist
        required_tables = [
            "agents",
            "knowledge_entries",
            "swarm_state",
            "task_history",
            "memory_sessions",
            "pheromone_trails",
            "mcp_tools",
            "mcp_resources",
            "schema_version",
        ]

        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in await cursor.fetchall()]

        missing_tables = set(required_tables) - set(existing_tables)
        if missing_tables:
            raise SchemaValidationError(f"Missing required tables: {missing_tables}")

        # Check that views exist
        required_views = ["active_agents", "agent_performance", "pheromone_strengths"]
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='view'")
        existing_views = [row[0] for row in await cursor.fetchall()]

        missing_views = set(required_views) - set(existing_views)
        if missing_views:
            raise SchemaValidationError(f"Missing required views: {missing_views}")

        # Verify initial data
        cursor = await db.execute(
            "SELECT COUNT(*) FROM swarm_state WHERE id = 'main_swarm'"
        )
        result = await cursor.fetchone()
        if result is None:
            raise DatabaseVerificationError("Unable to fetch swarm state count")
        swarm_count = result[0]
        if swarm_count == 0:
            raise SchemaValidationError("Initial swarm state not created")

        logger.info("Database verification completed successfully")

    async def reset_database(self) -> bool:
        """Reset the database by removing and recreating it.

        Returns:
            True if reset successful, False otherwise
        """
        try:
            if self.db_path.exists():
                self.db_path.unlink()
                logger.info("Removed existing database: %s", self.db_path)

            return await self.initialize_database()

        except (OSError, PermissionError) as e:
            logger.error("Database reset failed: %s", e)
            return False


async def main():
    """Main function to initialize the database."""
    initializer = DatabaseInitializer()

    # Initialize the database
    success = await initializer.initialize_database()

    if success:
        print("✅ Database initialization completed successfully!")
        print(f"📁 Database location: {initializer.db_path.absolute()}")
    else:
        print("❌ Database initialization failed!")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
