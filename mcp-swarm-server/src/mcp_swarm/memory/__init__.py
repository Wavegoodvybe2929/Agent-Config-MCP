"""
Memory Management System

This module provides persistent memory management with SQLite backend,
cross-session state persistence, and memory optimization for swarm intelligence.
"""

from .manager import MemoryManager

__all__ = ["MemoryManager"]