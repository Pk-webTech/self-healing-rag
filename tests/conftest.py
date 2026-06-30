"""
tests/conftest.py
Shared pytest fixtures for Phase 1 tests.
"""
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))