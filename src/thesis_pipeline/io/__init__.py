"""
I/O utilities for thesis data pipeline.

Provides consistent interface for:
- Path management
- Parquet file reading/writing
- Data validation
"""

from .paths import get_data_path, get_project_root
from .parquet import read_parquet, write_parquet

__all__ = [
    'get_data_path',
    'get_project_root',
    'read_parquet',
    'write_parquet'
]
