"""
Parquet file I/O utilities.

Provides consistent interface for reading/writing parquet files using pandas/polars.
"""

import pandas as pd
import polars as pl
from pathlib import Path
from typing import Union, Optional, Literal, Any, Dict
import warnings


def read_parquet(
    path: Union[str, Path],
    engine: Literal['pandas', 'polars'] = 'pandas',
    columns: Optional[list[str]] = None,
    **kwargs
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Read parquet file using pandas or polars.
    
    Parameters
    ----------
    path : str or Path
        Path to parquet file
    engine : str
        'pandas' or 'polars'
    columns : list of str, optional
        Columns to read (for optimization)
    **kwargs
        Additional arguments passed to read_parquet
        
    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Loaded data
        
    Examples
    --------
    >>> # Read with pandas (default)
    >>> df = read_parquet('data.parquet')
    
    >>> # Read specific columns with polars
    >>> df = read_parquet('data.parquet', engine='polars', columns=['id', 'text'])
    
    >>> # Read with custom options
    >>> df = read_parquet('data.parquet', use_nullable_dtypes=True)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if engine == 'pandas':
        return pd.read_parquet(path, columns=columns, **kwargs)
    elif engine == 'polars':
        if columns:
            return pl.read_parquet(path, columns=columns, **kwargs)
        else:
            return pl.read_parquet(path, **kwargs)
    else:
        raise ValueError(f"Unknown engine: {engine}. Use 'pandas' or 'polars'")


def write_parquet(
    df: Union[pd.DataFrame, pl.DataFrame],
    path: Union[str, Path],
    compression: str = 'snappy',
    index: bool = False,
    **kwargs
) -> None:
    """
    Write DataFrame to parquet file.
    
    Automatically detects pandas vs polars DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Data to write
    path : str or Path
        Output file path
    compression : str
        Compression algorithm: 'snappy', 'gzip', 'brotli', 'zstd'
        Default: 'snappy' (good balance of speed/compression)
    index : bool
        Write index (pandas only, default: False)
    **kwargs
        Additional arguments passed to to_parquet
        
    Examples
    --------
    >>> # Write pandas DataFrame
    >>> write_parquet(df, 'output.parquet')
    
    >>> # Write with gzip compression
    >>> write_parquet(df, 'output.parquet', compression='gzip')
    
    >>> # Write polars DataFrame
    >>> write_parquet(pl_df, 'output.parquet')
    """
    path = Path(path)
    
    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(df, pd.DataFrame):
        df.to_parquet(path, compression=compression, index=index, **kwargs)
    elif isinstance(df, pl.DataFrame):
        df.write_parquet(path, compression=compression, **kwargs)
    else:
        raise TypeError(f"Unknown DataFrame type: {type(df)}. Use pandas or polars DataFrame")
    
    # Verify write
    if not path.exists():
        raise IOError(f"Failed to write file: {path}")
    
    # Print confirmation
    file_size_mb = path.stat().st_size / (1024**2)
    print(f"✓ Wrote {len(df):,} rows to {path.name} ({file_size_mb:.1f} MB)")


def read_parquet_metadata(
    path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Read parquet file metadata without loading full data.
    
    Parameters
    ----------
    path : str or Path
        Path to parquet file
        
    Returns
    -------
    dict
        Metadata including schema, row count, file size
        
    Examples
    --------
    >>> meta = read_parquet_metadata('data.parquet')
    >>> print(f"Rows: {meta['num_rows']}")
    >>> print(f"Columns: {meta['columns']}")
    """
    import pyarrow.parquet as pq
    
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Read parquet metadata
    parquet_file = pq.ParquetFile(path)
    
    metadata = {
        'path': str(path),
        'file_size_mb': path.stat().st_size / (1024**2),
        'num_rows': parquet_file.metadata.num_rows,
        'num_row_groups': parquet_file.metadata.num_row_groups,
        'columns': parquet_file.schema.names,
        'schema': str(parquet_file.schema),
        'created_by': parquet_file.metadata.created_by
    }
    
    return metadata


def optimize_parquet(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    compression: str = 'zstd',
    row_group_size: Optional[int] = None
) -> None:
    """
    Re-write parquet file with optimized compression and row groups.
    
    Useful for reducing file size or improving read performance.
    
    Parameters
    ----------
    input_path : str or Path
        Input parquet file
    output_path : str or Path, optional
        Output path (default: overwrite input)
    compression : str
        Compression algorithm (default: 'zstd' for better compression)
    row_group_size : int, optional
        Rows per row group (default: pyarrow default ~64MB)
        
    Examples
    --------
    >>> # Re-compress with better compression
    >>> optimize_parquet('data.parquet', compression='zstd')
    
    >>> # Save to new file
    >>> optimize_parquet('data.parquet', 'data_optimized.parquet')
    """
    import pyarrow.parquet as pq
    
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
    
    print(f"Optimizing {input_path.name}...")
    
    # Read
    table = pq.read_table(input_path)
    
    # Write with optimization
    pq.write_table(
        table,
        output_path,
        compression=compression,
        row_group_size=row_group_size
    )
    
    # Show size change
    old_size = input_path.stat().st_size / (1024**2)
    new_size = output_path.stat().st_size / (1024**2)
    reduction = 100 * (1 - new_size / old_size)
    
    print(f"✓ Optimized: {old_size:.1f} MB → {new_size:.1f} MB ({reduction:.1f}% reduction)")


# Example usage
if __name__ == '__main__':
    print("Parquet I/O Test")
    print("=" * 60)
    
    # Test 1: Write and read with pandas
    print("\nTest 1: Pandas write/read")
    df_pandas = pd.DataFrame({
        'id': range(1000),
        'text': [f'sample_{i}' for i in range(1000)],
        'value': range(1000, 2000)
    })
    
    test_file = Path('test_output.parquet')
    write_parquet(df_pandas, test_file)
    df_read = read_parquet(test_file, engine='pandas')
    
    print(f"Original: {len(df_pandas)} rows")
    print(f"Read back: {len(df_read)} rows")
    assert len(df_pandas) == len(df_read), "Row count mismatch!"
    
    # Test 2: Read with polars
    print("\nTest 2: Polars read")
    df_polars = read_parquet(test_file, engine='polars')
    print(f"Polars read: {len(df_polars)} rows")
    
    # Test 3: Read metadata
    print("\nTest 3: Read metadata")
    meta = read_parquet_metadata(test_file)
    print(f"Metadata:")
    print(f"  Rows: {meta['num_rows']:,}")
    print(f"  Columns: {meta['columns']}")
    print(f"  File size: {meta['file_size_mb']:.2f} MB")
    
    # Test 4: Column selection
    print("\nTest 4: Column selection")
    df_subset = read_parquet(test_file, columns=['id', 'text'])
    print(f"Selected columns: {df_subset.columns.tolist()}")
    
    # Cleanup
    test_file.unlink()
    print("\n✓ All tests passed!")
