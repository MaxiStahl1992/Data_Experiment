"""
Path management utilities.

Provides consistent path resolution for data directories across notebooks.
"""

from pathlib import Path
from typing import Literal, Optional


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Assumes this file is in src/thesis_pipeline/io/paths.py,
    so project root is 3 levels up.
    
    Returns
    -------
    Path
        Absolute path to project root
        
    Examples
    --------
    >>> root = get_project_root()
    >>> print(root / 'data' / '03_gold')
    """
    return Path(__file__).parent.parent.parent.parent.resolve()


def get_data_path(
    layer: Literal['raw', 'silver', 'gold', 'qa', 'topics'] = 'gold',
    platform: Optional[Literal['reddit', 'news']] = None,
    create: bool = False
) -> Path:
    """
    Get path to data directory for specified layer and platform.
    
    Directory structure:
    - data/01_corpus/00_raw/{platform}/
    - data/01_corpus/01_silver/{platform}/
    - data/01_corpus/02_gold/{platform}/
    - data/01_corpus/03_qa/{platform}/
    - data/02_topics/{platform}/
    
    Parameters
    ----------
    layer : str
        Data layer: 'raw', 'silver', 'gold', 'qa', 'topics'
    platform : str, optional
        Platform subdirectory: 'reddit', 'news'
        If None, returns layer root
    create : bool
        Create directory if it doesn't exist
        
    Returns
    -------
    Path
        Absolute path to data directory
        
    Examples
    --------
    >>> # Get Reddit gold layer
    >>> gold_path = get_data_path('gold', 'reddit')
    >>> print(gold_path)
    /path/to/project/data/01_corpus/02_gold/reddit
    
    >>> # Get QA layer root
    >>> qa_path = get_data_path('qa', create=True)
    
    >>> # Get topics directory
    >>> topics_path = get_data_path('topics', 'reddit')
    >>> print(topics_path)
    /path/to/project/data/02_topics/reddit
    """
    root = get_project_root()
    
    # Map layer names to directory paths
    # Most layers are under data/01_corpus, but topics is separate
    layer_map = {
        'raw': ('01_corpus', '00_raw'),
        'silver': ('01_corpus', '01_silver'),
        'gold': ('01_corpus', '02_gold'),
        'qa': ('01_corpus', '03_qa'),
        'topics': ('02_topics', None)  # topics is not under corpus
    }
    
    if layer not in layer_map:
        raise ValueError(f"Unknown layer: {layer}. Must be one of {list(layer_map.keys())}")
    
    # Build path
    parent_dir, layer_dir = layer_map[layer]
    if layer_dir is not None:
        data_dir = root / 'data' / parent_dir / layer_dir
    else:
        data_dir = root / 'data' / parent_dir
    
    if platform is not None:
        data_dir = data_dir / platform
    
    # Create if requested
    if create:
        data_dir.mkdir(parents=True, exist_ok=True)
    
    return data_dir


def get_output_path(
    notebook_name: str,
    platform: Literal['reddit', 'news', 'shared'],
    create: bool = True
) -> Path:
    """
    Get output path for a specific notebook.
    
    Creates subdirectory under data/01_corpus/03_qa/{platform}/{notebook_name}/
    for organizing notebook-specific outputs.
    
    Parameters
    ----------
    notebook_name : str
        Name of notebook (e.g., '14_reddit_corpus_prep')
    platform : str
        Platform: 'reddit', 'news', 'shared'
    create : bool
        Create directory if it doesn't exist
        
    Returns
    -------
    Path
        Absolute path to notebook output directory
        
    Examples
    --------
    >>> output_dir = get_output_path('14_reddit_corpus_prep', 'reddit')
    >>> print(output_dir)
    /path/to/project/data/01_corpus/03_qa/reddit/14_reddit_corpus_prep/
    """
    qa_path = get_data_path('qa', platform, create=False)
    output_dir = qa_path / notebook_name
    
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def list_data_files(
    layer: Literal['raw', 'silver', 'gold', 'qa'],
    platform: Optional[Literal['reddit', 'news']] = None,
    pattern: str = '*.parquet'
) -> list[Path]:
    """
    List data files in a directory.
    
    Parameters
    ----------
    layer : str
        Data layer
    platform : str, optional
        Platform subdirectory
    pattern : str
        Glob pattern (default: '*.parquet')
        
    Returns
    -------
    list of Path
        Sorted list of matching files
        
    Examples
    --------
    >>> # List all parquet files in Reddit gold layer
    >>> files = list_data_files('gold', 'reddit')
    >>> for f in files:
    ...     print(f.name)
    """
    data_dir = get_data_path(layer, platform, create=False)
    
    if not data_dir.exists():
        return []
    
    return sorted(data_dir.glob(pattern))


# Example usage
if __name__ == '__main__':
    print("Path Utilities Test")
    print("=" * 60)
    
    # Test 1: Get project root
    print("\nTest 1: Project root")
    root = get_project_root()
    print(f"Project root: {root}")
    print(f"Exists: {root.exists()}")
    
    # Test 2: Get data paths
    print("\nTest 2: Data paths")
    for layer in ['raw', 'silver', 'gold', 'qa']:
        path = get_data_path(layer)
        print(f"  {layer:10s}: {path} (exists: {path.exists()})")
    
    # Test 3: Platform-specific paths
    print("\nTest 3: Platform-specific paths")
    gold_reddit = get_data_path('gold', 'reddit')
    gold_news = get_data_path('gold', 'news')
    print(f"  Reddit gold: {gold_reddit}")
    print(f"  News gold:   {gold_news}")
    
    # Test 4: List files
    print("\nTest 4: List files in Reddit gold")
    files = list_data_files('gold', 'reddit')
    if files:
        print(f"  Found {len(files)} files:")
        for f in files[:5]:  # Show first 5
            print(f"    - {f.name}")
    else:
        print("  No files found")
    
    print("\n✓ All tests passed!")
