import os
from pathlib import Path
from typing import Generator, Union, Dict, List
import fnmatch

def walk_files(
    root_path: Union[str, Path], 
    include_patterns: List[str] = None, 
    exclude_patterns: List[str] = None
) -> Generator[Dict[str, str], None, None]:
    root_path = Path(root_path)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")
    
    print(f"Starting directory walk from: {root_path.absolute()}")
    
    def should_process_file(filename: str) -> bool:
        included = not include_patterns or any(fnmatch.fnmatch(filename, pattern) for pattern in include_patterns)
        excluded = exclude_patterns and any(fnmatch.fnmatch(filename, pattern) for pattern in exclude_patterns)
        return included and not excluded
    
    for path in root_path.rglob('*'):
        if path.is_file() and should_process_file(path.name):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                yield {
                    'path': str(path),
                    'name': path.name,
                    'content': content,
                    'size': path.stat().st_size
                }
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading file {path}: {str(e)}")
                continue