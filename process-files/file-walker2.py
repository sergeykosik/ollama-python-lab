import os
from pathlib import Path
from typing import Generator, Union, Dict, List
import fnmatch

def walk_files(
    root_path: Union[str, Path], 
    include_patterns: List[str] = None, 
    exclude_patterns: List[str] = None
) -> Generator[Dict[str, str], None, None]:
    """
    Walk through all text files in a directory and its subdirectories with filtering options.
    
    Args:
        root_path: Starting directory path as string or Path object
        include_patterns: List of patterns to include (e.g., ['*.cshtml', '*.cs'])
        exclude_patterns: List of patterns to exclude (e.g., ['*.bin', '*.exe'])
    
    Yields:
        Dictionary containing file information:
            - path: Full path to the file
            - name: File name
            - content: File content as string
            - size: File size in bytes
    """
    root_path = Path(root_path)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")
    
    print(f"Starting directory walk from: {root_path.absolute()}")
    
    def should_process_file(filename: str) -> bool:
        # If no include patterns are specified, include all files
        included = not include_patterns or any(fnmatch.fnmatch(filename, pattern) for pattern in include_patterns)
        # If no exclude patterns are specified, exclude no files
        excluded = exclude_patterns and any(fnmatch.fnmatch(filename, pattern) for pattern in exclude_patterns)
        return included and not excluded
    
    for path in root_path.rglob('*'):
        if path.is_file() and should_process_file(path.name):
            try:
                # Try reading as text
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                yield {
                    'path': str(path),
                    'name': path.name,
                    'content': content,
                    'size': path.stat().st_size
                }
            except UnicodeDecodeError:
                # Skip binary file silently
                continue
            except Exception as e:
                print(f"Error reading file {path}: {str(e)}")
                continue

# Example usage
directory_path = Path('/workspaces/AccountsPrep/systems/AccountsPrep/src/AccountsPrep.WebUI/Views')

# Define file patterns
include_patterns = ['BankAccounts.cshtml', """ '*.cshtml' """]  # Include only .cshtml
exclude_patterns = ['*.cs']    # Exclude non .cshtml files

print(f"Checking path: {directory_path}")
print(f"Path exists: {directory_path.exists()}")

try:
    print(f"\nLooking for files matching: {include_patterns}")
    print(f"Excluding files matching: {exclude_patterns}")
    
    for file_info in walk_files(directory_path, include_patterns, exclude_patterns):
        print(f"\nFile: {file_info['name']}")
        print(f"Size: {file_info['size']} bytes")
        # Print first 50 characters of content
        preview = file_info['content'][:50] + "..." if len(file_info['content']) > 50 else file_info['content']
        print(f"Preview: {preview}")
except FileNotFoundError as e:
    print(f"\nError: {e}")
    print("\nLet's check the actual directory structure:")
    
    # Check /workspaces directory
    workspace_dir = Path('/workspaces')
    if workspace_dir.exists():
        print("\nContents of /workspaces:")
        for item in workspace_dir.iterdir():
            print(f"- {item}")

# You can also use it with different patterns
# Examples:
"""
# Find all Razor views
files = walk_files(directory_path, include_patterns=['*.cshtml'])

# Find all C# files except tests
files = walk_files(directory_path, 
                  include_patterns=['*.cs'], 
                  exclude_patterns=['*.Test.cs', '*Tests.cs'])

# Find specific files
files = walk_files(directory_path, 
                  include_patterns=['Layout.cshtml', 'HomeController.cs'])

# Exclude specific directories by excluding their files
files = walk_files(directory_path, 
                  exclude_patterns=['obj/*', 'bin/*', 'node_modules/*'])
"""