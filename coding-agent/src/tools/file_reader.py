"""File reading tools for the agent."""

import json
import logging
from pathlib import Path
from typing import Optional
from langchain.tools import BaseTool
from pydantic import Field

from ..utils.file_utils import read_file_safely, get_file_metadata

logger = logging.getLogger(__name__)


class ReadFileTool(BaseTool):
    """Tool for reading file contents."""

    name: str = "read_file"
    description: str = """
    Reads the contents of a file.
    Input should be a file path.
    Returns the file contents as text.

    Example input: "/path/to/file.py"
    """

    def _run(self, file_path: str) -> str:
        """Read a file.

        Args:
            file_path: Path to the file

        Returns:
            File contents or error message
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return f"Error: File not found: {file_path}"

            if not path.is_file():
                return f"Error: Not a file: {file_path}"

            content = read_file_safely(path)
            if content is None:
                return f"Error: Could not read file: {file_path}"

            return content

        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return f"Error: {e}"

    async def _arun(self, file_path: str) -> str:
        """Async version."""
        return self._run(file_path)


class ReadFileLinesTool(BaseTool):
    """Tool for reading specific lines from a file."""

    name: str = "read_file_lines"
    description: str = """
    Reads specific lines from a file.
    Input should be JSON: {"file_path": "/path/to/file.py", "start_line": 10, "end_line": 20}
    Returns the specified lines.
    """

    def _run(self, input_str: str) -> str:
        """Read specific lines from a file.

        Args:
            input_str: JSON with file_path, start_line, end_line

        Returns:
            File lines or error message
        """
        try:
            input_data = json.loads(input_str)
            file_path = input_data['file_path']
            start_line = input_data.get('start_line', 1)
            end_line = input_data.get('end_line')

            path = Path(file_path)
            if not path.exists():
                return f"Error: File not found: {file_path}"

            content = read_file_safely(path)
            if content is None:
                return f"Error: Could not read file: {file_path}"

            lines = content.splitlines()

            # Adjust for 1-based indexing
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line) if end_line else len(lines)

            selected_lines = lines[start_idx:end_idx]

            # Format with line numbers
            result_lines = [
                f"{i + start_line}: {line}"
                for i, line in enumerate(selected_lines)
            ]

            return '\n'.join(result_lines)

        except json.JSONDecodeError:
            return "Error: Invalid JSON input"
        except Exception as e:
            logger.error(f"Error reading file lines: {e}")
            return f"Error: {e}"

    async def _arun(self, input_str: str) -> str:
        """Async version."""
        return self._run(input_str)


class FileMetadataTool(BaseTool):
    """Tool for getting file metadata."""

    name: str = "file_metadata"
    description: str = """
    Gets metadata about a file (size, type, lines, last modified, etc.).
    Input should be a file path.
    Returns JSON with file metadata.
    """

    def _run(self, file_path: str) -> str:
        """Get file metadata.

        Args:
            file_path: Path to the file

        Returns:
            JSON with metadata or error message
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return f"Error: File not found: {file_path}"

            metadata = get_file_metadata(path)
            if not metadata:
                return f"Error: Could not get metadata for: {file_path}"

            result = {
                'path': metadata.path,
                'type': metadata.file_type.value,
                'language': metadata.language,
                'size_bytes': metadata.size_bytes,
                'size_kb': round(metadata.size_bytes / 1024, 2),
                'lines': metadata.lines,
                'last_modified': metadata.last_modified.isoformat(),
                'hash': metadata.file_hash
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting file metadata: {e}")
            return f"Error: {e}"

    async def _arun(self, file_path: str) -> str:
        """Async version."""
        return self._run(file_path)


class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents."""

    name: str = "list_directory"
    description: str = """
    Lists the contents of a directory.
    Input should be a directory path.
    Returns list of files and directories.

    Optional: Can provide JSON with {"path": "/dir", "pattern": "*.py"} to filter files.
    """

    def _run(self, input_str: str) -> str:
        """List directory contents.

        Args:
            input_str: Directory path or JSON with path and pattern

        Returns:
            Directory listing or error message
        """
        try:
            # Try parsing as JSON first
            try:
                input_data = json.loads(input_str)
                dir_path = input_data['path']
                pattern = input_data.get('pattern', '*')
            except (json.JSONDecodeError, KeyError):
                # Treat as plain path
                dir_path = input_str
                pattern = '*'

            path = Path(dir_path)
            if not path.exists():
                return f"Error: Directory not found: {dir_path}"

            if not path.is_dir():
                return f"Error: Not a directory: {dir_path}"

            # List contents
            items = {
                'directories': [],
                'files': []
            }

            for item in sorted(path.glob(pattern)):
                if item.is_dir():
                    items['directories'].append(str(item.name))
                else:
                    items['files'].append(str(item.name))

            result = {
                'path': str(path),
                'total_directories': len(items['directories']),
                'total_files': len(items['files']),
                'directories': items['directories'][:50],  # Limit to 50
                'files': items['files'][:50]  # Limit to 50
            }

            if len(items['directories']) > 50:
                result['note'] = f"Showing first 50 of {len(items['directories'])} directories"
            if len(items['files']) > 50:
                result['note'] = f"Showing first 50 of {len(items['files'])} files"

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error listing directory: {e}")
            return f"Error: {e}"

    async def _arun(self, input_str: str) -> str:
        """Async version."""
        return self._run(input_str)


class SearchFilesTool(BaseTool):
    """Tool for searching files by name pattern."""

    name: str = "search_files"
    description: str = """
    Searches for files matching a pattern in a directory tree.
    Input should be JSON: {"directory": "/path/to/dir", "pattern": "*.py"}
    Returns list of matching file paths.

    Example: {"directory": "./src", "pattern": "*test*.py"}
    """

    def _run(self, input_str: str) -> str:
        """Search for files.

        Args:
            input_str: JSON with directory and pattern

        Returns:
            List of matching files or error message
        """
        try:
            input_data = json.loads(input_str)
            directory = input_data['directory']
            pattern = input_data['pattern']

            path = Path(directory)
            if not path.exists():
                return f"Error: Directory not found: {directory}"

            if not path.is_dir():
                return f"Error: Not a directory: {directory}"

            # Search recursively
            matches = list(path.rglob(pattern))

            # Limit results
            max_results = 100
            file_matches = [str(f) for f in matches if f.is_file()][:max_results]

            result = {
                'directory': str(path),
                'pattern': pattern,
                'total_matches': len(file_matches),
                'files': file_matches
            }

            if len(matches) > max_results:
                result['note'] = f"Showing first {max_results} results"

            return json.dumps(result, indent=2)

        except json.JSONDecodeError:
            return "Error: Invalid JSON input"
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return f"Error: {e}"

    async def _arun(self, input_str: str) -> str:
        """Async version."""
        return self._run(input_str)
