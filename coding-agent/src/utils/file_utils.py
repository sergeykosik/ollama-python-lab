"""File utility functions for the coding agent."""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Set, Iterator
import gitignore_parser
from ..models.schemas import FileType, FileMetadata
from datetime import datetime

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        MD5 hash as hexadecimal string
    """
    md5_hash = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {file_path}: {e}")
        return ""


def get_file_metadata(file_path: Path) -> Optional[FileMetadata]:
    """Extract metadata from a file.

    Args:
        file_path: Path to the file

    Returns:
        FileMetadata object or None if error occurs
    """
    try:
        stat = file_path.stat()
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        lines = len(content.splitlines())

        return FileMetadata(
            path=str(file_path),
            file_type=FileType.from_extension(file_path.suffix),
            size_bytes=stat.st_size,
            lines=lines,
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            file_hash=compute_file_hash(file_path),
            language=get_language_from_extension(file_path.suffix),
            encoding='utf-8'
        )
    except Exception as e:
        logger.error(f"Error extracting metadata from {file_path}: {e}")
        return None


def get_language_from_extension(extension: str) -> Optional[str]:
    """Get programming language from file extension.

    Args:
        extension: File extension (with or without leading dot)

    Returns:
        Language name or None
    """
    extension = extension.lower().lstrip('.')
    mapping = {
        'py': 'Python',
        'js': 'JavaScript',
        'jsx': 'JavaScript',
        'ts': 'TypeScript',
        'tsx': 'TypeScript',
        'java': 'Java',
        'cs': 'C#',
        'cpp': 'C++',
        'c': 'C',
        'h': 'C/C++',
        'hpp': 'C++',
        'go': 'Go',
        'rs': 'Rust',
        'rb': 'Ruby',
        'php': 'PHP',
        'swift': 'Swift',
        'kt': 'Kotlin',
        'scala': 'Scala',
        'sql': 'SQL',
        'md': 'Markdown',
        'json': 'JSON',
        'yaml': 'YAML',
        'yml': 'YAML',
        'xml': 'XML',
        'html': 'HTML',
        'css': 'CSS',
        'scss': 'SCSS',
        'sass': 'SASS',
    }
    return mapping.get(extension)


def should_ignore_file(
    file_path: Path,
    ignore_patterns: List[str],
    gitignore_path: Optional[Path] = None
) -> bool:
    """Check if a file should be ignored.

    Args:
        file_path: Path to check
        ignore_patterns: List of glob patterns to ignore
        gitignore_path: Optional path to .gitignore file

    Returns:
        True if file should be ignored
    """
    # Check if file is in ignore patterns
    for pattern in ignore_patterns:
        if file_path.match(pattern):
            return True

    # Check gitignore if provided
    if gitignore_path and gitignore_path.exists():
        try:
            matches = gitignore_parser.parse_gitignore(gitignore_path)
            if matches(str(file_path)):
                return True
        except Exception as e:
            logger.warning(f"Error parsing gitignore: {e}")

    return False


def find_files(
    directory: Path,
    file_patterns: List[str],
    ignore_patterns: List[str],
    max_size_mb: int = 10,
    use_gitignore: bool = True
) -> Iterator[Path]:
    """Find files matching patterns in a directory.

    Args:
        directory: Root directory to search
        file_patterns: List of glob patterns to match
        ignore_patterns: List of glob patterns to ignore
        max_size_mb: Maximum file size in MB
        use_gitignore: Whether to respect .gitignore

    Yields:
        Path objects for matching files
    """
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return

    # Find .gitignore
    gitignore_path = None
    if use_gitignore:
        gitignore_path = directory / '.gitignore'
        if not gitignore_path.exists():
            # Try parent directories
            for parent in directory.parents:
                candidate = parent / '.gitignore'
                if candidate.exists():
                    gitignore_path = candidate
                    break

    max_size_bytes = max_size_mb * 1024 * 1024

    for pattern in file_patterns:
        for file_path in directory.rglob(pattern):
            if not file_path.is_file():
                continue

            # Check size
            try:
                if file_path.stat().st_size > max_size_bytes:
                    logger.debug(f"Skipping large file: {file_path}")
                    continue
            except OSError:
                continue

            # Check if should ignore
            if should_ignore_file(file_path, ignore_patterns, gitignore_path):
                logger.debug(f"Ignoring file: {file_path}")
                continue

            yield file_path


def read_file_safely(file_path: Path, encoding: str = 'utf-8') -> Optional[str]:
    """Safely read a file with error handling.

    Args:
        file_path: Path to the file
        encoding: File encoding

    Returns:
        File contents or None if error occurs
    """
    try:
        return file_path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        # Try different encodings
        for alt_encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                logger.info(f"Retrying {file_path} with {alt_encoding}")
                return file_path.read_text(encoding=alt_encoding)
            except UnicodeDecodeError:
                continue
        logger.error(f"Could not decode {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None


def write_file_safely(file_path: Path, content: str, create_dirs: bool = True) -> bool:
    """Safely write content to a file.

    Args:
        file_path: Path to the file
        content: Content to write
        create_dirs: Whether to create parent directories

    Returns:
        True if successful, False otherwise
    """
    try:
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.write_text(content, encoding='utf-8')
        return True
    except Exception as e:
        logger.error(f"Error writing to {file_path}: {e}")
        return False


def get_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """Find the project root directory.

    Looks for common markers like .git, pyproject.toml, package.json, etc.

    Args:
        start_path: Path to start searching from (defaults to cwd)

    Returns:
        Project root path or None
    """
    if start_path is None:
        start_path = Path.cwd()

    markers = [
        '.git',
        'pyproject.toml',
        'setup.py',
        'package.json',
        'pom.xml',
        'build.gradle',
        'Cargo.toml',
        '.project'
    ]

    current = start_path.resolve()
    for parent in [current] + list(current.parents):
        for marker in markers:
            if (parent / marker).exists():
                return parent

    return None


def get_relative_path(file_path: Path, base_path: Optional[Path] = None) -> Path:
    """Get relative path from base path.

    Args:
        file_path: The file path
        base_path: Base path (defaults to project root)

    Returns:
        Relative path
    """
    if base_path is None:
        base_path = get_project_root(file_path)
        if base_path is None:
            base_path = Path.cwd()

    try:
        return file_path.relative_to(base_path)
    except ValueError:
        return file_path


def count_lines_of_code(directory: Path, file_patterns: List[str]) -> dict:
    """Count lines of code in a directory.

    Args:
        directory: Directory to analyze
        file_patterns: File patterns to include

    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_files': 0,
        'total_lines': 0,
        'by_language': {}
    }

    for file_path in find_files(directory, file_patterns, [], max_size_mb=50):
        try:
            content = read_file_safely(file_path)
            if content:
                lines = len(content.splitlines())
                lang = get_language_from_extension(file_path.suffix) or 'Unknown'

                stats['total_files'] += 1
                stats['total_lines'] += lines

                if lang not in stats['by_language']:
                    stats['by_language'][lang] = {'files': 0, 'lines': 0}

                stats['by_language'][lang]['files'] += 1
                stats['by_language'][lang]['lines'] += lines
        except Exception as e:
            logger.debug(f"Error processing {file_path}: {e}")

    return stats
