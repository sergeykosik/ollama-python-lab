"""
Utility Functions
Helper functions for file operations and data management
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


def load_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load multiple files and return their data

    Args:
        file_paths: List of file paths to load

    Returns:
        List of file data dictionaries
    """
    files_data = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            files_data.append({
                'name': os.path.basename(file_path),
                'path': file_path,
                'content': content,
                'language': Path(file_path).suffix[1:],
                'size': len(content),
                'lines': len(content.splitlines())
            })
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    return files_data


def save_review_history(review_data: Dict[str, Any], history_file: str = ".review_history.json"):
    """
    Save review to history file

    Args:
        review_data: Review data to save
        history_file: Path to history file
    """
    history = []

    # Load existing history
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []

    # Add new review
    history.append(review_data)

    # Keep last 100 reviews
    history = history[-100:]

    # Save updated history
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")


def load_review_history(history_file: str = ".review_history.json") -> List[Dict[str, Any]]:
    """
    Load review history from file

    Args:
        history_file: Path to history file

    Returns:
        List of review data
    """
    if not os.path.exists(history_file):
        return []

    try:
        with open(history_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading history: {e}")
        return []


def format_timestamp(iso_timestamp: str) -> str:
    """
    Format ISO timestamp to readable format

    Args:
        iso_timestamp: ISO format timestamp

    Returns:
        Formatted timestamp string
    """
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_timestamp


def get_language_from_extension(extension: str) -> str:
    """
    Get programming language name from file extension

    Args:
        extension: File extension (with or without dot)

    Returns:
        Language name
    """
    extension = extension.lstrip('.')

    language_map = {
        'py': 'python',
        'js': 'javascript',
        'ts': 'typescript',
        'jsx': 'javascript',
        'tsx': 'typescript',
        'java': 'java',
        'cpp': 'cpp',
        'c': 'c',
        'h': 'c',
        'hpp': 'cpp',
        'go': 'go',
        'rs': 'rust',
        'rb': 'ruby',
        'php': 'php',
        'cs': 'csharp',
        'swift': 'swift',
        'kt': 'kotlin',
        'scala': 'scala',
        'r': 'r',
        'sql': 'sql',
        'sh': 'bash',
        'yaml': 'yaml',
        'yml': 'yaml',
        'json': 'json',
        'xml': 'xml',
        'html': 'html',
        'css': 'css',
        'scss': 'scss',
        'sass': 'sass',
        'md': 'markdown'
    }

    return language_map.get(extension.lower(), extension)


def count_lines_of_code(content: str, language: str = 'python') -> Dict[str, int]:
    """
    Count lines of code, comments, and blank lines

    Args:
        content: File content
        language: Programming language

    Returns:
        Dictionary with line counts
    """
    lines = content.split('\n')
    total_lines = len(lines)
    blank_lines = 0
    comment_lines = 0
    code_lines = 0

    # Define comment patterns by language
    comment_patterns = {
        'python': ['#'],
        'javascript': ['//', '/*', '*'],
        'typescript': ['//', '/*', '*'],
        'java': ['//', '/*', '*'],
        'cpp': ['//', '/*', '*'],
        'c': ['//', '/*', '*'],
        'go': ['//', '/*', '*'],
        'rust': ['//', '/*', '*'],
        'ruby': ['#'],
        'php': ['//', '#', '/*', '*'],
    }

    patterns = comment_patterns.get(language, ['#', '//', '/*', '*'])

    for line in lines:
        stripped = line.strip()

        if not stripped:
            blank_lines += 1
        elif any(stripped.startswith(pattern) for pattern in patterns):
            comment_lines += 1
        else:
            code_lines += 1

    return {
        'total': total_lines,
        'code': code_lines,
        'comments': comment_lines,
        'blank': blank_lines
    }


def extract_functions_and_classes(content: str, language: str = 'python') -> Dict[str, List[str]]:
    """
    Extract function and class names from code

    Args:
        content: File content
        language: Programming language

    Returns:
        Dictionary with functions and classes lists
    """
    import re

    functions = []
    classes = []

    if language == 'python':
        # Find class definitions
        class_pattern = r'^\s*class\s+(\w+)'
        classes = re.findall(class_pattern, content, re.MULTILINE)

        # Find function definitions
        func_pattern = r'^\s*def\s+(\w+)'
        functions = re.findall(func_pattern, content, re.MULTILINE)

    elif language in ['javascript', 'typescript']:
        # Find class definitions
        class_pattern = r'class\s+(\w+)'
        classes = re.findall(class_pattern, content)

        # Find function definitions
        func_patterns = [
            r'function\s+(\w+)',
            r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>',
            r'(\w+)\s*:\s*function',
        ]
        for pattern in func_patterns:
            functions.extend(re.findall(pattern, content))

    elif language == 'java':
        # Find class definitions
        class_pattern = r'class\s+(\w+)'
        classes = re.findall(class_pattern, content)

        # Find method definitions
        func_pattern = r'(?:public|private|protected)\s+\w+\s+(\w+)\s*\('
        functions = re.findall(func_pattern, content)

    return {
        'functions': list(set(functions)),  # Remove duplicates
        'classes': list(set(classes))
    }


def generate_file_summary(file_data: Dict[str, Any]) -> str:
    """
    Generate a summary of a file

    Args:
        file_data: File data dictionary

    Returns:
        Summary string
    """
    content = file_data.get('content', '')
    language = file_data.get('language', 'unknown')

    line_counts = count_lines_of_code(content, language)
    entities = extract_functions_and_classes(content, language)

    summary = f"File: {file_data['name']}\n"
    summary += f"Language: {language}\n"
    summary += f"Total Lines: {line_counts['total']} "
    summary += f"(Code: {line_counts['code']}, Comments: {line_counts['comments']}, Blank: {line_counts['blank']})\n"

    if entities['classes']:
        summary += f"Classes: {', '.join(entities['classes'][:5])}"
        if len(entities['classes']) > 5:
            summary += f" and {len(entities['classes']) - 5} more"
        summary += "\n"

    if entities['functions']:
        summary += f"Functions: {', '.join(entities['functions'][:5])}"
        if len(entities['functions']) > 5:
            summary += f" and {len(entities['functions']) - 5} more"
        summary += "\n"

    return summary


def validate_code_file(file_path: str) -> bool:
    """
    Validate if a file is a valid code file

    Args:
        file_path: Path to file

    Returns:
        True if valid code file
    """
    # Check extension
    valid_extensions = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
        '.go', '.rs', '.rb', '.php', '.cs', '.swift', '.kt', '.scala'
    }

    ext = Path(file_path).suffix.lower()
    if ext not in valid_extensions:
        return False

    # Check file size (< 5MB)
    try:
        file_size = os.path.getsize(file_path)
        if file_size > 5 * 1024 * 1024:
            return False
    except Exception:
        return False

    return True


def truncate_content(content: str, max_length: int = 1000) -> str:
    """
    Truncate content to max length with ellipsis

    Args:
        content: Content to truncate
        max_length: Maximum length

    Returns:
        Truncated content
    """
    if len(content) <= max_length:
        return content

    return content[:max_length] + "..."
