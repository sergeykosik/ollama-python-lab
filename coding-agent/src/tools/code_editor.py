"""Code editing and generation tools."""

import json
import logging
from pathlib import Path
from langchain.tools import BaseTool
from pydantic import Field

from ..utils.file_utils import write_file_safely, read_file_safely

logger = logging.getLogger(__name__)


class CodeEditorTool(BaseTool):
    """Tool for editing code files."""

    name: str = "edit_code"
    description: str = """
    Edits or creates code files.
    Input should be JSON with the following structure:
    {
        "file_path": "/path/to/file.py",
        "operation": "create|update|replace",
        "content": "new content",
        "line_start": 10,  # Optional, for update operation
        "line_end": 20,    # Optional, for update operation
        "description": "What this change does"
    }

    Operations:
    - create: Create a new file
    - update: Update specific lines
    - replace: Replace entire file content

    Example:
    {
        "file_path": "src/new_module.py",
        "operation": "create",
        "content": "def hello():\\n    print('Hello')",
        "description": "Create a new hello module"
    }
    """

    def _run(self, input_str: str) -> str:
        """Edit or create a code file.

        Args:
            input_str: JSON with edit specification

        Returns:
            Success message or error
        """
        try:
            input_data = json.loads(input_str)
            file_path = input_data['file_path']
            operation = input_data['operation'].lower()
            content = input_data['content']
            description = input_data.get('description', '')

            path = Path(file_path)

            if operation == 'create':
                if path.exists():
                    return f"Error: File already exists: {file_path}"

                if write_file_safely(path, content):
                    return json.dumps({
                        'success': True,
                        'operation': 'create',
                        'file_path': str(path),
                        'message': f"Created file: {file_path}",
                        'description': description
                    })
                else:
                    return f"Error: Could not create file: {file_path}"

            elif operation == 'replace':
                if write_file_safely(path, content):
                    return json.dumps({
                        'success': True,
                        'operation': 'replace',
                        'file_path': str(path),
                        'message': f"Replaced content in: {file_path}",
                        'description': description
                    })
                else:
                    return f"Error: Could not write to file: {file_path}"

            elif operation == 'update':
                if not path.exists():
                    return f"Error: File not found: {file_path}"

                existing_content = read_file_safely(path)
                if existing_content is None:
                    return f"Error: Could not read file: {file_path}"

                lines = existing_content.splitlines(keepends=True)
                line_start = input_data.get('line_start', 1)
                line_end = input_data.get('line_end', line_start)

                # Validate line numbers
                if line_start < 1 or line_end > len(lines):
                    return f"Error: Invalid line range: {line_start}-{line_end}"

                # Update lines
                new_content_lines = content.splitlines(keepends=True)
                updated_lines = (
                    lines[:line_start - 1] +
                    new_content_lines +
                    lines[line_end:]
                )

                new_content = ''.join(updated_lines)

                if write_file_safely(path, new_content):
                    return json.dumps({
                        'success': True,
                        'operation': 'update',
                        'file_path': str(path),
                        'lines_updated': f"{line_start}-{line_end}",
                        'message': f"Updated lines {line_start}-{line_end} in: {file_path}",
                        'description': description
                    })
                else:
                    return f"Error: Could not update file: {file_path}"

            else:
                return f"Error: Invalid operation: {operation}"

        except json.JSONDecodeError:
            return "Error: Invalid JSON input"
        except KeyError as e:
            return f"Error: Missing required field: {e}"
        except Exception as e:
            logger.error(f"Error in code editor: {e}")
            return f"Error: {e}"

    async def _arun(self, input_str: str) -> str:
        """Async version."""
        return self._run(input_str)


class CodeGeneratorTool(BaseTool):
    """Tool for generating code snippets."""

    name: str = "generate_code"
    description: str = """
    Generates code based on a specification.
    Input should be JSON:
    {
        "language": "python",
        "description": "Create a function to calculate fibonacci",
        "context": "This will be part of a math utilities module",
        "include_tests": true
    }

    Returns generated code.
    """

    def _run(self, input_str: str) -> str:
        """Generate code.

        Args:
            input_str: JSON with generation specification

        Returns:
            Generated code or error
        """
        try:
            input_data = json.loads(input_str)
            language = input_data.get('language', 'python')
            description = input_data['description']
            context = input_data.get('context', '')
            include_tests = input_data.get('include_tests', False)

            # This is a placeholder - in a real implementation,
            # this would use the LLM to generate code
            # For now, return a template

            result = {
                'language': language,
                'description': description,
                'code': f"# Generated code for: {description}\n# TODO: Implement this functionality\n",
                'note': 'This is a template. Use the LLM directly for actual code generation.'
            }

            if include_tests:
                result['tests'] = f"# Test for: {description}\n# TODO: Implement tests\n"

            return json.dumps(result, indent=2)

        except json.JSONDecodeError:
            return "Error: Invalid JSON input"
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return f"Error: {e}"

    async def _arun(self, input_str: str) -> str:
        """Async version."""
        return self._run(input_str)


class RefactorCodeTool(BaseTool):
    """Tool for refactoring code."""

    name: str = "refactor_code"
    description: str = """
    Suggests refactoring for code.
    Input should be JSON:
    {
        "file_path": "/path/to/file.py",
        "function_name": "complex_function",  # Optional
        "refactor_type": "extract_method|rename|simplify"
    }

    Returns refactoring suggestions.
    """

    def _run(self, input_str: str) -> str:
        """Suggest code refactoring.

        Args:
            input_str: JSON with refactoring request

        Returns:
            Refactoring suggestions or error
        """
        try:
            input_data = json.loads(input_str)
            file_path = input_data['file_path']
            refactor_type = input_data.get('refactor_type', 'simplify')

            path = Path(file_path)
            if not path.exists():
                return f"Error: File not found: {file_path}"

            content = read_file_safely(path)
            if content is None:
                return f"Error: Could not read file: {file_path}"

            # This is a placeholder for actual refactoring logic
            result = {
                'file_path': str(path),
                'refactor_type': refactor_type,
                'suggestions': [
                    'Break down large functions into smaller ones',
                    'Extract magic numbers into constants',
                    'Add type hints for better code clarity',
                    'Improve variable naming for readability'
                ],
                'note': 'Use LLM for detailed refactoring suggestions'
            }

            return json.dumps(result, indent=2)

        except json.JSONDecodeError:
            return "Error: Invalid JSON input"
        except Exception as e:
            logger.error(f"Error refactoring code: {e}")
            return f"Error: {e}"

    async def _arun(self, input_str: str) -> str:
        """Async version."""
        return self._run(input_str)
