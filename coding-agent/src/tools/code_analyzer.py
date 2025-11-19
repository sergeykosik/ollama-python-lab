"""Code analysis tools for the agent."""

import json
import logging
from pathlib import Path
from typing import Optional
from langchain.tools import BaseTool
from pydantic import Field

from ..parsers.code_parser import CodeParser, get_function_signature, get_class_signature
from ..utils.file_utils import read_file_safely

logger = logging.getLogger(__name__)


class CodeAnalyzerTool(BaseTool):
    """Tool for analyzing code structure."""

    name: str = "analyze_code"
    description: str = """
    Analyzes code structure and extracts information about functions, classes, and imports.
    Input should be a file path to a code file.
    Returns structured analysis including:
    - Functions and their signatures
    - Classes and their structure
    - Import statements
    - Code metrics (lines, complexity)

    Example input: "/path/to/file.py"
    """

    code_parser: CodeParser = Field(default_factory=CodeParser)

    def _run(self, file_path: str) -> str:
        """Analyze a code file.

        Args:
            file_path: Path to the code file

        Returns:
            JSON string with analysis results
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return f"Error: File not found: {file_path}"

            analysis = self.code_parser.parse_file(path)
            if not analysis:
                return f"Error: Could not analyze file: {file_path}"

            # Format results
            result = {
                'file': str(path),
                'language': analysis.metadata.language,
                'metrics': analysis.metrics,
                'functions': [],
                'classes': [],
                'imports': analysis.imports
            }

            # Add functions
            for entity in analysis.entities:
                if entity.type in ['function', 'async_function']:
                    result['functions'].append({
                        'name': entity.name,
                        'signature': get_function_signature(entity),
                        'line_start': entity.line_start,
                        'line_end': entity.line_end,
                        'parameters': entity.parameters,
                        'docstring': entity.docstring,
                        'complexity': entity.complexity
                    })
                elif entity.type == 'class':
                    result['classes'].append({
                        'name': entity.name,
                        'signature': get_class_signature(entity),
                        'line_start': entity.line_start,
                        'line_end': entity.line_end,
                        'bases': entity.parameters,
                        'docstring': entity.docstring
                    })

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error in code analyzer: {e}")
            return f"Error analyzing code: {e}"

    async def _arun(self, file_path: str) -> str:
        """Async version."""
        return self._run(file_path)


class FindFunctionTool(BaseTool):
    """Tool for finding functions in code."""

    name: str = "find_function"
    description: str = """
    Finds a specific function in a code file and returns its details.
    Input should be JSON: {"file_path": "/path/to/file.py", "function_name": "my_function"}
    Returns the function signature, location, and code.
    """

    code_parser: CodeParser = Field(default_factory=CodeParser)

    def _run(self, input_str: str) -> str:
        """Find a function in a file.

        Args:
            input_str: JSON string with file_path and function_name

        Returns:
            Function details or error message
        """
        try:
            input_data = json.loads(input_str)
            file_path = input_data['file_path']
            function_name = input_data['function_name']

            path = Path(file_path)
            if not path.exists():
                return f"Error: File not found: {file_path}"

            # Parse file
            analysis = self.code_parser.parse_file(path)
            if not analysis:
                return f"Error: Could not parse file: {file_path}"

            # Find function
            for entity in analysis.entities:
                if entity.name == function_name and entity.type in ['function', 'async_function']:
                    # Read the function code
                    content = read_file_safely(path)
                    if content:
                        lines = content.splitlines()
                        function_code = '\n'.join(
                            lines[entity.line_start - 1:entity.line_end]
                        )

                        return json.dumps({
                            'name': entity.name,
                            'signature': get_function_signature(entity),
                            'file': str(path),
                            'line_start': entity.line_start,
                            'line_end': entity.line_end,
                            'parameters': entity.parameters,
                            'return_type': entity.return_type,
                            'docstring': entity.docstring,
                            'complexity': entity.complexity,
                            'code': function_code
                        }, indent=2)

            return f"Function '{function_name}' not found in {file_path}"

        except json.JSONDecodeError:
            return "Error: Invalid JSON input. Expected: {\"file_path\": \"...\", \"function_name\": \"...\"}"
        except Exception as e:
            logger.error(f"Error finding function: {e}")
            return f"Error: {e}"

    async def _arun(self, input_str: str) -> str:
        """Async version."""
        return self._run(input_str)


class FindClassTool(BaseTool):
    """Tool for finding classes in code."""

    name: str = "find_class"
    description: str = """
    Finds a specific class in a code file and returns its details.
    Input should be JSON: {"file_path": "/path/to/file.py", "class_name": "MyClass"}
    Returns the class definition, methods, and properties.
    """

    code_parser: CodeParser = Field(default_factory=CodeParser)

    def _run(self, input_str: str) -> str:
        """Find a class in a file.

        Args:
            input_str: JSON string with file_path and class_name

        Returns:
            Class details or error message
        """
        try:
            input_data = json.loads(input_str)
            file_path = input_data['file_path']
            class_name = input_data['class_name']

            path = Path(file_path)
            if not path.exists():
                return f"Error: File not found: {file_path}"

            # Parse file
            analysis = self.code_parser.parse_file(path)
            if not analysis:
                return f"Error: Could not parse file: {file_path}"

            # Find class
            for entity in analysis.entities:
                if entity.name == class_name and entity.type == 'class':
                    # Read the class code
                    content = read_file_safely(path)
                    if content:
                        lines = content.splitlines()
                        class_code = '\n'.join(
                            lines[entity.line_start - 1:entity.line_end]
                        )

                        return json.dumps({
                            'name': entity.name,
                            'signature': get_class_signature(entity),
                            'file': str(path),
                            'line_start': entity.line_start,
                            'line_end': entity.line_end,
                            'bases': entity.parameters,
                            'docstring': entity.docstring,
                            'code': class_code
                        }, indent=2)

            return f"Class '{class_name}' not found in {file_path}"

        except json.JSONDecodeError:
            return "Error: Invalid JSON input. Expected: {\"file_path\": \"...\", \"class_name\": \"...\"}"
        except Exception as e:
            logger.error(f"Error finding class: {e}")
            return f"Error: {e}"

    async def _arun(self, input_str: str) -> str:
        """Async version."""
        return self._run(input_str)


class CodeComplexityTool(BaseTool):
    """Tool for analyzing code complexity."""

    name: str = "analyze_complexity"
    description: str = """
    Analyzes the complexity of code in a file.
    Input should be a file path.
    Returns complexity metrics for all functions in the file.
    """

    code_parser: CodeParser = Field(default_factory=CodeParser)

    def _run(self, file_path: str) -> str:
        """Analyze code complexity.

        Args:
            file_path: Path to the code file

        Returns:
            Complexity analysis
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return f"Error: File not found: {file_path}"

            analysis = self.code_parser.parse_file(path)
            if not analysis:
                return f"Error: Could not analyze file: {file_path}"

            # Extract functions with complexity
            functions_complexity = []
            for entity in analysis.entities:
                if entity.type in ['function', 'async_function'] and entity.complexity:
                    functions_complexity.append({
                        'name': entity.name,
                        'complexity': entity.complexity,
                        'lines': entity.line_end - entity.line_start + 1,
                        'parameters': len(entity.parameters)
                    })

            # Sort by complexity
            functions_complexity.sort(key=lambda x: x['complexity'], reverse=True)

            result = {
                'file': str(path),
                'total_functions': len(functions_complexity),
                'average_complexity': sum(f['complexity'] for f in functions_complexity) / len(functions_complexity) if functions_complexity else 0,
                'max_complexity': max((f['complexity'] for f in functions_complexity), default=0),
                'functions': functions_complexity
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error analyzing complexity: {e}")
            return f"Error: {e}"

    async def _arun(self, file_path: str) -> str:
        """Async version."""
        return self._run(file_path)
