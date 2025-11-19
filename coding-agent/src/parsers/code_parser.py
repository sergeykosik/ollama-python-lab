"""Code parsing utilities using AST and tree-sitter."""

import ast
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from ..models.schemas import CodeEntity, CodeAnalysis, FileMetadata, FileType
from ..utils.file_utils import get_file_metadata, read_file_safely

logger = logging.getLogger(__name__)


class PythonParser:
    """Parser for Python code using AST."""

    def parse_file(self, file_path: Path) -> Optional[CodeAnalysis]:
        """Parse a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            CodeAnalysis object or None if parsing fails
        """
        content = read_file_safely(file_path)
        if not content:
            return None

        metadata = get_file_metadata(file_path)
        if not metadata:
            return None

        try:
            tree = ast.parse(content)
            entities = self._extract_entities(tree)
            imports = self._extract_imports(tree)

            return CodeAnalysis(
                file_path=str(file_path),
                metadata=metadata,
                entities=entities,
                imports=imports,
                dependencies=[],
                issues=[],
                metrics=self._calculate_metrics(tree, content)
            )
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def _extract_entities(self, tree: ast.AST) -> List[CodeEntity]:
        """Extract code entities from AST.

        Args:
            tree: AST tree

        Returns:
            List of CodeEntity objects
        """
        entities = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                entity = self._parse_function(node)
                entities.append(entity)
            elif isinstance(node, ast.AsyncFunctionDef):
                entity = self._parse_function(node, is_async=True)
                entities.append(entity)
            elif isinstance(node, ast.ClassDef):
                entity = self._parse_class(node)
                entities.append(entity)

        return entities

    def _parse_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_async: bool = False
    ) -> CodeEntity:
        """Parse a function definition.

        Args:
            node: Function AST node
            is_async: Whether function is async

        Returns:
            CodeEntity object
        """
        # Extract parameters
        params = [arg.arg for arg in node.args.args]

        # Extract return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)

        # Extract decorators
        decorators = [ast.unparse(dec) for dec in node.decorator_list]

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Calculate complexity (simple count of decision points)
        complexity = self._calculate_complexity(node)

        return CodeEntity(
            name=node.name,
            type='async_function' if is_async else 'function',
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=docstring,
            parameters=params,
            return_type=return_type,
            decorators=decorators,
            complexity=complexity
        )

    def _parse_class(self, node: ast.ClassDef) -> CodeEntity:
        """Parse a class definition.

        Args:
            node: Class AST node

        Returns:
            CodeEntity object
        """
        # Extract base classes
        bases = [ast.unparse(base) for base in node.bases]

        # Extract decorators
        decorators = [ast.unparse(dec) for dec in node.decorator_list]

        # Extract docstring
        docstring = ast.get_docstring(node)

        return CodeEntity(
            name=node.name,
            type='class',
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=docstring,
            parameters=bases,  # Store base classes in parameters
            decorators=decorators
        )

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements.

        Args:
            tree: AST tree

        Returns:
            List of import strings
        """
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)

        return imports

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity.

        Args:
            node: AST node

        Returns:
            Complexity score
        """
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Decision points increase complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _calculate_metrics(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Calculate code metrics.

        Args:
            tree: AST tree
            content: File content

        Returns:
            Dictionary of metrics
        """
        lines = content.splitlines()
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]

        # Count different node types
        function_count = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef))
        class_count = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef))
        import_count = sum(
            1 for _ in ast.walk(tree)
            if isinstance(_, (ast.Import, ast.ImportFrom))
        )

        return {
            'total_lines': len(lines),
            'code_lines': len(code_lines),
            'blank_lines': len(lines) - len(code_lines),
            'function_count': function_count,
            'class_count': class_count,
            'import_count': import_count
        }


class GenericParser:
    """Generic parser for non-Python files."""

    def parse_file(self, file_path: Path) -> Optional[CodeAnalysis]:
        """Parse a generic code file.

        Args:
            file_path: Path to file

        Returns:
            CodeAnalysis object with basic information
        """
        metadata = get_file_metadata(file_path)
        if not metadata:
            return None

        content = read_file_safely(file_path)
        if not content:
            return None

        # Basic analysis without AST
        imports = self._extract_imports_regex(content, metadata.file_type)
        metrics = self._calculate_basic_metrics(content)

        return CodeAnalysis(
            file_path=str(file_path),
            metadata=metadata,
            entities=[],  # Would need tree-sitter for proper parsing
            imports=imports,
            dependencies=[],
            issues=[],
            metrics=metrics
        )

    def _extract_imports_regex(self, content: str, file_type: FileType) -> List[str]:
        """Extract imports using regex patterns.

        Args:
            content: File content
            file_type: Type of file

        Returns:
            List of import strings
        """
        import re
        imports = []

        if file_type == FileType.JAVASCRIPT or file_type == FileType.TYPESCRIPT:
            # Match: import ... from '...'
            pattern = r"import\s+.*\s+from\s+['\"]([^'\"]+)['\"]"
            imports.extend(re.findall(pattern, content))
            # Match: require('...')
            pattern = r"require\(['\"]([^'\"]+)['\"]\)"
            imports.extend(re.findall(pattern, content))
        elif file_type == FileType.JAVA:
            # Match: import ...;
            pattern = r"import\s+([\w.]+);"
            imports.extend(re.findall(pattern, content))
        elif file_type == FileType.CSHARP:
            # Match: using ...;
            pattern = r"using\s+([\w.]+);"
            imports.extend(re.findall(pattern, content))

        return imports

    def _calculate_basic_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate basic metrics for any file.

        Args:
            content: File content

        Returns:
            Dictionary of metrics
        """
        lines = content.splitlines()
        code_lines = [line for line in lines if line.strip()]

        return {
            'total_lines': len(lines),
            'code_lines': len(code_lines),
            'blank_lines': len(lines) - len(code_lines),
            'character_count': len(content)
        }


class CodeParser:
    """Main code parser that delegates to specific parsers."""

    def __init__(self):
        """Initialize code parser."""
        self.python_parser = PythonParser()
        self.generic_parser = GenericParser()

    def parse_file(self, file_path: Path) -> Optional[CodeAnalysis]:
        """Parse a code file.

        Args:
            file_path: Path to file

        Returns:
            CodeAnalysis object or None if parsing fails
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        # Determine file type
        file_type = FileType.from_extension(file_path.suffix)

        # Use appropriate parser
        if file_type == FileType.PYTHON:
            return self.python_parser.parse_file(file_path)
        else:
            return self.generic_parser.parse_file(file_path)

    def parse_code_string(self, code: str, language: str = "python") -> Optional[List[CodeEntity]]:
        """Parse code from a string.

        Args:
            code: Code string
            language: Programming language

        Returns:
            List of CodeEntity objects or None if parsing fails
        """
        if language.lower() == "python":
            try:
                tree = ast.parse(code)
                return self.python_parser._extract_entities(tree)
            except Exception as e:
                logger.error(f"Error parsing code string: {e}")
                return None
        else:
            logger.warning(f"Parsing for {language} not yet implemented")
            return None


def get_function_signature(entity: CodeEntity) -> str:
    """Get a readable function signature.

    Args:
        entity: CodeEntity object

    Returns:
        Function signature string
    """
    if entity.type not in ['function', 'async_function']:
        return entity.name

    params_str = ', '.join(entity.parameters)
    prefix = 'async ' if entity.type == 'async_function' else ''
    return_str = f" -> {entity.return_type}" if entity.return_type else ""

    return f"{prefix}def {entity.name}({params_str}){return_str}"


def get_class_signature(entity: CodeEntity) -> str:
    """Get a readable class signature.

    Args:
        entity: CodeEntity object

    Returns:
        Class signature string
    """
    if entity.type != 'class':
        return entity.name

    bases_str = f"({', '.join(entity.parameters)})" if entity.parameters else ""
    return f"class {entity.name}{bases_str}"
