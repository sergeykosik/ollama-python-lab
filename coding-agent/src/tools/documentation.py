"""Documentation search and analysis tools."""

import json
import logging
from pathlib import Path
from typing import Optional
from langchain.tools import BaseTool
from pydantic import Field

from ..parsers.markdown_parser import MarkdownParser
from ..knowledge.indexer import CodebaseIndexer

logger = logging.getLogger(__name__)


class SearchDocumentationTool(BaseTool):
    """Tool for searching documentation."""

    name: str = "search_documentation"
    description: str = """
    Searches through documentation using semantic search.
    Input should be a search query string.
    Returns relevant documentation sections.

    Example input: "How to configure the database connection"
    """

    indexer: Optional[CodebaseIndexer] = Field(default=None)

    def _run(self, query: str) -> str:
        """Search documentation.

        Args:
            query: Search query

        Returns:
            Search results or error message
        """
        if not self.indexer:
            return "Error: Indexer not available"

        try:
            # Search with documentation filter
            results = self.indexer.search_with_scores(
                query=query,
                k=5,
                filter_dict={'type': 'documentation'}
            )

            if not results:
                return f"No documentation found for query: {query}"

            # Format results
            output = {
                'query': query,
                'total_results': len(results),
                'results': []
            }

            for doc, score in results:
                output['results'].append({
                    'content': doc.page_content[:500],  # Limit length
                    'source': doc.metadata.get('source', 'Unknown'),
                    'section': doc.metadata.get('section', 'N/A'),
                    'score': round(score, 3)
                })

            return json.dumps(output, indent=2)

        except Exception as e:
            logger.error(f"Error searching documentation: {e}")
            return f"Error: {e}"

    async def _arun(self, query: str) -> str:
        """Async version."""
        return self._run(query)


class ReadMarkdownTool(BaseTool):
    """Tool for reading and parsing markdown files."""

    name: str = "read_markdown"
    description: str = """
    Reads and parses a markdown documentation file.
    Input should be a file path to a markdown file.
    Returns parsed sections, links, and code blocks.

    Example input: "/docs/README.md"
    """

    markdown_parser: MarkdownParser = Field(default_factory=MarkdownParser)

    def _run(self, file_path: str) -> str:
        """Read markdown file.

        Args:
            file_path: Path to markdown file

        Returns:
            Parsed markdown or error message
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return f"Error: File not found: {file_path}"

            parsed = self.markdown_parser.parse_file(path)
            if not parsed:
                return f"Error: Could not parse file: {file_path}"

            # Simplify output
            output = {
                'file': str(path),
                'sections': [
                    {
                        'title': s['title'],
                        'level': s['level'],
                        'content_preview': s['content'][:200]
                    }
                    for s in parsed['sections']
                ],
                'total_links': len(parsed['links']),
                'total_code_blocks': len(parsed['code_blocks']),
                'links': parsed['links'][:10],  # First 10 links
                'code_blocks': [
                    {
                        'language': cb['language'],
                        'lines': len(cb['code'].splitlines())
                    }
                    for cb in parsed['code_blocks']
                ]
            }

            return json.dumps(output, indent=2)

        except Exception as e:
            logger.error(f"Error reading markdown: {e}")
            return f"Error: {e}"

    async def _arun(self, file_path: str) -> str:
        """Async version."""
        return self._run(file_path)


class GetMarkdownSectionTool(BaseTool):
    """Tool for getting a specific section from markdown."""

    name: str = "get_markdown_section"
    description: str = """
    Gets a specific section from a markdown file.
    Input should be JSON: {"file_path": "/docs/README.md", "section_title": "Installation"}
    Returns the section content.

    Example: {"file_path": "./README.md", "section_title": "Usage"}
    """

    markdown_parser: MarkdownParser = Field(default_factory=MarkdownParser)

    def _run(self, input_str: str) -> str:
        """Get markdown section.

        Args:
            input_str: JSON with file_path and section_title

        Returns:
            Section content or error message
        """
        try:
            input_data = json.loads(input_str)
            file_path = input_data['file_path']
            section_title = input_data['section_title']

            path = Path(file_path)
            if not path.exists():
                return f"Error: File not found: {file_path}"

            parsed = self.markdown_parser.parse_file(path)
            if not parsed:
                return f"Error: Could not parse file: {file_path}"

            # Find matching section
            for section_dict in parsed['sections']:
                if section_dict['title'].lower() == section_title.lower():
                    output = {
                        'file': str(path),
                        'section': section_dict['title'],
                        'level': section_dict['level'],
                        'content': section_dict['content']
                    }
                    return json.dumps(output, indent=2)

            return f"Section '{section_title}' not found in {file_path}"

        except json.JSONDecodeError:
            return "Error: Invalid JSON input"
        except Exception as e:
            logger.error(f"Error getting markdown section: {e}")
            return f"Error: {e}"

    async def _arun(self, input_str: str) -> str:
        """Async version."""
        return self._run(input_str)


class SearchCodeInDocsTool(BaseTool):
    """Tool for finding code examples in documentation."""

    name: str = "search_code_in_docs"
    description: str = """
    Searches for code examples in documentation.
    Input should be JSON: {"query": "database connection", "language": "python"}
    Returns code blocks from documentation matching the criteria.

    Example: {"query": "authentication", "language": "javascript"}
    """

    indexer: Optional[CodebaseIndexer] = Field(default=None)
    markdown_parser: MarkdownParser = Field(default_factory=MarkdownParser)

    def _run(self, input_str: str) -> str:
        """Search for code in documentation.

        Args:
            input_str: JSON with query and optional language

        Returns:
            Code examples or error message
        """
        try:
            input_data = json.loads(input_str)
            query = input_data['query']
            language = input_data.get('language')

            if not self.indexer:
                return "Error: Indexer not available"

            # Search documentation
            results = self.indexer.search(
                query=query,
                k=10,
                filter_dict={'type': 'documentation'}
            )

            code_examples = []

            # Parse markdown files to find code blocks
            seen_files = set()
            for doc in results:
                source = doc.metadata.get('source')
                if source and source not in seen_files:
                    seen_files.add(source)
                    path = Path(source)

                    if path.exists() and path.suffix == '.md':
                        parsed = self.markdown_parser.parse_file(path)
                        if parsed:
                            for code_block in parsed['code_blocks']:
                                if language is None or code_block['language'] == language:
                                    code_examples.append({
                                        'source': str(path),
                                        'language': code_block['language'],
                                        'code': code_block['code'][:500]  # Limit length
                                    })

            output = {
                'query': query,
                'language_filter': language,
                'total_examples': len(code_examples),
                'examples': code_examples[:5]  # First 5 examples
            }

            return json.dumps(output, indent=2)

        except json.JSONDecodeError:
            return "Error: Invalid JSON input"
        except Exception as e:
            logger.error(f"Error searching code in docs: {e}")
            return f"Error: {e}"

    async def _arun(self, input_str: str) -> str:
        """Async version."""
        return self._run(input_str)
