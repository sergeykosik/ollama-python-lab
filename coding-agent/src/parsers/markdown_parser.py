"""Markdown parsing utilities."""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from ..utils.file_utils import read_file_safely

logger = logging.getLogger(__name__)


class MarkdownSection:
    """Represents a section in a markdown document."""

    def __init__(
        self,
        title: str,
        level: int,
        content: str,
        line_start: int,
        line_end: int
    ):
        """Initialize markdown section.

        Args:
            title: Section title
            level: Heading level (1-6)
            content: Section content
            line_start: Starting line number
            line_end: Ending line number
        """
        self.title = title
        self.level = level
        self.content = content
        self.line_start = line_start
        self.line_end = line_end
        self.subsections: List[MarkdownSection] = []

    def add_subsection(self, section: 'MarkdownSection'):
        """Add a subsection."""
        self.subsections.append(section)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'level': self.level,
            'content': self.content,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'subsections': [s.to_dict() for s in self.subsections]
        }


class MarkdownParser:
    """Parser for markdown files."""

    def __init__(self):
        """Initialize markdown parser."""
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        self.link_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
        self.code_block_pattern = re.compile(r'^```(\w+)?$')
        self.list_item_pattern = re.compile(r'^(\s*)([-*+]|\d+\.)\s+(.+)$')

    def parse_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse a markdown file.

        Args:
            file_path: Path to markdown file

        Returns:
            Dictionary with parsed content or None if error occurs
        """
        content = read_file_safely(file_path)
        if not content:
            return None

        return self.parse_content(content)

    def parse_content(self, content: str) -> Dict[str, Any]:
        """Parse markdown content.

        Args:
            content: Markdown content

        Returns:
            Dictionary with parsed content
        """
        lines = content.splitlines()
        sections = self._parse_sections(lines)
        links = self._extract_links(content)
        code_blocks = self._extract_code_blocks(lines)
        lists = self._extract_lists(lines)

        return {
            'sections': [s.to_dict() for s in sections],
            'links': links,
            'code_blocks': code_blocks,
            'lists': lists,
            'metadata': self._extract_metadata(content)
        }

    def _parse_sections(self, lines: List[str]) -> List[MarkdownSection]:
        """Parse markdown sections.

        Args:
            lines: List of lines

        Returns:
            List of top-level MarkdownSection objects
        """
        sections = []
        current_section: Optional[MarkdownSection] = None
        section_content: List[str] = []
        section_start = 0

        for i, line in enumerate(lines):
            heading_match = self.heading_pattern.match(line)

            if heading_match:
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(section_content).strip()
                    current_section.line_end = i - 1

                # Create new section
                hashes = heading_match.group(1)
                title = heading_match.group(2).strip()
                level = len(hashes)

                current_section = MarkdownSection(
                    title=title,
                    level=level,
                    content='',
                    line_start=i,
                    line_end=i
                )

                sections.append(current_section)
                section_content = []
                section_start = i + 1
            else:
                section_content.append(line)

        # Save last section
        if current_section:
            current_section.content = '\n'.join(section_content).strip()
            current_section.line_end = len(lines) - 1

        # Build hierarchy
        return self._build_section_hierarchy(sections)

    def _build_section_hierarchy(
        self,
        sections: List[MarkdownSection]
    ) -> List[MarkdownSection]:
        """Build hierarchical structure of sections.

        Args:
            sections: Flat list of sections

        Returns:
            List of top-level sections with nested subsections
        """
        if not sections:
            return []

        root_sections = []
        stack: List[MarkdownSection] = []

        for section in sections:
            # Pop stack until we find a parent with lower level
            while stack and stack[-1].level >= section.level:
                stack.pop()

            if stack:
                # Add as subsection to the current parent
                stack[-1].add_subsection(section)
            else:
                # Top-level section
                root_sections.append(section)

            stack.append(section)

        return root_sections

    def _extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extract all links from content.

        Args:
            content: Markdown content

        Returns:
            List of dictionaries with 'text' and 'url' keys
        """
        links = []
        for match in self.link_pattern.finditer(content):
            links.append({
                'text': match.group(1),
                'url': match.group(2)
            })
        return links

    def _extract_code_blocks(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract code blocks from content.

        Args:
            lines: List of lines

        Returns:
            List of code block dictionaries
        """
        code_blocks = []
        in_code_block = False
        current_block = {'language': None, 'code': [], 'line_start': 0}

        for i, line in enumerate(lines):
            match = self.code_block_pattern.match(line)

            if match:
                if not in_code_block:
                    # Start of code block
                    in_code_block = True
                    current_block = {
                        'language': match.group(1) or 'text',
                        'code': [],
                        'line_start': i
                    }
                else:
                    # End of code block
                    in_code_block = False
                    current_block['code'] = '\n'.join(current_block['code'])
                    current_block['line_end'] = i
                    code_blocks.append(current_block)
            elif in_code_block:
                current_block['code'].append(line)

        return code_blocks

    def _extract_lists(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract lists from content.

        Args:
            lines: List of lines

        Returns:
            List of list structures
        """
        lists = []
        current_list: Optional[Dict[str, Any]] = None
        current_items: List[str] = []

        for i, line in enumerate(lines):
            match = self.list_item_pattern.match(line)

            if match:
                indent = len(match.group(1))
                marker = match.group(2)
                content = match.group(3)

                if current_list is None:
                    # Start new list
                    list_type = 'ordered' if marker[0].isdigit() else 'unordered'
                    current_list = {
                        'type': list_type,
                        'items': [],
                        'line_start': i
                    }

                current_list['items'].append({
                    'content': content,
                    'indent': indent
                })
            else:
                # End of list
                if current_list:
                    current_list['line_end'] = i - 1
                    lists.append(current_list)
                    current_list = None

        # Save last list
        if current_list:
            current_list['line_end'] = len(lines) - 1
            lists.append(current_list)

        return lists

    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from markdown.

        Looks for YAML frontmatter and other metadata.

        Args:
            content: Markdown content

        Returns:
            Dictionary with metadata
        """
        metadata = {}

        # Check for YAML frontmatter
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                # Simple key-value parsing
                for line in frontmatter.splitlines():
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()

        return metadata

    def search_sections(
        self,
        sections: List[MarkdownSection],
        query: str,
        case_sensitive: bool = False
    ) -> List[MarkdownSection]:
        """Search sections by title or content.

        Args:
            sections: List of sections to search
            query: Search query
            case_sensitive: Whether search is case-sensitive

        Returns:
            List of matching sections
        """
        matches = []

        for section in sections:
            text_to_search = section.title + ' ' + section.content
            if not case_sensitive:
                text_to_search = text_to_search.lower()
                query = query.lower()

            if query in text_to_search:
                matches.append(section)

            # Search subsections
            matches.extend(self.search_sections(section.subsections, query, case_sensitive))

        return matches

    def get_table_of_contents(
        self,
        sections: List[MarkdownSection],
        max_level: int = 3
    ) -> str:
        """Generate table of contents.

        Args:
            sections: List of sections
            max_level: Maximum heading level to include

        Returns:
            Markdown-formatted table of contents
        """
        lines = []

        def add_section(section: MarkdownSection, depth: int = 0):
            if section.level <= max_level:
                indent = '  ' * depth
                lines.append(f"{indent}- [{section.title}](#{self._slugify(section.title)})")

            for subsection in section.subsections:
                add_section(subsection, depth + 1)

        for section in sections:
            add_section(section)

        return '\n'.join(lines)

    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug.

        Args:
            text: Text to slugify

        Returns:
            Slugified text
        """
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[\s_]+', '-', text)
        text = text.strip('-')
        return text


def extract_code_snippets(content: str, language: Optional[str] = None) -> List[str]:
    """Extract code snippets from markdown.

    Args:
        content: Markdown content
        language: Optional language filter

    Returns:
        List of code snippets
    """
    parser = MarkdownParser()
    lines = content.splitlines()
    code_blocks = parser._extract_code_blocks(lines)

    snippets = []
    for block in code_blocks:
        if language is None or block['language'] == language:
            snippets.append(block['code'])

    return snippets


def convert_to_plain_text(content: str) -> str:
    """Convert markdown to plain text.

    Args:
        content: Markdown content

    Returns:
        Plain text version
    """
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', content)

    # Remove inline code
    text = re.sub(r'`[^`]+`', '', text)

    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove images
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)

    # Remove headings markers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Remove bold/italic
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # Remove list markers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    return text.strip()
