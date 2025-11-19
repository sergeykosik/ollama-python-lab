"""Log file parsing utilities."""

import re
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Iterator
from ..models.schemas import LogEntry
from ..utils.file_utils import read_file_safely

logger = logging.getLogger(__name__)


class LogParser:
    """Parser for log files."""

    # Common log patterns
    PATTERNS = {
        'python': re.compile(
            r'(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+'
            r'(?P<source>\S+)\s+-\s+(?P<level>\w+)\s+-\s+(?P<message>.*)'
        ),
        'generic': re.compile(
            r'(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+'
            r'\[(?P<level>\w+)\]\s+(?P<message>.*)'
        ),
        'apache': re.compile(
            r'(?P<ip>[\d.]+)\s+-\s+-\s+\[(?P<timestamp>[^\]]+)\]\s+'
            r'"(?P<method>\w+)\s+(?P<path>\S+)\s+HTTP/[\d.]+"\s+'
            r'(?P<status>\d+)\s+(?P<size>\d+)'
        ),
        'nginx': re.compile(
            r'(?P<ip>[\d.]+)\s+-\s+-\s+\[(?P<timestamp>[^\]]+)\]\s+'
            r'"(?P<request>[^"]+)"\s+(?P<status>\d+)\s+(?P<size>\d+)'
        ),
        'simple': re.compile(
            r'(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL|FATAL)\s*:\s*(?P<message>.*)'
        ),
    }

    def parse_file(self, file_path: Path, pattern_name: str = 'auto') -> List[LogEntry]:
        """Parse a log file.

        Args:
            file_path: Path to log file
            pattern_name: Name of pattern to use ('auto', 'python', 'generic', etc.)

        Returns:
            List of LogEntry objects
        """
        content = read_file_safely(file_path)
        if not content:
            return []

        if pattern_name == 'auto':
            pattern_name = self._detect_format(content)

        return list(self.parse_lines(content.splitlines(), pattern_name))

    def parse_lines(
        self,
        lines: List[str],
        pattern_name: str = 'simple'
    ) -> Iterator[LogEntry]:
        """Parse log lines.

        Args:
            lines: List of log lines
            pattern_name: Name of pattern to use

        Yields:
            LogEntry objects
        """
        pattern = self.PATTERNS.get(pattern_name, self.PATTERNS['simple'])

        for line_num, line in enumerate(lines, 1):
            entry = self._parse_line(line, pattern, line_num)
            if entry:
                yield entry

    def _parse_line(
        self,
        line: str,
        pattern: re.Pattern,
        line_number: int
    ) -> Optional[LogEntry]:
        """Parse a single log line.

        Args:
            line: Log line
            pattern: Regex pattern to use
            line_number: Line number in file

        Returns:
            LogEntry object or None if line doesn't match
        """
        match = pattern.search(line)
        if not match:
            # If no match, treat as a simple message
            return LogEntry(
                level='UNKNOWN',
                message=line.strip(),
                line_number=line_number
            )

        groups = match.groupdict()

        # Parse timestamp if present
        timestamp = None
        if 'timestamp' in groups:
            timestamp = self._parse_timestamp(groups['timestamp'])

        # Extract level
        level = groups.get('level', 'INFO').upper()

        # Extract message
        message = groups.get('message', line).strip()

        # Extract source
        source = groups.get('source')

        # Additional metadata
        metadata = {k: v for k, v in groups.items() if k not in ['timestamp', 'level', 'message', 'source']}

        return LogEntry(
            timestamp=timestamp,
            level=level,
            message=message,
            source=source,
            line_number=line_number,
            metadata=metadata
        )

    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string.

        Args:
            timestamp_str: Timestamp string

        Returns:
            datetime object or None if parsing fails
        """
        # Common timestamp formats
        formats = [
            '%Y-%m-%d %H:%M:%S,%f',
            '%Y-%m-%d %H:%M:%S',
            '%d/%b/%Y:%H:%M:%S %z',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str.strip(), fmt)
            except ValueError:
                continue

        logger.debug(f"Could not parse timestamp: {timestamp_str}")
        return None

    def _detect_format(self, content: str) -> str:
        """Detect log format from content.

        Args:
            content: Log file content

        Returns:
            Pattern name
        """
        # Sample first few lines
        lines = content.splitlines()[:10]
        sample = '\n'.join(lines)

        # Try each pattern
        for name, pattern in self.PATTERNS.items():
            if pattern.search(sample):
                logger.info(f"Detected log format: {name}")
                return name

        logger.info("Could not detect log format, using simple pattern")
        return 'simple'

    def filter_by_level(
        self,
        entries: List[LogEntry],
        min_level: str = 'INFO'
    ) -> List[LogEntry]:
        """Filter log entries by minimum level.

        Args:
            entries: List of log entries
            min_level: Minimum log level

        Returns:
            Filtered list
        """
        level_order = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'FATAL']
        if min_level not in level_order:
            return entries

        min_index = level_order.index(min_level)
        return [
            entry for entry in entries
            if entry.level in level_order and level_order.index(entry.level) >= min_index
        ]

    def filter_by_time(
        self,
        entries: List[LogEntry],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[LogEntry]:
        """Filter log entries by time range.

        Args:
            entries: List of log entries
            start_time: Start time (inclusive)
            end_time: End time (inclusive)

        Returns:
            Filtered list
        """
        filtered = entries

        if start_time:
            filtered = [e for e in filtered if e.timestamp and e.timestamp >= start_time]

        if end_time:
            filtered = [e for e in filtered if e.timestamp and e.timestamp <= end_time]

        return filtered

    def search(
        self,
        entries: List[LogEntry],
        pattern: str,
        case_sensitive: bool = False
    ) -> List[LogEntry]:
        """Search log entries by pattern.

        Args:
            entries: List of log entries
            pattern: Search pattern (regex)
            case_sensitive: Whether search is case-sensitive

        Returns:
            Matching entries
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)

        return [entry for entry in entries if regex.search(entry.message)]

    def get_statistics(self, entries: List[LogEntry]) -> Dict[str, Any]:
        """Get statistics from log entries.

        Args:
            entries: List of log entries

        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_entries': len(entries),
            'by_level': {},
            'by_source': {},
            'time_range': None,
            'errors': []
        }

        # Count by level
        for entry in entries:
            level = entry.level
            stats['by_level'][level] = stats['by_level'].get(level, 0) + 1

        # Count by source
        for entry in entries:
            if entry.source:
                source = entry.source
                stats['by_source'][source] = stats['by_source'].get(source, 0) + 1

        # Time range
        entries_with_time = [e for e in entries if e.timestamp]
        if entries_with_time:
            timestamps = [e.timestamp for e in entries_with_time]
            stats['time_range'] = {
                'start': min(timestamps),
                'end': max(timestamps),
                'duration_seconds': (max(timestamps) - min(timestamps)).total_seconds()
            }

        # Collect error messages
        error_entries = [e for e in entries if e.level in ['ERROR', 'CRITICAL', 'FATAL']]
        stats['errors'] = [
            {
                'level': e.level,
                'message': e.message,
                'timestamp': e.timestamp.isoformat() if e.timestamp else None,
                'source': e.source
            }
            for e in error_entries[:10]  # First 10 errors
        ]

        return stats


def format_log_entry(entry: LogEntry) -> str:
    """Format a log entry as a string.

    Args:
        entry: LogEntry object

    Returns:
        Formatted string
    """
    parts = []

    if entry.timestamp:
        parts.append(entry.timestamp.strftime('%Y-%m-%d %H:%M:%S'))

    if entry.source:
        parts.append(f"[{entry.source}]")

    parts.append(f"[{entry.level}]")
    parts.append(entry.message)

    return ' '.join(parts)


def analyze_error_patterns(entries: List[LogEntry]) -> Dict[str, Any]:
    """Analyze error patterns in log entries.

    Args:
        entries: List of log entries

    Returns:
        Dictionary with error analysis
    """
    error_entries = [e for e in entries if e.level in ['ERROR', 'CRITICAL', 'FATAL']]

    # Group similar errors
    error_groups: Dict[str, List[LogEntry]] = {}
    for entry in error_entries:
        # Simple grouping by first 50 characters of message
        key = entry.message[:50] if len(entry.message) > 50 else entry.message
        if key not in error_groups:
            error_groups[key] = []
        error_groups[key].append(entry)

    # Find most common errors
    common_errors = sorted(
        [
            {
                'pattern': pattern,
                'count': len(occurrences),
                'example': occurrences[0].message,
                'first_seen': min(e.timestamp for e in occurrences if e.timestamp) if any(e.timestamp for e in occurrences) else None,
                'last_seen': max(e.timestamp for e in occurrences if e.timestamp) if any(e.timestamp for e in occurrences) else None
            }
            for pattern, occurrences in error_groups.items()
        ],
        key=lambda x: x['count'],
        reverse=True
    )

    return {
        'total_errors': len(error_entries),
        'unique_patterns': len(error_groups),
        'most_common': common_errors[:10]
    }
