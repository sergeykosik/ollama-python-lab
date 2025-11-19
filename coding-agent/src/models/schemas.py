"""Data schemas for the coding agent."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class FileType(str, Enum):
    """Supported file types."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    MARKDOWN = "markdown"
    SQL = "sql"
    JSON = "json"
    YAML = "yaml"
    LOG = "log"
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, extension: str) -> "FileType":
        """Get FileType from file extension."""
        extension = extension.lower().lstrip('.')
        mapping = {
            'py': cls.PYTHON,
            'js': cls.JAVASCRIPT,
            'jsx': cls.JAVASCRIPT,
            'ts': cls.TYPESCRIPT,
            'tsx': cls.TYPESCRIPT,
            'java': cls.JAVA,
            'cs': cls.CSHARP,
            'md': cls.MARKDOWN,
            'sql': cls.SQL,
            'json': cls.JSON,
            'yaml': cls.YAML,
            'yml': cls.YAML,
            'log': cls.LOG,
        }
        return mapping.get(extension, cls.UNKNOWN)


class CodeEntity(BaseModel):
    """Represents a code entity (function, class, method)."""
    name: str
    type: str  # 'function', 'class', 'method', 'variable'
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    parameters: List[str] = Field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = Field(default_factory=list)
    complexity: Optional[int] = None


class FileMetadata(BaseModel):
    """Metadata about a source file."""
    path: str
    file_type: FileType
    size_bytes: int
    lines: int
    last_modified: datetime
    file_hash: str
    language: Optional[str] = None
    encoding: str = "utf-8"


class CodeAnalysis(BaseModel):
    """Results of code analysis."""
    file_path: str
    metadata: FileMetadata
    entities: List[CodeEntity] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class DatabaseSchema(BaseModel):
    """Database schema information."""
    table_name: str
    columns: List[Dict[str, Any]]
    primary_key: Optional[List[str]] = None
    foreign_keys: List[Dict[str, str]] = Field(default_factory=list)
    indexes: List[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    """Database query result."""
    query: str
    rows: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: float
    columns: List[str]


class LogEntry(BaseModel):
    """Parsed log entry."""
    timestamp: Optional[datetime] = None
    level: str
    message: str
    source: Optional[str] = None
    line_number: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A chunk of document for vector storage."""
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    total_chunks: int
    embedding: Optional[List[float]] = None


class SearchResult(BaseModel):
    """Search result from vector store."""
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str


class CodeEditRequest(BaseModel):
    """Request to edit code."""
    file_path: str
    operation: str  # 'create', 'update', 'delete', 'refactor'
    content: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    context: Optional[str] = None
    instructions: str

    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Ensure operation is valid."""
        valid_ops = ['create', 'update', 'delete', 'refactor']
        if v.lower() not in valid_ops:
            raise ValueError(f"operation must be one of {valid_ops}")
        return v.lower()


class CodeEditResponse(BaseModel):
    """Response from code edit operation."""
    success: bool
    file_path: str
    operation: str
    new_content: Optional[str] = None
    message: str
    errors: List[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Response from the agent."""
    query: str
    answer: str
    sources: List[str] = Field(default_factory=list)
    intermediate_steps: List[Dict[str, Any]] = Field(default_factory=list)
    execution_time_ms: float
    iterations: int


class IndexingStatus(BaseModel):
    """Status of indexing operation."""
    total_files: int
    processed_files: int
    failed_files: int
    total_chunks: int
    start_time: datetime
    end_time: Optional[datetime] = None
    errors: List[str] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Check if indexing is complete."""
        return self.end_time is not None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class ToolCallResult(BaseModel):
    """Result of a tool call."""
    tool_name: str
    input: str
    output: str
    success: bool
    execution_time_ms: float
    error: Optional[str] = None


class ConversationMessage(BaseModel):
    """A message in the conversation."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Ensure role is valid."""
        valid_roles = ['user', 'assistant', 'system']
        if v.lower() not in valid_roles:
            raise ValueError(f"role must be one of {valid_roles}")
        return v.lower()


class ConversationHistory(BaseModel):
    """Conversation history."""
    messages: List[ConversationMessage] = Field(default_factory=list)
    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation."""
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_recent_messages(self, n: int = 10) -> List[ConversationMessage]:
        """Get the n most recent messages."""
        return self.messages[-n:]

    def clear(self):
        """Clear conversation history."""
        self.messages.clear()
        self.updated_at = datetime.now()
