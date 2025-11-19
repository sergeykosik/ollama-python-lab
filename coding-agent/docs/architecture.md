# Coding Agent Architecture

This document provides a detailed overview of the Coding Agent's architecture, design decisions, and implementation details.

## System Overview

The Coding Agent is built as a modular, extensible system using the **ReAct (Reasoning and Acting)** pattern. It combines:

1. **Large Language Model (LLM)**: Ollama-hosted models for reasoning
2. **Vector Store**: ChromaDB for semantic code search
3. **Tool System**: LangChain tools for various operations
4. **Memory Management**: Conversation and code context tracking
5. **Parser Layer**: Multi-language code and document parsing

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                       │
│                    (CLI / Future: Web UI)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Agent Core (ReAct)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Reasoning  │  │   Planning   │  │   Execution  │      │
│  │   (LLM)      │→ │   (Tools)    │→ │   (Actions)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Knowledge  │  │     Tools    │  │    Memory    │
│     Base     │  │    System    │  │  Management  │
└──────────────┘  └──────────────┘  └──────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ ChromaDB │  │  MySQL   │  │   Files  │  │  Parsers │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Agent Core (`src/agent/core.py`)

The central component that orchestrates all operations.

**Responsibilities**:
- Initialize all subsystems (LLM, vector store, tools, memory)
- Execute the ReAct reasoning loop
- Manage tool invocations
- Handle conversation flow

**Key Classes**:
- `CodingAgent`: Main agent class
  - `__init__()`: Initialize all components
  - `run()`: Execute a user query
  - `index_codebase()`: Index code for search
  - `search_codebase()`: Semantic code search
  - `shutdown()`: Cleanup resources

**Design Pattern**: Facade pattern - provides simplified interface to complex subsystems

### 2. Knowledge Base Layer (`src/knowledge/`)

Handles semantic search and code indexing.

#### 2.1 Embeddings (`embeddings.py`)

**Purpose**: Generate vector embeddings for text and code

**Classes**:
- `EmbeddingManager`: Base embedding manager
  - Supports Ollama and HuggingFace models
  - Implements retry logic for reliability
  - Batching for efficiency

- `CodeEmbeddingManager`: Specialized for code
  - Adds language context to embeddings
  - Handles code-specific preprocessing

- `DocumentEmbeddingManager`: Specialized for documentation
  - Adds title and section context

**Key Decisions**:
- Use Ollama's `nomic-embed-text` by default (good balance of quality and speed)
- Implement retry logic with exponential backoff
- Batch processing to reduce API calls

#### 2.2 Vector Store (`vector_store.py`)

**Purpose**: Persistent vector storage and similarity search

**Classes**:
- `VectorStore`: ChromaDB wrapper
  - `add_documents()`: Store documents with embeddings
  - `similarity_search()`: Find similar documents
  - `similarity_search_with_score()`: Search with relevance scores

- `CodeVectorStore`: Specialized for code
- `DocumentVectorStore`: Specialized for documentation

**Key Decisions**:
- ChromaDB for persistence and scalability
- Metadata filtering for targeted searches
- Separate collections for code vs. documentation

#### 2.3 Indexer (`indexer.py`)

**Purpose**: Index codebases and create searchable chunks

**Classes**:
- `CodebaseIndexer`: Main indexing engine
  - Language-aware text splitting
  - Incremental indexing (tracks file hashes)
  - Handles multiple file types

**Process**:
1. Scan directory for files matching patterns
2. Parse files (code or markdown)
3. Split into chunks (with overlap for context)
4. Generate embeddings
5. Store in vector store with metadata

**Key Decisions**:
- Chunk size: 1000 chars (balance between context and precision)
- Chunk overlap: 200 chars (maintain context across chunks)
- Language-specific splitters for better code chunking
- File hash tracking to avoid re-indexing unchanged files

### 3. Tools Layer (`src/tools/`)

LangChain tools that the agent can invoke.

#### Tool Categories

**Code Analysis Tools** (`code_analyzer.py`):
- `CodeAnalyzerTool`: Analyze code structure
- `FindFunctionTool`: Locate specific functions
- `FindClassTool`: Locate class definitions
- `CodeComplexityTool`: Calculate complexity metrics

**File Operations** (`file_reader.py`):
- `ReadFileTool`: Read file contents
- `ReadFileLinesTool`: Read specific line ranges
- `FileMetadataTool`: Get file metadata
- `ListDirectoryTool`: List directory contents
- `SearchFilesTool`: Find files by pattern

**Database Tools** (`database.py`):
- `DatabaseQueryTool`: Execute SQL queries
- `GetTableSchemaTool`: Inspect table schemas
- `ListTablesTool`: List all tables
- `GetSampleDataTool`: Get sample data
- `GetTableRelationshipsTool`: Find foreign keys

**Code Editing** (`code_editor.py`):
- `CodeEditorTool`: Create/edit files
- `RefactorCodeTool`: Suggest refactorings

**Documentation** (`documentation.py`):
- `SearchDocumentationTool`: Semantic doc search
- `ReadMarkdownTool`: Parse markdown files
- `GetMarkdownSectionTool`: Extract specific sections
- `SearchCodeInDocsTool`: Find code examples

#### Tool Design

Each tool:
- Inherits from LangChain's `BaseTool`
- Implements `_run()` method for synchronous execution
- Provides clear description for the LLM
- Returns structured results (often JSON)
- Handles errors gracefully

**Example**:
```python
class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Reads the contents of a file..."

    def _run(self, file_path: str) -> str:
        # Implementation
        pass
```

### 4. Parser Layer (`src/parsers/`)

Extracts structured information from different file types.

#### 4.1 Code Parser (`code_parser.py`)

**Purpose**: Parse source code and extract entities

**Classes**:
- `PythonParser`: AST-based Python parsing
  - Extracts functions, classes, imports
  - Calculates cyclomatic complexity
  - Extracts docstrings and type hints

- `GenericParser`: Regex-based parsing for other languages
- `CodeParser`: Dispatcher that selects appropriate parser

**Capabilities**:
- Function signatures with parameters and return types
- Class hierarchies and decorators
- Import dependencies
- Code metrics (lines, complexity)

#### 4.2 Log Parser (`log_parser.py`)

**Purpose**: Parse and analyze log files

**Features**:
- Multiple log format support (Python, Apache, Nginx, etc.)
- Auto-detection of log format
- Filtering by level, time range, pattern
- Error pattern analysis
- Statistics generation

#### 4.3 Markdown Parser (`markdown_parser.py`)

**Purpose**: Parse markdown documentation

**Features**:
- Hierarchical section extraction
- Link extraction
- Code block extraction (with language tags)
- List parsing
- YAML frontmatter support

### 5. Models Layer (`src/models/`)

Data models and configuration.

#### 5.1 Configuration (`config.py`)

**Purpose**: Load and validate configuration

**Classes**:
- `OllamaConfig`, `VectorStoreConfig`, etc.: Pydantic models
- `AppConfig`: Main configuration container
- `EnvSettings`: Environment variable loader

**Features**:
- YAML configuration files
- Environment variable override
- Validation using Pydantic
- Type safety

#### 5.2 Schemas (`schemas.py`)

**Purpose**: Data models for the system

**Key Models**:
- `CodeEntity`: Represents functions, classes
- `CodeAnalysis`: Complete file analysis
- `DatabaseSchema`: Table schema information
- `SearchResult`: Vector search results
- `AgentResponse`: Agent execution results
- `ConversationMessage`: Chat messages
- `IndexingStatus`: Indexing progress

### 6. Utilities Layer (`src/utils/`)

Helper functions and utilities.

#### 6.1 File Utils (`file_utils.py`)

- File discovery with gitignore support
- File metadata extraction
- Safe file reading/writing
- Project root detection
- Line counting and statistics

#### 6.2 Database Utils (`db_utils.py`)

- Database connection management
- Query execution with timing
- Schema introspection
- Query sanitization (security)
- Connection pooling

### 7. Memory Management (`src/agent/memory.py`)

Manages conversation and code context.

**Classes**:
- `AgentMemory`: Conversation history
  - Message storage with trimming
  - Context dictionary
  - LangChain memory adapter

- `CodeContextMemory`: Code-specific context
  - Recently accessed files
  - Analyzed functions cache
  - Recent queries

- `CombinedMemory`: Unified interface

**Key Decisions**:
- Keep last 50 messages (configurable)
- Separate code context from conversation
- Support for context variables

### 8. Prompts (`src/agent/prompts.py`)

Prompt templates for the LLM.

**Templates**:
- `SYSTEM_PROMPT_TEMPLATE`: Main ReAct prompt
- `CODE_ANALYSIS_PROMPT`: Code analysis template
- `CODE_GENERATION_PROMPT`: Code generation template
- `DEBUGGING_PROMPT`: Debugging assistance
- Additional specialized templates

**Design**:
- Use LangChain's `PromptTemplate`
- Clear instructions for tool usage
- Structured output format
- Context injection

## Data Flow

### Query Execution Flow

1. **User Input** → CLI receives query
2. **Agent Initialization** → Load config, initialize components
3. **Memory Retrieval** → Load conversation history
4. **ReAct Loop**:
   ```
   a. Thought: LLM reasons about the query
   b. Action: Decides which tool to use
   c. Action Input: Prepares tool input
   d. Observation: Tool executes and returns result
   e. Repeat until answer is found
   ```
5. **Response** → Format and display answer
6. **Memory Update** → Save conversation

### Indexing Flow

1. **Directory Scan** → Find files matching patterns
2. **File Processing**:
   - Read file content
   - Parse file (code/markdown)
   - Split into chunks
3. **Embedding Generation** → Create vectors
4. **Storage** → Save to ChromaDB with metadata
5. **Tracking** → Update file hash registry

### Search Flow

1. **Query Embedding** → Convert query to vector
2. **Similarity Search** → Find nearest vectors in ChromaDB
3. **Metadata Filtering** → Apply type/language filters
4. **Ranking** → Sort by similarity score
5. **Results** → Return top k results

## Design Decisions

### Why ReAct Pattern?

**Advantages**:
- Transparent reasoning process
- Can use multiple tools in sequence
- Self-correcting (can retry with different tools)
- Interpretable (can see thought process)

**Alternatives Considered**:
- Simple RAG: Less flexible, no multi-step reasoning
- Function calling: More rigid, less adaptive

### Why ChromaDB?

**Advantages**:
- Easy to use, minimal setup
- Good performance for medium-scale datasets
- Persistent storage
- Built-in filtering

**Alternatives Considered**:
- FAISS: Better performance but no persistence
- Pinecone: Cloud-based, additional cost
- Weaviate: More complex setup

### Why Ollama?

**Advantages**:
- Runs locally (privacy, no API costs)
- Good model selection (CodeLlama, DeepSeek)
- Easy to use
- Fast iteration

**Alternatives Considered**:
- OpenAI: Higher quality but costs money
- Together AI: Good but requires internet
- Local transformers: More control but harder to use

### Code Chunking Strategy

**Decision**: Use language-aware recursive splitting

**Rationale**:
- Preserves code structure (functions stay together)
- Overlap maintains context
- Language-specific separators (classes, functions)

**Alternatives**:
- Fixed-size chunks: Simpler but breaks semantics
- AST-based: More accurate but complex

## Performance Considerations

### Indexing Performance

**Bottlenecks**:
- File I/O (reading files)
- Embedding generation (API calls to Ollama)
- Vector insertion

**Optimizations**:
- Batch embedding generation (32 texts at a time)
- Incremental indexing (skip unchanged files)
- Parallel file processing (future enhancement)

### Query Performance

**Bottlenecks**:
- LLM inference time
- Vector similarity search
- Tool execution time

**Optimizations**:
- Limit search results (k=5 by default)
- Cache frequently accessed data
- Use faster models for simple queries

### Memory Usage

**Considerations**:
- Vector embeddings (768-dim floats)
- Conversation history (50 messages max)
- Loaded code chunks

**Optimizations**:
- Stream large results
- Trim old messages
- Clear cache periodically

## Security Considerations

### Database Security

- **SQL Injection Prevention**: Query sanitization
- **Read-only queries**: Only SELECT allowed
- **Credential Management**: Environment variables
- **Connection pooling**: Prevent resource exhaustion

### File System Security

- **Path Validation**: Prevent directory traversal
- **Size Limits**: Max file size checks
- **Gitignore Respect**: Don't index sensitive files
- **Read-only by default**: Write operations are explicit

### LLM Security

- **Prompt Injection**: Input validation
- **Output Sanitization**: Check for dangerous code
- **Rate Limiting**: Prevent abuse
- **Timeout**: Max execution time

## Extensibility

### Adding New Tools

1. Create tool class inheriting from `BaseTool`
2. Implement `_run()` method
3. Register in `_initialize_tools()`

### Adding New Parsers

1. Create parser class
2. Implement parsing logic
3. Register in appropriate tool or indexer

### Adding New LLM Providers

1. Create provider wrapper
2. Update configuration models
3. Modify agent initialization

### Adding New Vector Stores

1. Implement vector store interface
2. Update configuration
3. Modify indexer to use new store

## Testing Strategy

### Unit Tests
- Individual tool functionality
- Parser accuracy
- Utility functions

### Integration Tests
- End-to-end query flow
- Indexing process
- Database integration

### Performance Tests
- Indexing speed
- Query latency
- Memory usage

## Deployment Considerations

### Local Development
- Use Docker Compose for dependencies
- Hot reload for code changes
- Debug logging enabled

### Production
- Environment-based configuration
- Structured logging
- Error tracking (Sentry)
- Metrics collection
- Health checks

## Future Enhancements

### Short Term
- Web UI interface
- More language parsers (Go, Rust)
- Streaming responses
- Better error handling

### Medium Term
- Multi-repository support
- Git integration (blame, history)
- Test generation
- Code execution sandbox

### Long Term
- Autonomous code refactoring
- Automated PR reviews
- Multi-agent collaboration
- Plugin ecosystem

## Conclusion

The Coding Agent is designed as a modular, extensible system that combines LLM reasoning with practical developer tools. The architecture supports easy addition of new capabilities while maintaining clean separation of concerns.

Key strengths:
- **Modularity**: Easy to extend and modify
- **Flexibility**: ReAct pattern adapts to different tasks
- **Performance**: Efficient indexing and search
- **Safety**: Security considerations built-in

For more information, see the main [README](README.md).
