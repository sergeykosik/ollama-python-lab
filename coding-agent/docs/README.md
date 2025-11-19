# Coding Agent

An intelligent AI-powered coding assistant built with LangChain and Ollama. This agent helps developers understand codebases, analyze code, query databases, search documentation, and generate code.

## Features

### Core Capabilities

- **Semantic Code Search**: Index and search your entire codebase using vector embeddings
- **Code Analysis**: Analyze code structure, complexity, and patterns
- **Database Integration**: Query MySQL databases and inspect schemas
- **Documentation Search**: Search and analyze markdown documentation
- **Code Generation**: Generate and edit code with context awareness
- **Interactive Chat**: Converse with the agent to solve coding problems

### Advanced Features

- **ReAct Agent Pattern**: Sophisticated reasoning and action loop
- **Multi-Language Support**: Python, JavaScript, TypeScript, Java, C#, and more
- **Conversation Memory**: Maintains context across interactions
- **Incremental Indexing**: Efficiently update the knowledge base
- **Tool Orchestration**: Intelligently combines multiple tools to solve problems

## Architecture

```
coding-agent/
├── src/
│   ├── agent/          # Core agent logic (ReAct agent, prompts, memory)
│   ├── knowledge/      # Vector store, embeddings, indexing
│   ├── tools/          # LangChain tools for various operations
│   ├── parsers/        # Code, log, and markdown parsers
│   ├── models/         # Configuration and data models
│   └── utils/          # Utility functions
├── config/             # Configuration files
├── data/               # Vector store and cache
├── docs/               # Documentation
└── main.py             # CLI entry point
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Ollama running locally (http://localhost:11434)
- MySQL database (optional, for database tools)

### Setup

1. **Clone or navigate to the repository**:
   ```bash
   cd coding-agent
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Ollama models**:
   ```bash
   # Pull the code generation model
   ollama pull codellama:13b

   # Pull the embedding model
   ollama pull nomic-embed-text
   ```

5. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

6. **Configure the agent**:
   Edit `config/agent_config.yaml` to customize settings.

## Quick Start

### 1. Index Your Codebase

```bash
python main.py index /path/to/your/project
```

This will:
- Scan all files matching configured patterns
- Parse code and documentation
- Generate embeddings
- Store in ChromaDB vector store

### 2. Search the Codebase

```bash
python main.py search "authentication logic"
```

### 3. Ask Questions

```bash
python main.py ask "How does the authentication system work?"
```

### 4. Interactive Mode

```bash
python main.py interactive
```

This starts an interactive session where you can have a conversation with the agent.

### 5. Analyze Code

```bash
python main.py analyze src/main.py
```

## Usage Examples

### Semantic Code Search

```bash
# Search for authentication-related code
python main.py search "user authentication" --code-only -k 10

# Search documentation
python main.py search "API documentation" --docs-only
```

### Code Analysis

```bash
# Ask about code structure
python main.py ask "What are the main classes in the authentication module?"

# Request code review
python main.py ask "Review the security of the login function in auth.py"

# Understand dependencies
python main.py ask "What external libraries does this project use?"
```

### Database Queries

```bash
# Ask about database schema
python main.py ask "Show me the schema for the users table"

# Query data
python main.py ask "How many active users are in the database?"

# Understand relationships
python main.py ask "What tables are related to the orders table?"
```

### Code Generation

```bash
# Generate new code
python main.py ask "Create a function to validate email addresses in Python"

# Request refactoring
python main.py ask "Suggest refactoring for the process_data function to improve readability"
```

### Interactive Session Examples

```
You: Find all API endpoints that handle user data

Agent: I'll search the codebase for API endpoints related to user data...
[Shows results with file paths and line numbers]

You: Show me the implementation of the /api/users endpoint

Agent: [Displays the code and explains the implementation]

You: Are there any security issues with this endpoint?

Agent: [Analyzes the code and provides security recommendations]
```

## CLI Commands

### `index`
Index a codebase directory.

```bash
python main.py index <directory> [OPTIONS]

Options:
  --patterns TEXT   File patterns to include (can be repeated)
  --ignore TEXT     Patterns to ignore (can be repeated)
```

### `search`
Search the indexed codebase.

```bash
python main.py search <query> [OPTIONS]

Options:
  --code-only      Search only code files
  --docs-only      Search only documentation
  -k INTEGER       Number of results (default: 5)
```

### `ask`
Ask the agent a question.

```bash
python main.py ask <query> [OPTIONS]

Options:
  --json-output    Output as JSON
```

### `interactive`
Start interactive mode.

```bash
python main.py interactive

Commands in interactive mode:
  - exit/quit: Exit the session
  - clear: Clear conversation history
  - stats: Show agent statistics
```

### `analyze`
Analyze a code file.

```bash
python main.py analyze <file_path>
```

### `stats`
Show agent statistics.

```bash
python main.py stats
```

## Configuration

### Environment Variables (.env)

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=codellama:13b

# Database
DB_HOST=localhost
DB_PORT=3306
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password

# Vector Store
VECTOR_STORE_DIR=./data/vector_store
COLLECTION_NAME=codebase

# Agent
MAX_ITERATIONS=15
TEMPERATURE=0.1
```

### Agent Configuration (config/agent_config.yaml)

Key settings:
- **ollama**: LLM configuration
- **vector_store**: Vector database settings
- **embeddings**: Embedding model configuration
- **indexing**: File patterns and ignore rules
- **agent**: Agent behavior settings

See `config/agent_config.yaml` for full configuration options.

## Available Tools

The agent has access to the following tools:

### Code Analysis Tools
- `analyze_code`: Analyze code structure and metrics
- `find_function`: Find and display specific functions
- `find_class`: Find and display class definitions
- `analyze_complexity`: Calculate code complexity

### File Operations Tools
- `read_file`: Read file contents
- `read_file_lines`: Read specific lines from a file
- `file_metadata`: Get file metadata
- `list_directory`: List directory contents
- `search_files`: Search for files by pattern

### Database Tools
- `query_database`: Execute SQL SELECT queries
- `get_table_schema`: Get table schema information
- `list_tables`: List all database tables
- `get_sample_data`: Get sample rows from a table
- `get_table_relationships`: Get foreign key relationships

### Code Editing Tools
- `edit_code`: Create or edit code files
- `refactor_code`: Get refactoring suggestions

### Documentation Tools
- `search_documentation`: Search documentation semantically
- `read_markdown`: Read and parse markdown files
- `get_markdown_section`: Extract specific sections from markdown
- `search_code_in_docs`: Find code examples in documentation

## Performance Tips

1. **Indexing**: Index only relevant directories to improve performance
2. **Chunk Size**: Adjust `chunk_size` in config for better results (larger = more context, smaller = more precise)
3. **Model Selection**:
   - CodeLlama 13B: Best balance of quality and speed
   - DeepSeek Coder: Higher quality, slower
   - Llama 3: Faster, good for simple queries
4. **Database**: Use connection pooling for better performance
5. **Caching**: The vector store is persisted, so reindexing is only needed when code changes

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start Ollama
ollama serve
```

### Vector Store Issues
```bash
# Clear and rebuild the vector store
rm -rf data/vector_store
python main.py index <directory>
```

### Database Connection Issues
- Verify database credentials in `.env`
- Ensure MySQL server is running
- Check network connectivity

### Memory Issues
- Reduce `chunk_size` in configuration
- Index fewer files at once
- Reduce `max_messages` in agent configuration

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/

# Check types
mypy src/

# Lint
flake8 src/
```

### Adding New Tools

1. Create a new tool class inheriting from `BaseTool`
2. Implement `_run()` method
3. Add tool to `_initialize_tools()` in `src/agent/core.py`

Example:
```python
from langchain.tools import BaseTool

class MyCustomTool(BaseTool):
    name = "my_tool"
    description = "Description of what the tool does"

    def _run(self, input_str: str) -> str:
        # Implementation
        return result
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Ollama](https://ollama.ai/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)

## Support

For issues and questions:
- GitHub Issues: [Report a bug](https://github.com/yourusername/coding-agent/issues)
- Documentation: [Full docs](./docs/architecture.md)

## Roadmap

- [ ] Support for more LLM providers (OpenAI, Anthropic)
- [ ] Web UI interface
- [ ] Git integration for code history analysis
- [ ] Test generation capabilities
- [ ] Multi-repository support
- [ ] Plugin system for custom tools
- [ ] Code execution sandbox
- [ ] Automated code review workflows
