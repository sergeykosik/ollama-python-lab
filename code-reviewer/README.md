# Code Reviewer - Advanced Agentic Application

An intelligent code review system powered by Ollama LLMs with context-aware analysis using vector embeddings and semantic search.

## Features

### 🎯 Core Capabilities

- **Multi-File Analysis**: Review multiple files simultaneously with cross-file insights
- **Context-Aware Reviews**: Index your entire codebase to provide relevant context for better analysis
- **Semantic Search**: Uses vector embeddings to automatically find related code
- **Customizable Review Depth**: Choose from Quick, Standard, Deep, or Comprehensive analysis levels
- **Focus Areas**: Target specific aspects like security, performance, or best practices
- **Auto-Suggest Fixes**: Get actionable suggestions with code examples
- **Review History**: Track all your reviews with full reports and metrics

### 🔍 Advanced Features

1. **Codebase Indexing**
   - Index your entire project for context-aware reviews
   - Semantic search using sentence transformers
   - Efficient vector storage and retrieval
   - Support for multiple programming languages

2. **Intelligent Analysis**
   - Issue detection with severity levels (Critical, Warning, Suggestion)
   - Security vulnerability identification
   - Performance optimization recommendations
   - Best practices validation
   - Code quality metrics

3. **Flexible Configuration**
   - Adjustable review depth
   - Customizable focus areas
   - Configurable context retrieval
   - Multiple Ollama model support

## Architecture

```
code-reviewer/
├── app.py                 # Main Streamlit application
├── code_analyzer.py       # LLM-based code analysis engine
├── context_manager.py     # Codebase indexing and context retrieval
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Components

1. **app.py**: Streamlit web interface with tabs for review, context management, and history
2. **code_analyzer.py**: Handles communication with Ollama and structures LLM analysis
3. **context_manager.py**: Manages codebase indexing using sentence transformers and vector search
4. **utils.py**: Helper functions for file operations and data processing

## Installation

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally
- At least one code-capable model (e.g., codellama, deepseek-coder, or starcoder)

### Setup Steps

1. **Install Ollama**
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh

   # Or visit https://ollama.com for other platforms
   ```

2. **Pull a Code Model**
   ```bash
   ollama pull codellama
   # or
   ollama pull deepseek-coder
   ```

3. **Install Python Dependencies**
   ```bash
   cd code-reviewer
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

5. **Access the UI**
   - Open your browser to `http://localhost:8501`

## Usage

### Quick Start

1. **Start Ollama** (if not already running)
   ```bash
   ollama serve
   ```

2. **Launch Code Reviewer**
   ```bash
   streamlit run app.py
   ```

3. **Index Your Codebase** (Optional but Recommended)
   - Navigate to the "Context Manager" tab
   - Enter your project directory path
   - Configure file patterns (e.g., `*.py,*.js,*.ts`)
   - Click "Index Codebase"

4. **Review Code**
   - Go to the "Code Review" tab
   - Upload files to review
   - Configure review settings (depth, focus areas)
   - Enable "Use Codebase Context" for enhanced analysis
   - Click "Start Review"

### Configuration

#### Ollama Settings
- **Host**: Default is `http://localhost:11434`
- **Model**: Choose from your installed models (e.g., `codellama`, `deepseek-coder`)

#### Review Settings
- **Review Depth**:
  - Quick: Fast overview of major issues
  - Standard: Balanced analysis (recommended)
  - Deep: Thorough examination
  - Comprehensive: Exhaustive analysis with edge cases

- **Focus Areas**: Select specific aspects to analyze
  - Code Quality
  - Security
  - Performance
  - Best Practices
  - Documentation
  - Testing
  - Architecture

- **Max Context Files**: Number of related files to include (1-20)

### Example Workflow

```python
# 1. Index your project
# Navigate to Context Manager tab
# Enter path: /home/user/my-project
# Patterns: *.py,*.js
# Excludes: *test*,*node_modules*

# 2. Upload files for review
# Upload: main.py, api.py, utils.py

# 3. Configure review
# Depth: Deep
# Focus: Security, Performance, Best Practices
# Context: Enabled

# 4. Review results
# - Overall feedback
# - Per-file issues with severity
# - Suggested fixes with code examples
# - Positive aspects identified
```

## Advanced Usage

### Custom Models

You can use any Ollama model that supports code analysis:

```bash
# Pull additional models
ollama pull deepseek-coder
ollama pull starcoder
ollama pull wizard-coder

# Use in the app via the sidebar settings
```

### Context Retrieval

The context manager uses sentence transformers to create semantic embeddings of your codebase:

1. Files are chunked into smaller segments
2. Each chunk is embedded using `all-MiniLM-L6-v2`
3. During review, relevant chunks are retrieved via cosine similarity
4. Top-K most relevant files are provided as context to the LLM

This enables the LLM to understand:
- Related functions and classes
- Similar patterns in your codebase
- Project conventions and standards
- Architectural context

### Review History

All reviews are stored in session state and can be:
- Viewed in the History tab
- Exported as JSON
- Used to track improvements over time

## Supported Languages

- Python (`.py`)
- JavaScript (`.js`, `.jsx`)
- TypeScript (`.ts`, `.tsx`)
- Java (`.java`)
- C/C++ (`.c`, `.cpp`, `.h`, `.hpp`)
- Go (`.go`)
- Rust (`.rs`)
- Ruby (`.rb`)
- PHP (`.php`)

## Performance Tips

1. **Index Once, Review Many**: Index your codebase once and reuse for multiple reviews
2. **Appropriate Depth**: Use "Quick" or "Standard" for routine reviews, "Deep" for critical code
3. **Focused Analysis**: Select specific focus areas rather than all options
4. **Limit Context**: Use 3-5 context files for optimal performance
5. **Model Selection**: Smaller models (codellama) are faster, larger models more thorough

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Model Not Found
```bash
# List installed models
ollama list

# Pull the required model
ollama pull codellama
```

### Slow Performance
- Reduce review depth to "Quick" or "Standard"
- Limit max context files
- Use a smaller/faster model
- Avoid indexing very large codebases (>10K files)

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## Technical Details

### Vector Embeddings
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding Size: 384 dimensions
- Similarity: Cosine similarity
- Chunk Size: 500 characters

### LLM Integration
- API: Ollama HTTP API
- Temperature: 0.3 (consistent analysis)
- Timeout: 120 seconds
- Streaming: Disabled for structured output

### Data Storage
- Session State: In-memory during app runtime
- History: Optional export to JSON
- Index: In-memory vector store (not persisted)

## Future Enhancements

- [ ] Persistent vector database (ChromaDB/FAISS)
- [ ] GitHub integration for PR reviews
- [ ] Automated fix application
- [ ] Custom rule definitions
- [ ] Batch processing for entire directories
- [ ] API mode for CI/CD integration
- [ ] Multi-model consensus analysis
- [ ] Code diff analysis

## Contributing

Contributions are welcome! Areas of interest:
- Additional language support
- UI/UX improvements
- Performance optimizations
- New analysis features
- Documentation

## License

MIT License - See LICENSE file for details

## Credits

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [Ollama](https://ollama.com/) - Local LLM inference
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [NumPy](https://numpy.org/) - Numerical computing

---

**Note**: This is an agentic application designed for development assistance. Always review and validate suggestions before applying them to production code.
