"""
Configuration Module
Central configuration for Code Reviewer application
"""

import os
from typing import Dict, Any


class Config:
    """Application configuration"""

    # Ollama settings
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "codellama")
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # seconds

    # Embedding model settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = 384  # for all-MiniLM-L6-v2

    # Chunking settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # characters
    MAX_CHUNK_OVERLAP = int(os.getenv("MAX_CHUNK_OVERLAP", "50"))  # characters

    # Context retrieval settings
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))  # number of context files
    MAX_CONTEXT_FILES = int(os.getenv("MAX_CONTEXT_FILES", "20"))

    # File processing settings
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "1048576"))  # 1MB in bytes
    SUPPORTED_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
        '.go', '.rs', '.rb', '.php', '.cs', '.swift', '.kt', '.scala'
    }

    # Review settings
    REVIEW_DEPTHS = ["Quick", "Standard", "Deep", "Comprehensive"]
    DEFAULT_REVIEW_DEPTH = "Standard"

    FOCUS_AREAS = [
        "Code Quality",
        "Security",
        "Performance",
        "Best Practices",
        "Documentation",
        "Testing",
        "Architecture"
    ]
    DEFAULT_FOCUS_AREAS = ["Code Quality", "Security", "Best Practices"]

    # LLM generation settings
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
    LLM_TOP_K = int(os.getenv("LLM_TOP_K", "40"))

    # UI settings
    APP_TITLE = "Code Reviewer - Agentic AI"
    APP_ICON = "🔍"
    MAX_HISTORY_ITEMS = int(os.getenv("MAX_HISTORY_ITEMS", "100"))

    # File patterns
    DEFAULT_INCLUDE_PATTERNS = ["*.py", "*.js", "*.ts", "*.java", "*.go"]
    DEFAULT_EXCLUDE_PATTERNS = ["*test*", "*node_modules*", "*venv*", "*.min.js", "*__pycache__*"]

    # Caching
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # seconds

    @classmethod
    def get_ollama_config(cls) -> Dict[str, Any]:
        """Get Ollama configuration"""
        return {
            "host": cls.OLLAMA_HOST,
            "model": cls.OLLAMA_MODEL,
            "timeout": cls.OLLAMA_TIMEOUT,
            "temperature": cls.LLM_TEMPERATURE,
            "top_p": cls.LLM_TOP_P,
            "top_k": cls.LLM_TOP_K
        }

    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """Get embedding configuration"""
        return {
            "model": cls.EMBEDDING_MODEL,
            "dimension": cls.EMBEDDING_DIMENSION,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.MAX_CHUNK_OVERLAP
        }

    @classmethod
    def get_review_config(cls) -> Dict[str, Any]:
        """Get review configuration"""
        return {
            "depths": cls.REVIEW_DEPTHS,
            "default_depth": cls.DEFAULT_REVIEW_DEPTH,
            "focus_areas": cls.FOCUS_AREAS,
            "default_focus_areas": cls.DEFAULT_FOCUS_AREAS,
            "max_context_files": cls.MAX_CONTEXT_FILES,
            "default_top_k": cls.DEFAULT_TOP_K
        }

    @classmethod
    def is_valid_file(cls, filename: str) -> bool:
        """Check if file extension is supported"""
        from pathlib import Path
        ext = Path(filename).suffix.lower()
        return ext in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def get_language_from_extension(cls, extension: str) -> str:
        """Get language name from file extension"""
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala'
        }
        return language_map.get(extension.lower(), 'text')


# Create a default config instance
config = Config()
