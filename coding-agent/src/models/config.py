"""Configuration models and loaders for the coding agent."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaConfig(BaseModel):
    """Configuration for Ollama LLM."""
    base_url: str = Field(default="http://localhost:11434")
    model: str = Field(default="codellama:13b")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    context_window: int = Field(default=8192, gt=0)
    timeout: int = Field(default=120, gt=0)


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    type: str = Field(default="chromadb")
    persist_directory: str = Field(default="./data/vector_store")
    collection_name: str = Field(default="codebase")
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get('chunk_size', 1000)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class EmbeddingsConfig(BaseModel):
    """Configuration for embeddings."""
    model: str = Field(default="nomic-embed-text")
    batch_size: int = Field(default=32, gt=0)


class DatabaseConfig(BaseModel):
    """Configuration for database connection."""
    host: str = Field(default="localhost")
    port: int = Field(default=3306, gt=0, le=65535)
    database: str = Field(default="your_db")
    pool_size: int = Field(default=5, gt=0)
    max_overflow: int = Field(default=10, ge=0)


class IndexingConfig(BaseModel):
    """Configuration for code indexing."""
    watch_directories: List[str] = Field(default_factory=lambda: ["./src", "./docs"])
    file_patterns: List[str] = Field(
        default_factory=lambda: ["*.py", "*.js", "*.ts", "*.java", "*.cs", "*.md"]
    )
    ignore_patterns: List[str] = Field(
        default_factory=lambda: ["node_modules/**", "__pycache__/**", ".git/**"]
    )
    max_file_size_mb: int = Field(default=10, gt=0)


class AgentConfig(BaseModel):
    """Configuration for agent behavior."""
    max_iterations: int = Field(default=15, gt=0)
    max_execution_time: int = Field(default=300, gt=0)
    verbose: bool = Field(default=True)
    handle_parsing_errors: bool = Field(default=True)


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO")
    file: str = Field(default="./logs/agent.log")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    max_bytes: int = Field(default=10485760)  # 10MB
    backup_count: int = Field(default=5, ge=0)

    @field_validator('level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"log level must be one of {valid_levels}")
        return v


class AppConfig(BaseModel):
    """Main application configuration."""
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class EnvSettings(BaseSettings):
    """Environment variables configuration."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "codellama:13b"

    # Database
    db_host: str = "localhost"
    db_port: int = 3306
    db_name: str = "your_database"
    db_user: str = "root"
    db_password: str = ""

    # Vector Store
    vector_store_dir: str = "./data/vector_store"
    collection_name: str = "codebase"

    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/agent.log"

    # Agent
    max_iterations: int = 15
    temperature: float = 0.1
    context_window: int = 8192


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge_configs(yaml_config: Dict[str, Any], env_settings: EnvSettings) -> Dict[str, Any]:
    """Merge YAML config with environment variables.

    Environment variables take precedence over YAML config.

    Args:
        yaml_config: Configuration from YAML file
        env_settings: Configuration from environment variables

    Returns:
        Merged configuration dictionary
    """
    # Start with YAML config
    merged = yaml_config.copy()

    # Override with environment variables
    if 'ollama' not in merged:
        merged['ollama'] = {}
    merged['ollama']['base_url'] = env_settings.ollama_base_url
    merged['ollama']['model'] = env_settings.ollama_model
    merged['ollama']['temperature'] = env_settings.temperature
    merged['ollama']['context_window'] = env_settings.context_window

    if 'database' not in merged:
        merged['database'] = {}
    merged['database']['host'] = env_settings.db_host
    merged['database']['port'] = env_settings.db_port
    merged['database']['database'] = env_settings.db_name

    if 'vector_store' not in merged:
        merged['vector_store'] = {}
    merged['vector_store']['persist_directory'] = env_settings.vector_store_dir
    merged['vector_store']['collection_name'] = env_settings.collection_name

    if 'logging' not in merged:
        merged['logging'] = {}
    merged['logging']['level'] = env_settings.log_level
    merged['logging']['file'] = env_settings.log_file

    if 'agent' not in merged:
        merged['agent'] = {}
    merged['agent']['max_iterations'] = env_settings.max_iterations

    return merged


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load and validate application configuration.

    Loads configuration from YAML file and environment variables.
    Environment variables take precedence over YAML config.

    Args:
        config_path: Path to YAML configuration file.
                    Defaults to 'config/agent_config.yaml'

    Returns:
        Validated AppConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If configuration is invalid
    """
    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "agent_config.yaml"

    # Load environment variables
    env_settings = EnvSettings()

    # Load YAML config
    try:
        yaml_config = load_yaml_config(config_path)
    except FileNotFoundError:
        # If YAML doesn't exist, use defaults merged with env vars
        yaml_config = {}

    # Merge configurations
    merged_config = merge_configs(yaml_config, env_settings)

    # Create and validate AppConfig
    return AppConfig(**merged_config)


def get_db_credentials(env_settings: Optional[EnvSettings] = None) -> Dict[str, str]:
    """Get database credentials from environment.

    Args:
        env_settings: Optional EnvSettings object. If not provided, will load from .env

    Returns:
        Dictionary with username and password
    """
    if env_settings is None:
        env_settings = EnvSettings()

    return {
        "username": env_settings.db_user,
        "password": env_settings.db_password
    }
