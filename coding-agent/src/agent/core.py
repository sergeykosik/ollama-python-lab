"""Core agent implementation."""

import logging
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import Ollama
from langchain.tools import Tool

from ..models.config import AppConfig, get_db_credentials, EnvSettings
from ..models.schemas import AgentResponse
from ..knowledge.embeddings import EmbeddingManager
from ..knowledge.vector_store import VectorStore, CodeVectorStore, DocumentVectorStore
from ..knowledge.indexer import CodebaseIndexer
from ..utils.db_utils import DatabaseConnection
from ..tools.code_analyzer import (
    CodeAnalyzerTool,
    FindFunctionTool,
    FindClassTool,
    CodeComplexityTool
)
from ..tools.file_reader import (
    ReadFileTool,
    ReadFileLinesTool,
    FileMetadataTool,
    ListDirectoryTool,
    SearchFilesTool
)
from ..tools.database import (
    DatabaseQueryTool,
    GetTableSchemaTool,
    ListTablesTool,
    GetSampleDataTool,
    GetTableRelationshipsTool
)
from ..tools.code_editor import (
    CodeEditorTool,
    RefactorCodeTool
)
from ..tools.documentation import (
    SearchDocumentationTool,
    ReadMarkdownTool,
    GetMarkdownSectionTool,
    SearchCodeInDocsTool
)
from .prompts import get_system_prompt
from .memory import CombinedMemory
import time

logger = logging.getLogger(__name__)


class CodingAgent:
    """Main coding agent class."""

    def __init__(self, config: AppConfig, env_settings: Optional[EnvSettings] = None):
        """Initialize the coding agent.

        Args:
            config: Application configuration
            env_settings: Optional environment settings
        """
        self.config = config
        self.env_settings = env_settings or EnvSettings()
        self.memory = CombinedMemory(max_messages=config.agent.max_iterations)

        # Initialize components
        self.llm = self._initialize_llm()
        self.embedding_manager = self._initialize_embeddings()
        self.vector_store = self._initialize_vector_store()
        self.indexer = self._initialize_indexer()
        self.db_connection = self._initialize_database()

        # Initialize tools
        self.tools = self._initialize_tools()

        # Create agent
        self.agent_executor = self._create_agent()

        logger.info("Coding agent initialized successfully")

    def _initialize_llm(self) -> Ollama:
        """Initialize the LLM.

        Returns:
            Ollama LLM instance
        """
        logger.info(f"Initializing Ollama with model: {self.config.ollama.model}")
        return Ollama(
            base_url=self.config.ollama.base_url,
            model=self.config.ollama.model,
            temperature=self.config.ollama.temperature,
            timeout=self.config.ollama.timeout
        )

    def _initialize_embeddings(self) -> EmbeddingManager:
        """Initialize embeddings manager.

        Returns:
            EmbeddingManager instance
        """
        logger.info("Initializing embeddings manager")
        return EmbeddingManager(
            model=self.config.embeddings.model,
            base_url=self.config.ollama.base_url,
            use_ollama=True,
            batch_size=self.config.embeddings.batch_size
        )

    def _initialize_vector_store(self) -> VectorStore:
        """Initialize vector store.

        Returns:
            VectorStore instance
        """
        logger.info("Initializing vector store")
        return VectorStore(
            collection_name=self.config.vector_store.collection_name,
            persist_directory=self.config.vector_store.persist_directory,
            embedding_manager=self.embedding_manager
        )

    def _initialize_indexer(self) -> CodebaseIndexer:
        """Initialize codebase indexer.

        Returns:
            CodebaseIndexer instance
        """
        logger.info("Initializing codebase indexer")
        return CodebaseIndexer(
            vector_store=self.vector_store,
            chunk_size=self.config.vector_store.chunk_size,
            chunk_overlap=self.config.vector_store.chunk_overlap
        )

    def _initialize_database(self) -> Optional[DatabaseConnection]:
        """Initialize database connection.

        Returns:
            DatabaseConnection instance or None if connection fails
        """
        try:
            logger.info("Initializing database connection")
            credentials = get_db_credentials(self.env_settings)

            db_conn = DatabaseConnection(
                host=self.config.database.host,
                port=self.config.database.port,
                database=self.config.database.database,
                username=credentials['username'],
                password=credentials['password'],
                pool_size=self.config.database.pool_size,
                max_overflow=self.config.database.max_overflow
            )

            if db_conn.connect():
                logger.info("Database connected successfully")
                return db_conn
            else:
                logger.warning("Database connection failed")
                return None

        except Exception as e:
            logger.warning(f"Could not initialize database: {e}")
            return None

    def _initialize_tools(self) -> List[Tool]:
        """Initialize all agent tools.

        Returns:
            List of Tool objects
        """
        logger.info("Initializing agent tools")
        tools = []

        # Code analysis tools
        tools.extend([
            CodeAnalyzerTool(),
            FindFunctionTool(),
            FindClassTool(),
            CodeComplexityTool()
        ])

        # File reading tools
        tools.extend([
            ReadFileTool(),
            ReadFileLinesTool(),
            FileMetadataTool(),
            ListDirectoryTool(),
            SearchFilesTool()
        ])

        # Code editing tools
        tools.extend([
            CodeEditorTool(),
            RefactorCodeTool()
        ])

        # Documentation tools
        search_doc_tool = SearchDocumentationTool()
        search_doc_tool.indexer = self.indexer

        search_code_tool = SearchCodeInDocsTool()
        search_code_tool.indexer = self.indexer

        tools.extend([
            search_doc_tool,
            ReadMarkdownTool(),
            GetMarkdownSectionTool(),
            search_code_tool
        ])

        # Database tools (if available)
        if self.db_connection:
            query_tool = DatabaseQueryTool()
            query_tool.db_connection = self.db_connection

            schema_tool = GetTableSchemaTool()
            schema_tool.db_connection = self.db_connection

            list_tool = ListTablesTool()
            list_tool.db_connection = self.db_connection

            sample_tool = GetSampleDataTool()
            sample_tool.db_connection = self.db_connection

            rel_tool = GetTableRelationshipsTool()
            rel_tool.db_connection = self.db_connection

            tools.extend([query_tool, schema_tool, list_tool, sample_tool, rel_tool])

        logger.info(f"Initialized {len(tools)} tools")
        return tools

    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor.

        Returns:
            AgentExecutor instance
        """
        logger.info("Creating agent executor")

        # Get system prompt
        prompt = get_system_prompt()

        # Create ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )

        # Create executor
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.config.agent.verbose,
            max_iterations=self.config.agent.max_iterations,
            max_execution_time=self.config.agent.max_execution_time,
            handle_parsing_errors=self.config.agent.handle_parsing_errors,
            return_intermediate_steps=True
        )

        return executor

    def run(self, query: str) -> AgentResponse:
        """Run the agent with a query.

        Args:
            query: User query

        Returns:
            AgentResponse object
        """
        logger.info(f"Processing query: {query[:100]}...")

        # Add to memory
        self.memory.add_user_message(query)

        # Track time
        start_time = time.time()

        try:
            # Run agent
            result = self.agent_executor.invoke({"input": query})

            execution_time = (time.time() - start_time) * 1000  # Convert to ms

            # Extract response
            answer = result.get('output', '')
            intermediate_steps = result.get('intermediate_steps', [])

            # Add to memory
            self.memory.add_ai_message(answer)

            # Format intermediate steps
            formatted_steps = []
            for action, observation in intermediate_steps:
                formatted_steps.append({
                    'tool': action.tool,
                    'input': action.tool_input,
                    'output': str(observation)[:500]  # Limit length
                })

            # Create response
            response = AgentResponse(
                query=query,
                answer=answer,
                sources=[],  # Could extract from intermediate steps
                intermediate_steps=formatted_steps,
                execution_time_ms=execution_time,
                iterations=len(intermediate_steps)
            )

            logger.info(f"Query processed in {execution_time:.2f}ms with {len(intermediate_steps)} iterations")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            execution_time = (time.time() - start_time) * 1000

            # Create error response
            return AgentResponse(
                query=query,
                answer=f"Error: {str(e)}",
                sources=[],
                intermediate_steps=[],
                execution_time_ms=execution_time,
                iterations=0
            )

    def index_codebase(
        self,
        directory: str,
        file_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Index a codebase directory.

        Args:
            directory: Directory to index
            file_patterns: Optional file patterns to include
            ignore_patterns: Optional patterns to ignore

        Returns:
            Indexing status dictionary
        """
        from pathlib import Path

        if file_patterns is None:
            file_patterns = self.config.indexing.file_patterns

        if ignore_patterns is None:
            ignore_patterns = self.config.indexing.ignore_patterns

        logger.info(f"Starting indexing of directory: {directory}")

        status = self.indexer.index_directory(
            directory=Path(directory),
            file_patterns=file_patterns,
            ignore_patterns=ignore_patterns,
            max_file_size_mb=self.config.indexing.max_file_size_mb
        )

        result = {
            'total_files': status.total_files,
            'processed_files': status.processed_files,
            'failed_files': status.failed_files,
            'duration_seconds': status.duration_seconds,
            'errors': status.errors[:10]  # First 10 errors
        }

        logger.info(f"Indexing complete: {result}")
        return result

    def search_codebase(
        self,
        query: str,
        k: int = 5,
        code_only: bool = False,
        docs_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Search the indexed codebase.

        Args:
            query: Search query
            k: Number of results
            code_only: Search only code files
            docs_only: Search only documentation

        Returns:
            List of search results
        """
        filter_dict = {}
        if code_only:
            filter_dict['type'] = 'code'
        elif docs_only:
            filter_dict['type'] = 'documentation'

        results = self.indexer.search_with_scores(
            query=query,
            k=k,
            filter_dict=filter_dict if filter_dict else None
        )

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content[:500],
                'source': doc.metadata.get('source', 'Unknown'),
                'type': doc.metadata.get('type', 'Unknown'),
                'language': doc.metadata.get('language'),
                'score': round(score, 3)
            })

        return formatted_results

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'indexer': self.indexer.get_stats(),
            'vector_store': self.vector_store.get_collection_stats(),
            'memory': self.memory.get_full_context(),
            'tools_count': len(self.tools),
            'database_connected': self.db_connection is not None
        }

        return stats

    def clear_memory(self):
        """Clear agent memory."""
        self.memory.clear_all()
        logger.info("Memory cleared")

    def shutdown(self):
        """Shutdown the agent and cleanup resources."""
        logger.info("Shutting down agent")

        if self.db_connection:
            self.db_connection.disconnect()

        logger.info("Agent shutdown complete")
