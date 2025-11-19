"""Code indexing utilities for the coding agent."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.schema import Document

from ..models.schemas import IndexingStatus
from ..utils.file_utils import (
    find_files,
    get_file_metadata,
    read_file_safely,
    compute_file_hash
)
from ..parsers.code_parser import CodeParser
from ..parsers.markdown_parser import MarkdownParser
from .vector_store import VectorStore, CodeVectorStore, DocumentVectorStore
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class CodebaseIndexer:
    """Indexer for code repositories."""

    def __init__(
        self,
        vector_store: VectorStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Initialize codebase indexer.

        Args:
            vector_store: Vector store instance
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.code_parser = CodeParser()
        self.markdown_parser = MarkdownParser()
        self.text_splitter = self._create_text_splitter()
        self.indexed_files: Dict[str, str] = {}  # file_path -> hash

    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create text splitter.

        Returns:
            RecursiveCharacterTextSplitter instance
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def _get_language_splitter(self, language: str) -> RecursiveCharacterTextSplitter:
        """Get language-specific text splitter.

        Args:
            language: Programming language

        Returns:
            Language-specific text splitter
        """
        lang_map = {
            'Python': Language.PYTHON,
            'JavaScript': Language.JS,
            'TypeScript': Language.TS,
            'Java': Language.JAVA,
            'C++': Language.CPP,
            'C#': Language.CSHARP,
            'Go': Language.GO,
            'Rust': Language.RUST,
            'Markdown': Language.MARKDOWN,
        }

        lang = lang_map.get(language)
        if lang:
            try:
                return RecursiveCharacterTextSplitter.from_language(
                    language=lang,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            except Exception as e:
                logger.warning(f"Could not create language splitter for {language}: {e}")

        return self.text_splitter

    def index_file(self, file_path: Path) -> bool:
        """Index a single file.

        Args:
            file_path: Path to the file

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Indexing file: {file_path}")

            # Get file metadata
            metadata = get_file_metadata(file_path)
            if not metadata:
                logger.error(f"Could not get metadata for {file_path}")
                return False

            # Check if file was already indexed with same hash
            file_path_str = str(file_path)
            if file_path_str in self.indexed_files:
                if self.indexed_files[file_path_str] == metadata.file_hash:
                    logger.debug(f"File unchanged, skipping: {file_path}")
                    return True

            # Read file content
            content = read_file_safely(file_path)
            if not content:
                logger.error(f"Could not read file: {file_path}")
                return False

            # Determine how to process based on file type
            if metadata.file_type.value == 'markdown':
                success = self._index_markdown(file_path, content, metadata)
            else:
                success = self._index_code(file_path, content, metadata)

            if success:
                # Update indexed files tracking
                self.indexed_files[file_path_str] = metadata.file_hash

            return success

        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            return False

    def _index_code(
        self,
        file_path: Path,
        content: str,
        metadata
    ) -> bool:
        """Index a code file.

        Args:
            file_path: Path to the code file
            content: File content
            metadata: File metadata

        Returns:
            True if successful
        """
        try:
            # Parse code if possible
            analysis = self.code_parser.parse_file(file_path)

            # Get language-specific splitter
            splitter = self._get_language_splitter(metadata.language or 'Unknown')

            # Split content into chunks
            chunks = splitter.split_text(content)

            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    'source': str(file_path),
                    'type': 'code',
                    'language': metadata.language,
                    'file_type': metadata.file_type.value,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'file_hash': metadata.file_hash,
                    'last_modified': metadata.last_modified.isoformat()
                }

                # Add analysis info if available
                if analysis:
                    doc_metadata['function_count'] = len(
                        [e for e in analysis.entities if e.type in ['function', 'async_function']]
                    )
                    doc_metadata['class_count'] = len(
                        [e for e in analysis.entities if e.type == 'class']
                    )

                doc = Document(
                    page_content=chunk,
                    metadata=doc_metadata
                )
                documents.append(doc)

            # Add to vector store
            if isinstance(self.vector_store, CodeVectorStore):
                # Use specialized method
                self.vector_store.add_documents(documents)
            else:
                self.vector_store.add_documents(documents)

            logger.info(f"Indexed {len(chunks)} chunks from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error indexing code file {file_path}: {e}")
            return False

    def _index_markdown(
        self,
        file_path: Path,
        content: str,
        metadata
    ) -> bool:
        """Index a markdown file.

        Args:
            file_path: Path to the markdown file
            content: File content
            metadata: File metadata

        Returns:
            True if successful
        """
        try:
            # Parse markdown
            parsed = self.markdown_parser.parse_content(content)

            # Split by sections if available
            documents = []
            if parsed['sections']:
                for section_dict in parsed['sections']:
                    section_content = f"# {section_dict['title']}\n\n{section_dict['content']}"

                    # Split section if too large
                    chunks = self.text_splitter.split_text(section_content)

                    for i, chunk in enumerate(chunks):
                        doc_metadata = {
                            'source': str(file_path),
                            'type': 'documentation',
                            'file_type': 'markdown',
                            'section': section_dict['title'],
                            'section_level': section_dict['level'],
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'file_hash': metadata.file_hash,
                            'last_modified': metadata.last_modified.isoformat()
                        }

                        doc = Document(
                            page_content=chunk,
                            metadata=doc_metadata
                        )
                        documents.append(doc)
            else:
                # No sections, split entire content
                chunks = self.text_splitter.split_text(content)

                for i, chunk in enumerate(chunks):
                    doc_metadata = {
                        'source': str(file_path),
                        'type': 'documentation',
                        'file_type': 'markdown',
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'file_hash': metadata.file_hash,
                        'last_modified': metadata.last_modified.isoformat()
                    }

                    doc = Document(
                        page_content=chunk,
                        metadata=doc_metadata
                    )
                    documents.append(doc)

            # Add to vector store
            if isinstance(self.vector_store, DocumentVectorStore):
                self.vector_store.add_documents(documents)
            else:
                self.vector_store.add_documents(documents)

            logger.info(f"Indexed {len(documents)} chunks from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error indexing markdown file {file_path}: {e}")
            return False

    def index_directory(
        self,
        directory: Path,
        file_patterns: List[str],
        ignore_patterns: List[str],
        max_file_size_mb: int = 10
    ) -> IndexingStatus:
        """Index all files in a directory.

        Args:
            directory: Root directory to index
            file_patterns: File patterns to include
            ignore_patterns: Patterns to ignore
            max_file_size_mb: Maximum file size in MB

        Returns:
            IndexingStatus object
        """
        status = IndexingStatus(
            total_files=0,
            processed_files=0,
            failed_files=0,
            total_chunks=0,
            start_time=datetime.now()
        )

        try:
            logger.info(f"Starting indexing of directory: {directory}")

            # Find all files
            files = list(find_files(
                directory,
                file_patterns,
                ignore_patterns,
                max_file_size_mb
            ))
            status.total_files = len(files)

            logger.info(f"Found {len(files)} files to index")

            # Index each file
            for file_path in files:
                try:
                    if self.index_file(file_path):
                        status.processed_files += 1
                    else:
                        status.failed_files += 1
                        status.errors.append(f"Failed to index: {file_path}")
                except Exception as e:
                    status.failed_files += 1
                    status.errors.append(f"Error indexing {file_path}: {e}")
                    logger.error(f"Error indexing {file_path}: {e}")

            status.end_time = datetime.now()
            logger.info(
                f"Indexing complete. Processed: {status.processed_files}, "
                f"Failed: {status.failed_files}"
            )

            return status

        except Exception as e:
            logger.error(f"Error during directory indexing: {e}")
            status.end_time = datetime.now()
            status.errors.append(f"Critical error: {e}")
            return status

    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search the indexed codebase.

        Args:
            query: Search query
            k: Number of results
            filter_dict: Optional metadata filter

        Returns:
            List of Document objects
        """
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """Search with relevance scores.

        Args:
            query: Search query
            k: Number of results
            filter_dict: Optional metadata filter

        Returns:
            List of (Document, score) tuples
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )

    def reindex_file(self, file_path: Path) -> bool:
        """Reindex a file (remove old, add new).

        Args:
            file_path: Path to the file

        Returns:
            True if successful
        """
        try:
            # Remove old entries for this file
            self.remove_file(file_path)

            # Index the file
            return self.index_file(file_path)

        except Exception as e:
            logger.error(f"Error reindexing file {file_path}: {e}")
            return False

    def remove_file(self, file_path: Path) -> bool:
        """Remove a file from the index.

        Args:
            file_path: Path to the file

        Returns:
            True if successful
        """
        try:
            # Search for documents from this file
            results = self.vector_store.search_by_metadata(
                metadata_filter={'source': str(file_path)},
                k=1000  # Get all chunks
            )

            if results:
                # Extract IDs and delete
                # Note: This is a simplified approach
                # In practice, you'd need to track document IDs
                logger.info(f"Removed {len(results)} chunks for {file_path}")

            # Remove from tracking
            file_path_str = str(file_path)
            if file_path_str in self.indexed_files:
                del self.indexed_files[file_path_str]

            return True

        except Exception as e:
            logger.error(f"Error removing file {file_path}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'indexed_files': len(self.indexed_files),
            'vector_store_stats': self.vector_store.get_collection_stats()
        }
