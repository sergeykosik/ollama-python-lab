"""Vector store management for the coding agent."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class VectorStore:
    """Wrapper for ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_manager: EmbeddingManager
    ):
        """Initialize vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the vector store
            embedding_manager: Embedding manager instance
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.embedding_manager = embedding_manager
        self.vector_store = self._initialize_store()

    def _initialize_store(self) -> Chroma:
        """Initialize ChromaDB vector store.

        Returns:
            Chroma instance
        """
        try:
            # Create persist directory if it doesn't exist
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            logger.info(f"Initializing vector store: {self.collection_name}")
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_manager.embeddings,
                persist_directory=str(self.persist_directory)
            )
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of Document objects
            batch_size: Batch size for adding documents

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        try:
            all_ids = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                ids = self.vector_store.add_documents(batch)
                all_ids.extend(ids)
                logger.debug(
                    f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}"
                )

            logger.info(f"Added {len(documents)} documents to vector store")
            return all_ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """Add texts to the vector store.

        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts
            batch_size: Batch size for adding texts

        Returns:
            List of document IDs
        """
        if not texts:
            return []

        try:
            all_ids = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = None
                if metadatas:
                    batch_metadatas = metadatas[i:i + batch_size]

                ids = self.vector_store.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )
                all_ids.extend(ids)
                logger.debug(
                    f"Added batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}"
                )

            logger.info(f"Added {len(texts)} texts to vector store")
            return all_ids
        except Exception as e:
            logger.error(f"Error adding texts: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of Document objects
        """
        try:
            if filter:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter
                )
            else:
                results = self.vector_store.similarity_search(query=query, k=k)

            logger.debug(f"Found {len(results)} results for query")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """Search for similar documents with scores.

        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of (Document, score) tuples
        """
        try:
            if filter:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter
                )
            else:
                results = self.vector_store.similarity_search_with_score(query=query, k=k)

            logger.debug(f"Found {len(results)} results with scores")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_store.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def clear(self) -> bool:
        """Clear all documents from the store.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_store.delete_collection()
            # Reinitialize
            self.vector_store = self._initialize_store()
            logger.info("Cleared vector store")
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.

        Returns:
            Dictionary with statistics
        """
        try:
            collection = self.vector_store._collection
            count = collection.count()

            return {
                'name': self.collection_name,
                'document_count': count,
                'persist_directory': str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def update_document(self, doc_id: str, document: Document) -> bool:
        """Update a document by ID.

        Args:
            doc_id: Document ID
            document: New Document object

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete old document
            self.delete([doc_id])
            # Add new document with same ID
            self.vector_store.add_documents([document], ids=[doc_id])
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False

    def search_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        k: int = 10
    ) -> List[Document]:
        """Search documents by metadata.

        Args:
            metadata_filter: Metadata filter criteria
            k: Maximum number of results

        Returns:
            List of matching Document objects
        """
        try:
            # Use a broad query to get documents, then filter
            results = self.vector_store.similarity_search(
                query="",
                k=k,
                filter=metadata_filter
            )
            return results
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []


class CodeVectorStore(VectorStore):
    """Specialized vector store for code."""

    def add_code_file(
        self,
        file_path: str,
        content: str,
        language: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Add a code file to the vector store.

        Args:
            file_path: Path to the code file
            content: File content
            language: Programming language
            metadata: Optional additional metadata

        Returns:
            Document ID or None if error
        """
        try:
            # Prepare metadata
            file_metadata = {
                'source': file_path,
                'type': 'code',
                'language': language,
                **(metadata or {})
            }

            # Create document
            doc = Document(
                page_content=content,
                metadata=file_metadata
            )

            # Add to store
            ids = self.add_documents([doc])
            return ids[0] if ids else None
        except Exception as e:
            logger.error(f"Error adding code file {file_path}: {e}")
            return None

    def search_code(
        self,
        query: str,
        language: Optional[str] = None,
        k: int = 5
    ) -> List[Document]:
        """Search for code.

        Args:
            query: Search query
            language: Optional language filter
            k: Number of results

        Returns:
            List of Document objects
        """
        filter_dict = {'type': 'code'}
        if language:
            filter_dict['language'] = language

        return self.similarity_search(query, k=k, filter=filter_dict)


class DocumentVectorStore(VectorStore):
    """Specialized vector store for documentation."""

    def add_documentation(
        self,
        file_path: str,
        content: str,
        title: Optional[str] = None,
        section: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Add documentation to the vector store.

        Args:
            file_path: Path to the documentation file
            content: Content
            title: Optional title
            section: Optional section name
            metadata: Optional additional metadata

        Returns:
            Document ID or None if error
        """
        try:
            # Prepare metadata
            doc_metadata = {
                'source': file_path,
                'type': 'documentation',
                **(metadata or {})
            }

            if title:
                doc_metadata['title'] = title
            if section:
                doc_metadata['section'] = section

            # Create document
            doc = Document(
                page_content=content,
                metadata=doc_metadata
            )

            # Add to store
            ids = self.add_documents([doc])
            return ids[0] if ids else None
        except Exception as e:
            logger.error(f"Error adding documentation {file_path}: {e}")
            return None

    def search_documentation(
        self,
        query: str,
        title: Optional[str] = None,
        k: int = 5
    ) -> List[Document]:
        """Search for documentation.

        Args:
            query: Search query
            title: Optional title filter
            k: Number of results

        Returns:
            List of Document objects
        """
        filter_dict = {'type': 'documentation'}
        if title:
            filter_dict['title'] = title

        return self.similarity_search(query, k=k, filter=filter_dict)
