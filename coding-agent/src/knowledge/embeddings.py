"""Embedding management for the coding agent."""

import logging
from typing import List, Optional
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manager for generating embeddings."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        use_ollama: bool = True,
        batch_size: int = 32
    ):
        """Initialize embedding manager.

        Args:
            model: Name of embedding model
            base_url: Ollama base URL
            use_ollama: Whether to use Ollama or HuggingFace
            batch_size: Batch size for embedding generation
        """
        self.model = model
        self.base_url = base_url
        self.use_ollama = use_ollama
        self.batch_size = batch_size
        self.embeddings = self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize embedding model.

        Returns:
            Embedding model instance
        """
        try:
            if self.use_ollama:
                logger.info(f"Initializing Ollama embeddings with model: {self.model}")
                return OllamaEmbeddings(
                    base_url=self.base_url,
                    model=self.model
                )
            else:
                logger.info(f"Initializing HuggingFace embeddings with model: {self.model}")
                return HuggingFaceEmbeddings(
                    model_name=self.model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'batch_size': self.batch_size}
                )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            Exception: If embedding generation fails after retries
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If embedding generation fails after retries
        """
        if not texts:
            return []

        try:
            # Process in batches
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                logger.debug(f"Embedded batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")

            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings.

        Returns:
            Embedding dimension
        """
        try:
            sample_embedding = self.embed_text("test")
            return len(sample_embedding)
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            # Default dimensions for common models
            defaults = {
                'nomic-embed-text': 768,
                'all-MiniLM-L6-v2': 384,
                'all-mpnet-base-v2': 768,
            }
            return defaults.get(self.model, 768)


class CodeEmbeddingManager(EmbeddingManager):
    """Specialized embedding manager for code."""

    def prepare_code_for_embedding(self, code: str, language: str) -> str:
        """Prepare code for embedding.

        Adds language context and cleans the code.

        Args:
            code: Source code
            language: Programming language

        Returns:
            Prepared code string
        """
        # Remove excessive whitespace
        lines = [line.rstrip() for line in code.splitlines()]
        code = '\n'.join(lines)

        # Add language context
        prepared = f"[Language: {language}]\n{code}"

        return prepared

    def embed_code(self, code: str, language: str) -> List[float]:
        """Generate embedding for code.

        Args:
            code: Source code
            language: Programming language

        Returns:
            Embedding vector
        """
        prepared_code = self.prepare_code_for_embedding(code, language)
        return self.embed_text(prepared_code)

    def embed_code_with_context(
        self,
        code: str,
        language: str,
        description: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> List[float]:
        """Generate embedding for code with additional context.

        Args:
            code: Source code
            language: Programming language
            description: Optional description or docstring
            file_path: Optional file path

        Returns:
            Embedding vector
        """
        # Build context
        context_parts = [f"[Language: {language}]"]

        if file_path:
            context_parts.append(f"[File: {file_path}]")

        if description:
            context_parts.append(f"[Description: {description}]")

        context_parts.append(code)

        full_text = '\n'.join(context_parts)
        return self.embed_text(full_text)


class DocumentEmbeddingManager(EmbeddingManager):
    """Specialized embedding manager for documents."""

    def prepare_document_for_embedding(
        self,
        content: str,
        title: Optional[str] = None,
        section: Optional[str] = None
    ) -> str:
        """Prepare document for embedding.

        Args:
            content: Document content
            title: Optional document title
            section: Optional section name

        Returns:
            Prepared document string
        """
        parts = []

        if title:
            parts.append(f"[Title: {title}]")

        if section:
            parts.append(f"[Section: {section}]")

        parts.append(content)

        return '\n'.join(parts)

    def embed_document(
        self,
        content: str,
        title: Optional[str] = None,
        section: Optional[str] = None
    ) -> List[float]:
        """Generate embedding for document.

        Args:
            content: Document content
            title: Optional document title
            section: Optional section name

        Returns:
            Embedding vector
        """
        prepared_doc = self.prepare_document_for_embedding(content, title, section)
        return self.embed_text(prepared_doc)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0-1)
    """
    import numpy as np

    v1 = np.array(vec1)
    v2 = np.array(vec2)

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return dot_product / (norm_v1 * norm_v2)
