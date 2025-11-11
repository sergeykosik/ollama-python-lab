"""
Context Manager Module
Handles codebase indexing and context retrieval using vector embeddings
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict


class ContextManager:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize Context Manager with embedding model

        Args:
            model_name: Sentence transformer model for embeddings
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.indexed_files = {}  # file_path -> {content, embedding, metadata}
        self.embeddings_matrix = None
        self.file_paths = []

    def index_codebase(self, root_path: str, include_patterns: List[str],
                       exclude_patterns: List[str]) -> int:
        """
        Index a codebase directory

        Args:
            root_path: Root directory to index
            include_patterns: File patterns to include (e.g., *.py)
            exclude_patterns: Patterns to exclude (e.g., *test*)

        Returns:
            Number of files indexed
        """
        self.indexed_files = {}
        files_to_index = []

        # Walk through directory
        for root, dirs, files in os.walk(root_path):
            # Filter directories to exclude
            dirs[:] = [d for d in dirs if not self._matches_patterns(d, exclude_patterns)]

            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_path)

                # Check if file matches include patterns and not exclude patterns
                if (self._matches_patterns(file, [p.replace('*', '') for p in include_patterns]) and
                    not self._matches_patterns(rel_path, exclude_patterns)):
                    files_to_index.append(file_path)

        # Index files
        for file_path in files_to_index:
            try:
                self._index_file(file_path, root_path)
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
                continue

        # Build embeddings matrix for efficient search
        self._build_embeddings_matrix()

        return len(self.indexed_files)

    def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any pattern"""
        if not patterns:
            return False

        text_lower = text.lower()
        for pattern in patterns:
            pattern_lower = pattern.lower().replace('*', '')
            if pattern_lower in text_lower:
                return True
        return False

    def _index_file(self, file_path: str, root_path: str):
        """Index a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Skip very large files (> 1MB)
            if len(content) > 1_000_000:
                return

            # Extract metadata
            rel_path = os.path.relpath(file_path, root_path)
            file_ext = Path(file_path).suffix
            file_name = Path(file_path).name

            # Create chunks for better context retrieval
            chunks = self._create_chunks(content)

            # Generate embeddings for chunks
            chunk_embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)

            # Store file information
            self.indexed_files[rel_path] = {
                'content': content,
                'chunks': chunks,
                'chunk_embeddings': chunk_embeddings,
                'filename': file_name,
                'extension': file_ext,
                'path': file_path,
                'lines': len(content.splitlines()),
                'hash': hashlib.md5(content.encode()).hexdigest()
            }

        except Exception as e:
            raise Exception(f"Failed to index {file_path}: {str(e)}")

    def _create_chunks(self, content: str, chunk_size: int = 500) -> List[str]:
        """
        Split content into chunks for better embedding and retrieval

        Args:
            content: File content
            chunk_size: Number of characters per chunk

        Returns:
            List of content chunks
        """
        # Split by lines first to maintain context
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line)

            if current_size + line_size > chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        # Add remaining chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks if chunks else [content]

    def _build_embeddings_matrix(self):
        """Build matrix of embeddings for efficient similarity search"""
        if not self.indexed_files:
            self.embeddings_matrix = None
            self.file_paths = []
            return

        all_embeddings = []
        self.file_paths = []

        for file_path, file_data in self.indexed_files.items():
            chunk_embeddings = file_data['chunk_embeddings']
            for chunk_idx, embedding in enumerate(chunk_embeddings):
                all_embeddings.append(embedding)
                self.file_paths.append((file_path, chunk_idx))

        self.embeddings_matrix = np.array(all_embeddings)

    def get_relevant_context(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant files for a query

        Args:
            query_text: Query text (code to review)
            top_k: Number of context files to return

        Returns:
            List of relevant files with their content
        """
        if not self.indexed_files or self.embeddings_matrix is None:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query_text], show_progress_bar=False)[0]

            # Calculate cosine similarities
            similarities = np.dot(self.embeddings_matrix, query_embedding) / (
                np.linalg.norm(self.embeddings_matrix, axis=1) * np.linalg.norm(query_embedding)
            )

            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[-top_k * 3:][::-1]  # Get more, then deduplicate files

            # Group by file and aggregate scores
            file_scores = defaultdict(list)
            for idx in top_indices:
                file_path, chunk_idx = self.file_paths[idx]
                file_scores[file_path].append((similarities[idx], chunk_idx))

            # Get top files by max similarity score
            top_files = sorted(file_scores.items(),
                               key=lambda x: max(score for score, _ in x[1]),
                               reverse=True)[:top_k]

            # Build context results
            context_results = []
            for file_path, scores in top_files:
                file_data = self.indexed_files[file_path]

                # Get most relevant chunks from this file
                best_chunks_indices = sorted(scores, key=lambda x: x[0], reverse=True)[:2]
                relevant_chunks = [file_data['chunks'][chunk_idx] for _, chunk_idx in best_chunks_indices]

                context_results.append({
                    'filename': file_data['filename'],
                    'path': file_path,
                    'content': file_data['content'],
                    'relevant_chunks': relevant_chunks,
                    'similarity_score': max(score for score, _ in scores),
                    'extension': file_data['extension'],
                    'lines': file_data['lines']
                })

            return context_results

        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []

    def get_indexed_files(self) -> List[str]:
        """Get list of all indexed file paths"""
        return list(self.indexed_files.keys())

    def clear_index(self):
        """Clear all indexed data"""
        self.indexed_files = {}
        self.embeddings_matrix = None
        self.file_paths = []

    def export_index(self) -> Dict[str, Any]:
        """Export index metadata (without embeddings for size)"""
        return {
            'total_files': len(self.indexed_files),
            'files': [
                {
                    'path': path,
                    'filename': data['filename'],
                    'extension': data['extension'],
                    'lines': data['lines'],
                    'hash': data['hash']
                }
                for path, data in self.indexed_files.items()
            ]
        }

    def search_by_filename(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for files by filename

        Args:
            query: Filename query

        Returns:
            List of matching files
        """
        query_lower = query.lower()
        results = []

        for file_path, file_data in self.indexed_files.items():
            if query_lower in file_data['filename'].lower() or query_lower in file_path.lower():
                results.append({
                    'filename': file_data['filename'],
                    'path': file_path,
                    'extension': file_data['extension'],
                    'lines': file_data['lines']
                })

        return results

    def get_file_content(self, file_path: str) -> str:
        """Get content of an indexed file"""
        if file_path in self.indexed_files:
            return self.indexed_files[file_path]['content']
        return ""

    def get_statistics(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        if not self.indexed_files:
            return {
                'total_files': 0,
                'total_lines': 0,
                'file_types': {}
            }

        file_types = defaultdict(int)
        total_lines = 0

        for file_data in self.indexed_files.values():
            file_types[file_data['extension']] += 1
            total_lines += file_data['lines']

        return {
            'total_files': len(self.indexed_files),
            'total_lines': total_lines,
            'file_types': dict(file_types)
        }
