"""Memory management for the coding agent."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

from ..models.schemas import ConversationHistory, ConversationMessage

logger = logging.getLogger(__name__)


class AgentMemory:
    """Memory manager for the agent."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        max_messages: int = 50,
        use_summary: bool = False
    ):
        """Initialize agent memory.

        Args:
            session_id: Optional session identifier
            max_messages: Maximum number of messages to keep
            use_summary: Whether to use summary-based memory
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.max_messages = max_messages
        self.use_summary = use_summary
        self.conversation_history = ConversationHistory(session_id=self.session_id)
        self.context: Dict[str, Any] = {}

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to the conversation history.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata
        """
        self.conversation_history.add_message(role, content, metadata)

        # Trim if needed
        if len(self.conversation_history.messages) > self.max_messages:
            # Keep system messages and recent messages
            system_msgs = [
                m for m in self.conversation_history.messages
                if m.role == 'system'
            ]
            recent_msgs = [
                m for m in self.conversation_history.messages
                if m.role != 'system'
            ][-self.max_messages:]

            self.conversation_history.messages = system_msgs + recent_msgs

    def get_messages(self, n: Optional[int] = None) -> List[ConversationMessage]:
        """Get recent messages.

        Args:
            n: Number of messages to retrieve (None for all)

        Returns:
            List of ConversationMessage objects
        """
        if n is None:
            return self.conversation_history.messages
        return self.conversation_history.get_recent_messages(n)

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a value from context.

        Args:
            key: Context key
            default: Default value if key not found

        Returns:
            Context value
        """
        return self.context.get(key, default)

    def set_context(self, key: str, value: Any):
        """Set a value in context.

        Args:
            key: Context key
            value: Value to store
        """
        self.context[key] = value

    def clear_context(self):
        """Clear all context."""
        self.context.clear()

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()

    def get_langchain_memory(self) -> ConversationBufferMemory:
        """Get LangChain compatible memory.

        Returns:
            ConversationBufferMemory instance
        """
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )

        # Populate with existing messages
        for msg in self.conversation_history.messages:
            if msg.role == 'user':
                memory.chat_memory.add_user_message(msg.content)
            elif msg.role == 'assistant':
                memory.chat_memory.add_ai_message(msg.content)

        return memory

    def get_summary(self, max_length: int = 500) -> str:
        """Get a summary of the conversation.

        Args:
            max_length: Maximum length of summary

        Returns:
            Conversation summary
        """
        if not self.conversation_history.messages:
            return "No conversation history"

        # Simple summary: recent messages
        recent = self.conversation_history.get_recent_messages(5)
        summary_parts = []

        for msg in recent:
            role = msg.role.upper()
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary_parts.append(f"{role}: {content}")

        summary = "\n".join(summary_parts)
        return summary[:max_length]

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'session_id': self.session_id,
            'message_count': len(self.conversation_history.messages),
            'context': self.context,
            'created_at': self.conversation_history.created_at.isoformat(),
            'updated_at': self.conversation_history.updated_at.isoformat()
        }


class CodeContextMemory:
    """Specialized memory for code-related context."""

    def __init__(self):
        """Initialize code context memory."""
        self.current_files: List[str] = []
        self.analyzed_functions: Dict[str, Dict[str, Any]] = {}
        self.recent_queries: List[str] = []
        self.working_directory: Optional[str] = None

    def add_file(self, file_path: str):
        """Add a file to current context.

        Args:
            file_path: Path to the file
        """
        if file_path not in self.current_files:
            self.current_files.append(file_path)

        # Keep only recent files
        if len(self.current_files) > 10:
            self.current_files = self.current_files[-10:]

    def add_function_analysis(
        self,
        function_name: str,
        file_path: str,
        analysis: Dict[str, Any]
    ):
        """Add function analysis to memory.

        Args:
            function_name: Name of the function
            file_path: Path to the file
            analysis: Analysis data
        """
        key = f"{file_path}::{function_name}"
        self.analyzed_functions[key] = {
            'file_path': file_path,
            'function_name': function_name,
            'analysis': analysis,
            'timestamp': datetime.now()
        }

    def get_function_analysis(
        self,
        function_name: str,
        file_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get function analysis from memory.

        Args:
            function_name: Name of the function
            file_path: Optional file path for disambiguation

        Returns:
            Analysis data or None
        """
        if file_path:
            key = f"{file_path}::{function_name}"
            return self.analyzed_functions.get(key)

        # Search without file path
        for key, data in self.analyzed_functions.items():
            if data['function_name'] == function_name:
                return data

        return None

    def add_query(self, query: str):
        """Add a query to recent queries.

        Args:
            query: Query string
        """
        self.recent_queries.append(query)

        # Keep only recent queries
        if len(self.recent_queries) > 20:
            self.recent_queries = self.recent_queries[-20:]

    def get_recent_files(self, n: int = 5) -> List[str]:
        """Get recently accessed files.

        Args:
            n: Number of files to return

        Returns:
            List of file paths
        """
        return self.current_files[-n:]

    def clear(self):
        """Clear all code context."""
        self.current_files.clear()
        self.analyzed_functions.clear()
        self.recent_queries.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'current_files': self.current_files,
            'analyzed_functions_count': len(self.analyzed_functions),
            'recent_queries_count': len(self.recent_queries),
            'working_directory': self.working_directory
        }


class CombinedMemory:
    """Combined memory manager for agent and code context."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        max_messages: int = 50
    ):
        """Initialize combined memory.

        Args:
            session_id: Optional session identifier
            max_messages: Maximum number of messages to keep
        """
        self.agent_memory = AgentMemory(session_id, max_messages)
        self.code_memory = CodeContextMemory()

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a user message."""
        self.agent_memory.add_message('user', content, metadata)

    def add_ai_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add an AI message."""
        self.agent_memory.add_message('assistant', content, metadata)

    def add_system_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a system message."""
        self.agent_memory.add_message('system', content, metadata)

    def clear_all(self):
        """Clear all memory."""
        self.agent_memory.clear_history()
        self.agent_memory.clear_context()
        self.code_memory.clear()

    def get_full_context(self) -> Dict[str, Any]:
        """Get full context from both memories.

        Returns:
            Combined context dictionary
        """
        return {
            'agent': self.agent_memory.to_dict(),
            'code': self.code_memory.to_dict(),
            'recent_files': self.code_memory.get_recent_files(),
            'recent_messages': [
                {
                    'role': m.role,
                    'content': m.content[:200],
                    'timestamp': m.timestamp.isoformat()
                }
                for m in self.agent_memory.get_messages(5)
            ]
        }
