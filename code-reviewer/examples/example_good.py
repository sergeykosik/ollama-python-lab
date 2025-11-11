"""
Example of Good Code - Best practices demonstrated
This file demonstrates good coding practices for comparison
"""

import logging
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum


# Configure logging
logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User role enumeration"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


@dataclass
class User:
    """User data class with type hints"""
    id: int
    username: str
    email: str
    role: UserRole

    def is_admin(self) -> bool:
        """Check if user has admin role"""
        return self.role == UserRole.ADMIN


class UserRepository:
    """
    Repository pattern for user management

    Separates data access logic from business logic
    """

    def __init__(self, database_connection):
        """
        Initialize repository with database connection

        Args:
            database_connection: Database connection object
        """
        self._db = database_connection
        self._cache: Dict[int, User] = {}
        logger.info("UserRepository initialized")

    def get_user(self, user_id: int) -> Optional[User]:
        """
        Get user by ID with caching

        Args:
            user_id: User ID to fetch

        Returns:
            User object if found, None otherwise
        """
        # Check cache first
        if user_id in self._cache:
            logger.debug(f"Cache hit for user {user_id}")
            return self._cache[user_id]

        try:
            # Use parameterized query to prevent SQL injection
            query = "SELECT * FROM users WHERE id = ?"
            result = self._db.execute(query, (user_id,))

            if result:
                user = self._create_user_from_result(result)
                self._cache[user_id] = user
                return user

            logger.warning(f"User {user_id} not found")
            return None

        except Exception as e:
            logger.error(f"Error fetching user {user_id}: {e}")
            raise

    def _create_user_from_result(self, result: Dict[str, Any]) -> User:
        """
        Create User object from database result

        Args:
            result: Database query result

        Returns:
            User object
        """
        return User(
            id=result['id'],
            username=result['username'],
            email=result['email'],
            role=UserRole(result['role'])
        )


@contextmanager
def safe_file_operation(filepath: str, mode: str = 'r'):
    """
    Context manager for safe file operations

    Ensures files are properly closed even if errors occur

    Args:
        filepath: Path to file
        mode: File open mode

    Yields:
        File object
    """
    file_handle = None
    try:
        file_handle = open(filepath, mode, encoding='utf-8')
        yield file_handle
    except IOError as e:
        logger.error(f"File operation error: {e}")
        raise
    finally:
        if file_handle:
            file_handle.close()


def validate_email(email: str) -> bool:
    """
    Validate email address format

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise
    """
    import re

    if not email or not isinstance(email, str):
        return False

    # RFC 5322 simplified pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def calculate_discount(
    price: float,
    discount_percent: float,
    min_price: float = 0.0
) -> float:
    """
    Calculate discounted price with validation

    Args:
        price: Original price (must be >= 0)
        discount_percent: Discount percentage (0-100)
        min_price: Minimum allowed price after discount

    Returns:
        Discounted price

    Raises:
        ValueError: If inputs are invalid
    """
    if price < 0:
        raise ValueError("Price cannot be negative")

    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount must be between 0 and 100")

    discount_amount = price * discount_percent / 100
    final_price = max(price - discount_amount, min_price)

    return round(final_price, 2)


def find_duplicates_efficient(items: List[Any]) -> List[Any]:
    """
    Find duplicate items efficiently using set

    Time complexity: O(n)
    Space complexity: O(n)

    Args:
        items: List of items to check

    Returns:
        List of duplicate items (order preserved)
    """
    seen = set()
    duplicates = []

    for item in items:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)

    return duplicates


class ConfigurationManager:
    """
    Manages application configuration safely

    Uses JSON instead of pickle for security
    """

    def __init__(self, config_path: str):
        """
        Initialize configuration manager

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file"""
        try:
            with safe_file_operation(self.config_path, 'r') as f:
                import json
                self._config = json.load(f)
            logger.info("Configuration loaded successfully")
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            self._config = {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)

    def save(self) -> None:
        """Save configuration to file"""
        try:
            with safe_file_operation(self.config_path, 'w') as f:
                import json
                json.dump(self._config, f, indent=2)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise


def process_data_pipeline(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process data through a pipeline efficiently

    Single pass instead of multiple iterations

    Args:
        data: Input data list

    Returns:
        Processed data list
    """
    processed = []

    for item in data:
        # Skip None values
        if item is None:
            continue

        # Apply transformations in single pass
        transformed = {
            key: (value * 2 + 1) if isinstance(value, (int, float)) else value
            for key, value in item.items()
        }

        processed.append(transformed)

    return processed
