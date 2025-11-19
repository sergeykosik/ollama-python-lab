"""Database utility functions for the coding agent."""

import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, inspect, text, MetaData, Table
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from ..models.schemas import DatabaseSchema, QueryResult
import time

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection manager."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        pool_size: int = 5,
        max_overflow: int = 10
    ):
        """Initialize database connection.

        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Database username
            password: Database password
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
        """
        self.connection_string = (
            f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        )
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self.pool_size = pool_size
        self.max_overflow = max_overflow

    def connect(self) -> bool:
        """Establish database connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.engine = create_engine(
                self.connection_string,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,  # Verify connections before using
                echo=False
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def disconnect(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")

    @contextmanager
    def get_session(self):
        """Get a database session.

        Yields:
            SQLAlchemy Session object
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not connected. Call connect() first.")

        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()

    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Execute a SQL query.

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            QueryResult object

        Raises:
            SQLAlchemyError: If query execution fails
        """
        if not self.engine:
            raise RuntimeError("Database not connected. Call connect() first.")

        start_time = time.time()
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})

                # Check if query returns rows
                if result.returns_rows:
                    rows = [dict(row._mapping) for row in result]
                    columns = list(result.keys())
                else:
                    rows = []
                    columns = []

                execution_time = (time.time() - start_time) * 1000  # Convert to ms

                return QueryResult(
                    query=query,
                    rows=rows,
                    row_count=len(rows),
                    execution_time_ms=execution_time,
                    columns=columns
                )
        except SQLAlchemyError as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def get_table_names(self) -> List[str]:
        """Get all table names in the database.

        Returns:
            List of table names
        """
        if not self.engine:
            raise RuntimeError("Database not connected. Call connect() first.")

        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_table_schema(self, table_name: str) -> Optional[DatabaseSchema]:
        """Get schema information for a table.

        Args:
            table_name: Name of the table

        Returns:
            DatabaseSchema object or None if table doesn't exist
        """
        if not self.engine:
            raise RuntimeError("Database not connected. Call connect() first.")

        try:
            inspector = inspect(self.engine)

            # Get columns
            columns = []
            for col in inspector.get_columns(table_name):
                columns.append({
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col['nullable'],
                    'default': col.get('default'),
                    'autoincrement': col.get('autoincrement', False)
                })

            # Get primary key
            pk_constraint = inspector.get_pk_constraint(table_name)
            primary_key = pk_constraint.get('constrained_columns', [])

            # Get foreign keys
            foreign_keys = []
            for fk in inspector.get_foreign_keys(table_name):
                foreign_keys.append({
                    'columns': fk['constrained_columns'],
                    'referred_table': fk['referred_table'],
                    'referred_columns': fk['referred_columns']
                })

            # Get indexes
            indexes = [idx['name'] for idx in inspector.get_indexes(table_name)]

            return DatabaseSchema(
                table_name=table_name,
                columns=columns,
                primary_key=primary_key if primary_key else None,
                foreign_keys=foreign_keys,
                indexes=indexes
            )
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {e}")
            return None

    def get_all_schemas(self) -> Dict[str, DatabaseSchema]:
        """Get schema information for all tables.

        Returns:
            Dictionary mapping table names to DatabaseSchema objects
        """
        schemas = {}
        for table_name in self.get_table_names():
            schema = self.get_table_schema(table_name)
            if schema:
                schemas[table_name] = schema
        return schemas

    def get_sample_data(
        self,
        table_name: str,
        limit: int = 5
    ) -> Optional[QueryResult]:
        """Get sample data from a table.

        Args:
            table_name: Name of the table
            limit: Number of rows to retrieve

        Returns:
            QueryResult object or None if error occurs
        """
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            return self.execute_query(query)
        except Exception as e:
            logger.error(f"Error getting sample data from {table_name}: {e}")
            return None

    def search_tables(self, search_term: str) -> List[str]:
        """Search for tables by name.

        Args:
            search_term: Term to search for

        Returns:
            List of matching table names
        """
        all_tables = self.get_table_names()
        search_term = search_term.lower()
        return [t for t in all_tables if search_term in t.lower()]

    def get_table_relationships(self) -> Dict[str, List[Dict[str, str]]]:
        """Get relationships between tables.

        Returns:
            Dictionary mapping table names to their relationships
        """
        if not self.engine:
            raise RuntimeError("Database not connected. Call connect() first.")

        inspector = inspect(self.engine)
        relationships = {}

        for table_name in self.get_table_names():
            table_rels = []
            for fk in inspector.get_foreign_keys(table_name):
                table_rels.append({
                    'from_table': table_name,
                    'from_columns': ', '.join(fk['constrained_columns']),
                    'to_table': fk['referred_table'],
                    'to_columns': ', '.join(fk['referred_columns'])
                })
            if table_rels:
                relationships[table_name] = table_rels

        return relationships


def sanitize_query(query: str) -> str:
    """Sanitize SQL query to prevent injection.

    Args:
        query: SQL query string

    Returns:
        Sanitized query string

    Raises:
        ValueError: If query contains suspicious content
    """
    # Convert to lowercase for checking
    query_lower = query.lower().strip()

    # Block dangerous operations
    dangerous_keywords = [
        'drop table',
        'drop database',
        'truncate',
        'delete from',
        'update ',
        'insert into',
        'alter table',
        'create table',
        'exec ',
        'execute '
    ]

    for keyword in dangerous_keywords:
        if keyword in query_lower:
            raise ValueError(f"Query contains forbidden keyword: {keyword}")

    # Ensure query is SELECT only
    if not query_lower.startswith('select'):
        raise ValueError("Only SELECT queries are allowed")

    return query


def format_query_result(result: QueryResult, max_rows: int = 100) -> str:
    """Format query result as a readable string.

    Args:
        result: QueryResult object
        max_rows: Maximum number of rows to include

    Returns:
        Formatted string
    """
    lines = [
        f"Query: {result.query}",
        f"Rows: {result.row_count}",
        f"Execution time: {result.execution_time_ms:.2f}ms",
        ""
    ]

    if result.rows:
        # Add column headers
        lines.append(" | ".join(result.columns))
        lines.append("-" * 80)

        # Add rows
        for i, row in enumerate(result.rows[:max_rows]):
            row_str = " | ".join(str(row.get(col, '')) for col in result.columns)
            lines.append(row_str)

        if result.row_count > max_rows:
            lines.append(f"... and {result.row_count - max_rows} more rows")

    return "\n".join(lines)


def get_connection_info(engine: Engine) -> Dict[str, Any]:
    """Get connection information.

    Args:
        engine: SQLAlchemy Engine

    Returns:
        Dictionary with connection information
    """
    return {
        'url': str(engine.url).replace(engine.url.password or '', '***'),
        'driver': engine.driver,
        'pool_size': engine.pool.size(),
        'checked_in': engine.pool.checkedin(),
        'checked_out': engine.pool.checkedout(),
        'overflow': engine.pool.overflow()
    }
