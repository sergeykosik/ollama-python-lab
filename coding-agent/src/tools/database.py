"""Database tools for the agent."""

import json
import logging
from typing import Optional
from langchain.tools import BaseTool
from pydantic import Field

from ..utils.db_utils import DatabaseConnection, sanitize_query, format_query_result

logger = logging.getLogger(__name__)


class DatabaseQueryTool(BaseTool):
    """Tool for querying the database."""

    name: str = "query_database"
    description: str = """
    Executes a SELECT query against the MySQL database.
    Input should be a SQL SELECT query string.
    Returns query results in JSON format.

    IMPORTANT: Only SELECT queries are allowed for safety.

    Example input: "SELECT * FROM users LIMIT 10"
    """

    db_connection: Optional[DatabaseConnection] = Field(default=None)

    def _run(self, query: str) -> str:
        """Execute a database query.

        Args:
            query: SQL query string

        Returns:
            Query results or error message
        """
        if not self.db_connection:
            return "Error: Database not connected"

        try:
            # Sanitize query
            sanitized_query = sanitize_query(query)

            # Execute query
            result = self.db_connection.execute_query(sanitized_query)

            # Format results
            output = {
                'row_count': result.row_count,
                'execution_time_ms': round(result.execution_time_ms, 2),
                'columns': result.columns,
                'rows': result.rows[:100]  # Limit to 100 rows
            }

            if result.row_count > 100:
                output['note'] = f"Showing first 100 of {result.row_count} rows"

            return json.dumps(output, indent=2, default=str)

        except ValueError as e:
            return f"Error: {e} (Only SELECT queries are allowed)"
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return f"Error executing query: {e}"

    async def _arun(self, query: str) -> str:
        """Async version."""
        return self._run(query)


class GetTableSchemaTool(BaseTool):
    """Tool for getting database table schema."""

    name: str = "get_table_schema"
    description: str = """
    Gets the schema information for a database table.
    Input should be a table name.
    Returns columns, types, primary key, foreign keys, and indexes.

    Example input: "users"
    """

    db_connection: Optional[DatabaseConnection] = Field(default=None)

    def _run(self, table_name: str) -> str:
        """Get table schema.

        Args:
            table_name: Name of the table

        Returns:
            Schema information or error message
        """
        if not self.db_connection:
            return "Error: Database not connected"

        try:
            schema = self.db_connection.get_table_schema(table_name)
            if not schema:
                return f"Error: Table '{table_name}' not found"

            result = {
                'table_name': schema.table_name,
                'columns': schema.columns,
                'primary_key': schema.primary_key,
                'foreign_keys': schema.foreign_keys,
                'indexes': schema.indexes
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting table schema: {e}")
            return f"Error: {e}"

    async def _arun(self, table_name: str) -> str:
        """Async version."""
        return self._run(table_name)


class ListTablesTool(BaseTool):
    """Tool for listing all database tables."""

    name: str = "list_tables"
    description: str = """
    Lists all tables in the database.
    Input can be empty string or a search term to filter tables.
    Returns list of table names.

    Example input: "" or "user" to search for tables containing "user"
    """

    db_connection: Optional[DatabaseConnection] = Field(default=None)

    def _run(self, search_term: str = "") -> str:
        """List database tables.

        Args:
            search_term: Optional search term to filter tables

        Returns:
            List of tables or error message
        """
        if not self.db_connection:
            return "Error: Database not connected"

        try:
            if search_term:
                tables = self.db_connection.search_tables(search_term)
                result = {
                    'search_term': search_term,
                    'total_tables': len(tables),
                    'tables': tables
                }
            else:
                tables = self.db_connection.get_table_names()
                result = {
                    'total_tables': len(tables),
                    'tables': tables
                }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return f"Error: {e}"

    async def _arun(self, search_term: str = "") -> str:
        """Async version."""
        return self._run(search_term)


class GetSampleDataTool(BaseTool):
    """Tool for getting sample data from a table."""

    name: str = "get_sample_data"
    description: str = """
    Gets sample rows from a database table.
    Input should be JSON: {"table_name": "users", "limit": 5}
    Returns sample rows from the table.

    Example input: {"table_name": "products", "limit": 10}
    """

    db_connection: Optional[DatabaseConnection] = Field(default=None)

    def _run(self, input_str: str) -> str:
        """Get sample data from a table.

        Args:
            input_str: JSON with table_name and optional limit

        Returns:
            Sample data or error message
        """
        if not self.db_connection:
            return "Error: Database not connected"

        try:
            input_data = json.loads(input_str)
            table_name = input_data['table_name']
            limit = input_data.get('limit', 5)

            result = self.db_connection.get_sample_data(table_name, limit)
            if not result:
                return f"Error: Could not get sample data from table '{table_name}'"

            output = {
                'table_name': table_name,
                'row_count': result.row_count,
                'columns': result.columns,
                'rows': result.rows
            }

            return json.dumps(output, indent=2, default=str)

        except json.JSONDecodeError:
            return "Error: Invalid JSON input"
        except Exception as e:
            logger.error(f"Error getting sample data: {e}")
            return f"Error: {e}"

    async def _arun(self, input_str: str) -> str:
        """Async version."""
        return self._run(input_str)


class GetTableRelationshipsTool(BaseTool):
    """Tool for getting table relationships."""

    name: str = "get_table_relationships"
    description: str = """
    Gets foreign key relationships between database tables.
    Input can be empty string to get all relationships.
    Returns mapping of tables to their relationships.

    Example input: ""
    """

    db_connection: Optional[DatabaseConnection] = Field(default=None)

    def _run(self, _: str = "") -> str:
        """Get table relationships.

        Args:
            _: Unused parameter

        Returns:
            Table relationships or error message
        """
        if not self.db_connection:
            return "Error: Database not connected"

        try:
            relationships = self.db_connection.get_table_relationships()

            result = {
                'total_tables_with_relationships': len(relationships),
                'relationships': relationships
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error getting table relationships: {e}")
            return f"Error: {e}"

    async def _arun(self, input_str: str = "") -> str:
        """Async version."""
        return self._run(input_str)
