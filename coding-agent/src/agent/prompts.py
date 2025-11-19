"""Prompt templates for the coding agent."""

from langchain.prompts import PromptTemplate


SYSTEM_PROMPT_TEMPLATE = """You are an expert coding assistant with access to a comprehensive codebase, database, and documentation.

Your primary goal is to help developers understand code, find information, generate code, and solve problems.

## Your Capabilities

You have access to the following tools:

{tools}

## Tool Usage Guidelines

1. **Code Analysis**: Use analyze_code, find_function, find_class tools to understand code structure
2. **File Operations**: Use read_file, read_file_lines, file_metadata for reading files
3. **Search**: Use search_files to find files by pattern
4. **Database**: Use database tools (query_database, get_table_schema, list_tables) for data queries
5. **Documentation**: Use documentation tools to search and read docs
6. **Code Editing**: Use code editing tools to modify or create files

## Best Practices

When analyzing or generating code:
1. **Search First**: Always search the codebase for similar patterns before suggesting new code
2. **Check Conventions**: Review existing code to understand the project's style and patterns
3. **Consider Context**: Think about how your changes fit into the broader codebase
4. **Validate**: Check database schemas before suggesting queries
5. **Document**: Explain your reasoning and provide context

## Response Format

Use this exact format for your responses:

Question: the input question you must answer
Thought: think about what you need to do
Action: the action to take (must be one of the available tools)
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the comprehensive answer to the original question

## Important Notes

- Always provide clear, actionable answers
- Include file paths and line numbers when referencing code
- Explain complex concepts in simple terms
- If you're unsure, search for more information before answering
- When suggesting code changes, explain the "why" not just the "what"

Begin!

Question: {input}
{agent_scratchpad}"""


CODE_ANALYSIS_PROMPT = """Analyze the following code and provide insights:

File: {file_path}
Language: {language}

Focus on:
1. Code structure and organization
2. Potential issues or improvements
3. Complexity and maintainability
4. Best practices adherence

Code:
{code}

Provide a detailed analysis."""


CODE_GENERATION_PROMPT = """Generate code based on the following specification:

Language: {language}
Description: {description}
Context: {context}

Requirements:
- Follow best practices for {language}
- Include appropriate error handling
- Add docstrings/comments
- Consider edge cases
{additional_requirements}

Generate the code:"""


CODE_REVIEW_PROMPT = """Review the following code changes:

File: {file_path}
Changes:
{changes}

Provide feedback on:
1. Code quality and style
2. Potential bugs or issues
3. Performance considerations
4. Security concerns
5. Suggested improvements

Review:"""


REFACTORING_PROMPT = """Suggest refactoring for the following code:

File: {file_path}
Current Code:
{code}

Refactoring Goals:
{goals}

Provide:
1. Specific refactoring suggestions
2. Improved code examples
3. Explanation of benefits
4. Potential risks or considerations

Suggestions:"""


DATABASE_QUERY_PROMPT = """Help construct a database query:

Goal: {goal}
Available Tables: {tables}
Schema Information: {schema}

Provide:
1. The SQL query
2. Explanation of the query
3. Expected results format
4. Any potential performance considerations

Query:"""


DEBUGGING_PROMPT = """Help debug the following issue:

Error/Issue: {error}
File: {file_path}
Relevant Code:
{code}

Stack Trace (if available):
{stack_trace}

Provide:
1. Analysis of the problem
2. Root cause identification
3. Solution suggestions
4. Prevention strategies

Analysis:"""


DOCUMENTATION_PROMPT = """Generate documentation for:

Type: {doc_type}  # function, class, module, API
Code:
{code}

Generate comprehensive documentation including:
1. Description and purpose
2. Parameters/arguments
3. Return values
4. Usage examples
5. Important notes or warnings

Documentation:"""


def get_system_prompt() -> PromptTemplate:
    """Get the main system prompt template.

    Returns:
        PromptTemplate for the system prompt
    """
    return PromptTemplate(
        template=SYSTEM_PROMPT_TEMPLATE,
        input_variables=["tools", "input", "agent_scratchpad"]
    )


def get_code_analysis_prompt() -> PromptTemplate:
    """Get code analysis prompt template."""
    return PromptTemplate(
        template=CODE_ANALYSIS_PROMPT,
        input_variables=["file_path", "language", "code"]
    )


def get_code_generation_prompt() -> PromptTemplate:
    """Get code generation prompt template."""
    return PromptTemplate(
        template=CODE_GENERATION_PROMPT,
        input_variables=["language", "description", "context", "additional_requirements"]
    )


def get_code_review_prompt() -> PromptTemplate:
    """Get code review prompt template."""
    return PromptTemplate(
        template=CODE_REVIEW_PROMPT,
        input_variables=["file_path", "changes"]
    )


def get_refactoring_prompt() -> PromptTemplate:
    """Get refactoring prompt template."""
    return PromptTemplate(
        template=REFACTORING_PROMPT,
        input_variables=["file_path", "code", "goals"]
    )


def get_database_query_prompt() -> PromptTemplate:
    """Get database query prompt template."""
    return PromptTemplate(
        template=DATABASE_QUERY_PROMPT,
        input_variables=["goal", "tables", "schema"]
    )


def get_debugging_prompt() -> PromptTemplate:
    """Get debugging prompt template."""
    return PromptTemplate(
        template=DEBUGGING_PROMPT,
        input_variables=["error", "file_path", "code", "stack_trace"]
    )


def get_documentation_prompt() -> PromptTemplate:
    """Get documentation prompt template."""
    return PromptTemplate(
        template=DOCUMENTATION_PROMPT,
        input_variables=["doc_type", "code"]
    )
