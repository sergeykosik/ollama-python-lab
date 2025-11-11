# Code Review Examples

This directory contains example Python files for testing the Code Reviewer application.

## Files

### example_api.py
Contains intentional code issues across multiple categories:
- **Security Issues**: Hardcoded credentials, SQL injection, insecure authentication
- **Performance Issues**: Inefficient loops, repeated API calls, linear search
- **Code Quality Issues**: Deep nesting, mutable default arguments, global state
- **Best Practice Violations**: Missing error handling, no logging, no validation

Use this file to test the code reviewer's ability to identify common problems.

### example_utils.py
Contains additional code issues and anti-patterns:
- **Security Issues**: Use of `eval()`, `pickle`, insufficient input sanitization
- **Resource Management**: File handle leaks, missing context managers
- **Architecture Issues**: Single Responsibility Principle violations
- **Performance Issues**: O(n²) algorithms where O(n) is possible
- **Dead Code**: Unused functions

Use this file to test detection of security vulnerabilities and design issues.

### example_good.py
Demonstrates best practices and clean code:
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Detailed docstrings with Args/Returns/Raises
- **Error Handling**: Proper exception handling and logging
- **Design Patterns**: Repository pattern, context managers, data classes
- **Security**: Parameterized queries, safe serialization
- **Performance**: Efficient algorithms, caching
- **SOLID Principles**: Single Responsibility, proper separation of concerns

Use this file to verify the reviewer recognizes good code and provides positive feedback.

## Testing Workflow

### 1. Single File Review
Upload one file (e.g., `example_api.py`) to test basic review functionality.

Expected results:
- Multiple security issues identified
- Performance problems highlighted
- Suggestions for improvements
- Severity classifications (critical/warning/suggestion)

### 2. Multi-File Review
Upload multiple files together to test cross-file analysis.

Try combinations like:
- `example_api.py` + `example_utils.py` (multiple issue files)
- `example_api.py` + `example_good.py` (mixed quality)
- All three files together (comprehensive review)

### 3. With Codebase Context
1. Index the `examples` directory as your codebase
2. Upload `example_api.py` for review with context enabled
3. The reviewer should reference patterns from other indexed files

### 4. Different Review Depths

Test each depth level:
- **Quick**: Fast overview, major issues only
- **Standard**: Balanced review (default)
- **Deep**: Thorough analysis of all aspects
- **Comprehensive**: Exhaustive review including edge cases

### 5. Focus Areas

Test specific focus areas:
- **Security only**: Should emphasize hardcoded credentials, SQL injection, eval()
- **Performance only**: Should focus on loops, caching, algorithm complexity
- **Best Practices only**: Should highlight design patterns, SOLID principles

## Expected Issues by File

### example_api.py
- 🔴 Critical: Hardcoded credentials (API_KEY, DATABASE_PASSWORD)
- 🔴 Critical: SQL injection vulnerability in fetch_user_data()
- 🔴 Critical: Plain text password in login()
- 🟡 Warning: Missing error handling in fetch_user_data()
- 🟡 Warning: Mutable default argument in add_users()
- 🟡 Warning: Global state (current_user)
- 🔵 Suggestion: Deep nesting in process_payments()
- 🔵 Suggestion: Repeated API calls in convert_prices()
- 🔵 Suggestion: Linear search in get_user()

### example_utils.py
- 🔴 Critical: Use of eval() in risky_eval()
- 🔴 Critical: SQL injection in build_sql_query()
- 🔴 Critical: Use of pickle in save_config()
- 🟡 Warning: File handle leak in read_file()
- 🟡 Warning: Missing error handling in parse_json()
- 🟡 Warning: No division by zero check in divide_numbers()
- 🔵 Suggestion: Overly simplistic email validation
- 🔵 Suggestion: Insufficient input sanitization
- 🔵 Suggestion: O(n²) algorithm in find_duplicates()
- 🔵 Suggestion: Dead code (unused_function)

### example_good.py
- ✨ Positive: Comprehensive type hints
- ✨ Positive: Proper error handling with logging
- ✨ Positive: Use of context managers
- ✨ Positive: Parameterized SQL queries
- ✨ Positive: Input validation
- ✨ Positive: Efficient algorithms
- ✨ Positive: Clear documentation
- ✨ Positive: Design patterns (Repository, etc.)

## Tips for Testing

1. **Start Simple**: Begin with a single file review to understand the output format
2. **Compare Results**: Review the same file at different depth levels to see the difference
3. **Test Context**: Index examples folder and upload files to see context-aware suggestions
4. **Verify Accuracy**: Check if the identified issues match the expected issues listed above
5. **Check Suggestions**: Verify that suggested fixes are appropriate and actionable

## Creating Your Own Test Files

To create additional test files:
1. Write code with specific issues you want to test
2. Document the intentional issues in comments
3. Test with different review configurations
4. Verify the AI correctly identifies the issues

## Feedback

If the code reviewer misses issues or provides incorrect feedback, note:
- Which file and issue
- Review depth and focus areas used
- Whether context was enabled
- What you expected vs. what was reported

This helps improve the prompts and configuration for better results.
