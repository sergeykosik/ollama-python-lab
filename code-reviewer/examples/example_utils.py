"""
Example Utilities Module - Contains various code issues for testing
"""

import os
import pickle


def read_file(filename):
    """Read file contents"""
    # Missing error handling
    f = open(filename, 'r')
    content = f.read()
    # Bug: File not closed (resource leak)
    return content


def save_config(config_data, filepath):
    """Save configuration to file"""
    # Security: Using pickle (insecure)
    with open(filepath, 'wb') as f:
        pickle.dump(config_data, f)


def validate_email(email):
    """Validate email address"""
    # Code quality: Overly simplistic validation
    return '@' in email and '.' in email


def sanitize_input(user_input):
    """Sanitize user input"""
    # Security: Insufficient sanitization
    # Just replacing < and > is not enough for XSS prevention
    return user_input.replace('<', '').replace('>', '')


class DataProcessor:
    """Process data - has several design issues"""

    # Code quality: No docstring for methods
    # Architecture: Class doing too many things (violates SRP)

    def __init__(self, data):
        self.data = data
        self.processed = False

    def process(self):
        # Performance: Multiple passes over data
        self.data = [x for x in self.data if x is not None]
        self.data = [x * 2 for x in self.data]
        self.data = [x + 1 for x in self.data]
        self.processed = True

    def save_to_db(self):
        # Architecture: Data processing class shouldn't handle DB
        pass

    def send_email(self):
        # Architecture: Data processing class shouldn't send emails
        pass

    def generate_report(self):
        # Architecture: Too many responsibilities
        pass


def parse_json(json_string):
    """Parse JSON string"""
    # Missing error handling for invalid JSON
    import json
    return json.loads(json_string)


def divide_numbers(a, b):
    """Divide two numbers"""
    # Bug: No check for division by zero
    return a / b


def get_env_variable(key):
    """Get environment variable"""
    # Code quality: Returns None if not found (should raise or have default)
    return os.getenv(key)


# Code quality: Dead code (unused function)
def unused_function():
    """This function is never called"""
    print("This code is never executed")
    return 42


def risky_eval(expression):
    """Evaluate an expression"""
    # Security: CRITICAL - Using eval with user input
    return eval(expression)


def build_sql_query(table, column, value):
    """Build SQL query"""
    # Security: SQL injection vulnerability
    return f"SELECT * FROM {table} WHERE {column} = '{value}'"


# Performance: Inefficient algorithm
def find_duplicates(items):
    """Find duplicate items - O(n²) implementation"""
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates


def should_use_set(items):
    """Better implementation using set - O(n)"""
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)
