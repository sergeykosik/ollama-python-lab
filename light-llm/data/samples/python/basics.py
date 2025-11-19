"""
Basic Python programming examples.
"""


def hello_world():
    """Print hello world."""
    print("Hello, World!")


def add(a, b):
    """Add two numbers."""
    return a + b


def multiply(x, y):
    """Multiply two numbers."""
    return x * y


def factorial(n):
    """Calculate factorial recursively."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def reverse_string(s):
    """Reverse a string."""
    return s[::-1]


def count_vowels(text):
    """Count vowels in a string."""
    vowels = "aeiouAEIOU"
    return sum(1 for char in text if char in vowels)


if __name__ == "__main__":
    print(hello_world())
    print(f"5 + 3 = {add(5, 3)}")
    print(f"4 * 6 = {multiply(4, 6)}")
    print(f"Factorial of 5 = {factorial(5)}")
    print(f"Fibonacci of 7 = {fibonacci(7)}")
    print(f"Is 17 prime? {is_prime(17)}")
