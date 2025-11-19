"""
Object-oriented programming examples in Python.
"""


class Rectangle:
    """A simple rectangle class."""

    def __init__(self, width, height):
        """Initialize rectangle with width and height."""
        self.width = width
        self.height = height

    def area(self):
        """Calculate area of the rectangle."""
        return self.width * self.height

    def perimeter(self):
        """Calculate perimeter of the rectangle."""
        return 2 * (self.width + self.height)

    def __str__(self):
        """String representation."""
        return f"Rectangle({self.width}x{self.height})"


class Circle:
    """A simple circle class."""

    def __init__(self, radius):
        """Initialize circle with radius."""
        self.radius = radius

    def area(self):
        """Calculate area of the circle."""
        import math
        return math.pi * self.radius ** 2

    def circumference(self):
        """Calculate circumference of the circle."""
        import math
        return 2 * math.pi * self.radius


class BankAccount:
    """A simple bank account class."""

    def __init__(self, account_number, initial_balance=0):
        """Initialize bank account."""
        self.account_number = account_number
        self.balance = initial_balance
        self.transactions = []

    def deposit(self, amount):
        """Deposit money into the account."""
        if amount > 0:
            self.balance += amount
            self.transactions.append(f"Deposit: +${amount}")
            return True
        return False

    def withdraw(self, amount):
        """Withdraw money from the account."""
        if 0 < amount <= self.balance:
            self.balance -= amount
            self.transactions.append(f"Withdrawal: -${amount}")
            return True
        return False

    def get_balance(self):
        """Get current balance."""
        return self.balance

    def get_transaction_history(self):
        """Get transaction history."""
        return self.transactions.copy()


class Stack:
    """A simple stack implementation."""

    def __init__(self):
        """Initialize empty stack."""
        self.items = []

    def push(self, item):
        """Push item onto stack."""
        self.items.append(item)

    def pop(self):
        """Pop item from stack."""
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("pop from empty stack")

    def peek(self):
        """Peek at top item without removing."""
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("peek from empty stack")

    def is_empty(self):
        """Check if stack is empty."""
        return len(self.items) == 0

    def size(self):
        """Get stack size."""
        return len(self.items)


if __name__ == "__main__":
    # Test Rectangle
    rect = Rectangle(5, 3)
    print(f"{rect} - Area: {rect.area()}, Perimeter: {rect.perimeter()}")

    # Test Circle
    circle = Circle(7)
    print(f"Circle area: {circle.area():.2f}")

    # Test BankAccount
    account = BankAccount("123456", 1000)
    account.deposit(500)
    account.withdraw(200)
    print(f"Balance: ${account.get_balance()}")
