"""
Example API Module - Contains intentional issues for testing code review
"""

import requests
import json


# Security issue: Hardcoded credentials
API_KEY = "sk-1234567890abcdef"
DATABASE_PASSWORD = "admin123"


def fetch_user_data(user_id):
    """Fetch user data from API"""
    # Security issue: SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_id}"

    # Performance issue: No error handling
    response = requests.get(f"https://api.example.com/users/{user_id}")
    data = response.json()

    return data


def process_payments(payment_list):
    """Process a list of payments"""
    # Performance issue: Inefficient loop
    for i in range(len(payment_list)):
        payment = payment_list[i]

        # Code quality: Nested if statements (too deep)
        if payment['status'] == 'pending':
            if payment['amount'] > 0:
                if payment['currency'] == 'USD':
                    if payment['verified']:
                        # Process payment
                        result = charge_card(payment)
                        if result:
                            send_confirmation(payment)

    return True


def charge_card(payment):
    """Charge a credit card - stub"""
    # Missing implementation
    pass


def send_confirmation(payment):
    """Send payment confirmation - stub"""
    # Missing error handling
    print(f"Confirmation sent for payment {payment['id']}")


class UserManager:
    """Manages user operations"""

    def __init__(self):
        # Code quality: Mutable default argument in next method
        self.users = []

    def add_users(self, new_users=[]):
        """Add multiple users"""
        # Bug: Mutable default argument
        for user in new_users:
            self.users.append(user)
        return self.users

    def get_user(self, user_id):
        """Get user by ID"""
        # Performance: Linear search instead of dict lookup
        for user in self.users:
            if user['id'] == user_id:
                return user
        return None

    def delete_user(self, user_id):
        """Delete a user"""
        # Best practice: No logging
        # Best practice: No confirmation
        self.users = [u for u in self.users if u['id'] != user_id]


# Code quality: Global state
current_user = None


def login(username, password):
    """Login user"""
    global current_user

    # Security: Plain text password comparison
    # Security: No rate limiting
    if username == "admin" and password == "admin":
        current_user = {"username": username, "role": "admin"}
        return True
    return False


def calculate_discount(price, discount_percent):
    """Calculate discounted price"""
    # Code quality: No input validation
    # Bug: Potential division by zero if called incorrectly
    discount = price * discount_percent / 100
    return price - discount


# Performance: Function called in loop without caching
def get_exchange_rate(from_currency, to_currency):
    """Get exchange rate - expensive API call"""
    response = requests.get(
        f"https://api.exchange.com/rate?from={from_currency}&to={to_currency}"
    )
    return response.json()['rate']


def convert_prices(prices, target_currency):
    """Convert list of prices to target currency"""
    # Performance: Repeated API calls in loop
    converted = []
    for price_data in prices:
        rate = get_exchange_rate(price_data['currency'], target_currency)
        converted_price = price_data['amount'] * rate
        converted.append(converted_price)
    return converted
