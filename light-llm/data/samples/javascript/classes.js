/**
 * Object-oriented programming examples in JavaScript.
 */

// Rectangle class
class Rectangle {
    constructor(width, height) {
        this.width = width;
        this.height = height;
    }

    area() {
        return this.width * this.height;
    }

    perimeter() {
        return 2 * (this.width + this.height);
    }

    toString() {
        return `Rectangle(${this.width}x${this.height})`;
    }
}

// Circle class
class Circle {
    constructor(radius) {
        this.radius = radius;
    }

    area() {
        return Math.PI * this.radius ** 2;
    }

    circumference() {
        return 2 * Math.PI * this.radius;
    }
}

// Bank Account class
class BankAccount {
    constructor(accountNumber, initialBalance = 0) {
        this.accountNumber = accountNumber;
        this.balance = initialBalance;
        this.transactions = [];
    }

    deposit(amount) {
        if (amount > 0) {
            this.balance += amount;
            this.transactions.push(`Deposit: +$${amount}`);
            return true;
        }
        return false;
    }

    withdraw(amount) {
        if (amount > 0 && amount <= this.balance) {
            this.balance -= amount;
            this.transactions.push(`Withdrawal: -$${amount}`);
            return true;
        }
        return false;
    }

    getBalance() {
        return this.balance;
    }

    getTransactionHistory() {
        return [...this.transactions];
    }
}

// Stack implementation
class Stack {
    constructor() {
        this.items = [];
    }

    push(item) {
        this.items.push(item);
    }

    pop() {
        if (this.isEmpty()) {
            throw new Error('Stack is empty');
        }
        return this.items.pop();
    }

    peek() {
        if (this.isEmpty()) {
            throw new Error('Stack is empty');
        }
        return this.items[this.items.length - 1];
    }

    isEmpty() {
        return this.items.length === 0;
    }

    size() {
        return this.items.length;
    }
}

// Queue implementation
class Queue {
    constructor() {
        this.items = [];
    }

    enqueue(item) {
        this.items.push(item);
    }

    dequeue() {
        if (this.isEmpty()) {
            throw new Error('Queue is empty');
        }
        return this.items.shift();
    }

    front() {
        if (this.isEmpty()) {
            throw new Error('Queue is empty');
        }
        return this.items[0];
    }

    isEmpty() {
        return this.items.length === 0;
    }

    size() {
        return this.items.length;
    }
}

// Linked List Node
class LinkedListNode {
    constructor(data) {
        this.data = data;
        this.next = null;
    }
}

// Linked List
class LinkedList {
    constructor() {
        this.head = null;
    }

    append(data) {
        const newNode = new LinkedListNode(data);

        if (!this.head) {
            this.head = newNode;
            return;
        }

        let current = this.head;
        while (current.next) {
            current = current.next;
        }
        current.next = newNode;
    }

    prepend(data) {
        const newNode = new LinkedListNode(data);
        newNode.next = this.head;
        this.head = newNode;
    }

    delete(data) {
        if (!this.head) return;

        if (this.head.data === data) {
            this.head = this.head.next;
            return;
        }

        let current = this.head;
        while (current.next) {
            if (current.next.data === data) {
                current.next = current.next.next;
                return;
            }
            current = current.next;
        }
    }

    toArray() {
        const result = [];
        let current = this.head;
        while (current) {
            result.push(current.data);
            current = current.next;
        }
        return result;
    }
}

// Export classes
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        Rectangle,
        Circle,
        BankAccount,
        Stack,
        Queue,
        LinkedList
    };
}
