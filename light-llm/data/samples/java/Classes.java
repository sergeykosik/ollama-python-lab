/**
 * Object-oriented programming examples in Java.
 */

/**
 * Rectangle class.
 */
class Rectangle {
    private double width;
    private double height;

    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    public double getWidth() {
        return width;
    }

    public double getHeight() {
        return height;
    }

    public double area() {
        return width * height;
    }

    public double perimeter() {
        return 2 * (width + height);
    }

    @Override
    public String toString() {
        return String.format("Rectangle(%.1fx%.1f)", width, height);
    }
}

/**
 * Circle class.
 */
class Circle {
    private double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    public double getRadius() {
        return radius;
    }

    public double area() {
        return Math.PI * radius * radius;
    }

    public double circumference() {
        return 2 * Math.PI * radius;
    }
}

/**
 * Bank Account class.
 */
class BankAccount {
    private String accountNumber;
    private double balance;
    private java.util.List<String> transactions;

    public BankAccount(String accountNumber, double initialBalance) {
        this.accountNumber = accountNumber;
        this.balance = initialBalance;
        this.transactions = new java.util.ArrayList<>();
    }

    public boolean deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            transactions.add(String.format("Deposit: +$%.2f", amount));
            return true;
        }
        return false;
    }

    public boolean withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            transactions.add(String.format("Withdrawal: -$%.2f", amount));
            return true;
        }
        return false;
    }

    public double getBalance() {
        return balance;
    }

    public java.util.List<String> getTransactionHistory() {
        return new java.util.ArrayList<>(transactions);
    }
}

/**
 * Stack implementation using generics.
 */
class Stack<T> {
    private java.util.ArrayList<T> items;

    public Stack() {
        items = new java.util.ArrayList<>();
    }

    public void push(T item) {
        items.add(item);
    }

    public T pop() {
        if (isEmpty()) {
            throw new IllegalStateException("Stack is empty");
        }
        return items.remove(items.size() - 1);
    }

    public T peek() {
        if (isEmpty()) {
            throw new IllegalStateException("Stack is empty");
        }
        return items.get(items.size() - 1);
    }

    public boolean isEmpty() {
        return items.isEmpty();
    }

    public int size() {
        return items.size();
    }
}

/**
 * Queue implementation using generics.
 */
class Queue<T> {
    private java.util.LinkedList<T> items;

    public Queue() {
        items = new java.util.LinkedList<>();
    }

    public void enqueue(T item) {
        items.addLast(item);
    }

    public T dequeue() {
        if (isEmpty()) {
            throw new IllegalStateException("Queue is empty");
        }
        return items.removeFirst();
    }

    public T front() {
        if (isEmpty()) {
            throw new IllegalStateException("Queue is empty");
        }
        return items.getFirst();
    }

    public boolean isEmpty() {
        return items.isEmpty();
    }

    public int size() {
        return items.size();
    }
}

/**
 * Linked List Node.
 */
class LinkedListNode<T> {
    T data;
    LinkedListNode<T> next;

    public LinkedListNode(T data) {
        this.data = data;
        this.next = null;
    }
}

/**
 * Linked List implementation.
 */
class LinkedList<T> {
    private LinkedListNode<T> head;

    public LinkedList() {
        head = null;
    }

    public void append(T data) {
        LinkedListNode<T> newNode = new LinkedListNode<>(data);

        if (head == null) {
            head = newNode;
            return;
        }

        LinkedListNode<T> current = head;
        while (current.next != null) {
            current = current.next;
        }
        current.next = newNode;
    }

    public void prepend(T data) {
        LinkedListNode<T> newNode = new LinkedListNode<>(data);
        newNode.next = head;
        head = newNode;
    }

    public void delete(T data) {
        if (head == null) return;

        if (head.data.equals(data)) {
            head = head.next;
            return;
        }

        LinkedListNode<T> current = head;
        while (current.next != null) {
            if (current.next.data.equals(data)) {
                current.next = current.next.next;
                return;
            }
            current = current.next;
        }
    }

    public java.util.List<T> toList() {
        java.util.List<T> result = new java.util.ArrayList<>();
        LinkedListNode<T> current = head;
        while (current != null) {
            result.add(current.data);
            current = current.next;
        }
        return result;
    }
}

/**
 * Main class for testing.
 */
public class Classes {
    public static void main(String[] args) {
        // Test Rectangle
        Rectangle rect = new Rectangle(5, 3);
        System.out.println(rect + " - Area: " + rect.area());

        // Test Circle
        Circle circle = new Circle(7);
        System.out.printf("Circle area: %.2f%n", circle.area());

        // Test BankAccount
        BankAccount account = new BankAccount("123456", 1000);
        account.deposit(500);
        account.withdraw(200);
        System.out.printf("Balance: $%.2f%n", account.getBalance());

        // Test Stack
        Stack<Integer> stack = new Stack<>();
        stack.push(1);
        stack.push(2);
        stack.push(3);
        System.out.println("Stack pop: " + stack.pop());

        // Test LinkedList
        LinkedList<Integer> list = new LinkedList<>();
        list.append(1);
        list.append(2);
        list.append(3);
        System.out.println("LinkedList: " + list.toList());
    }
}
