/**
 * Basic JavaScript programming examples.
 */

// Simple functions
function helloWorld() {
    console.log("Hello, World!");
}

function add(a, b) {
    return a + b;
}

function multiply(x, y) {
    return x * y;
}

// Recursive factorial
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// Fibonacci sequence
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Check if number is prime
function isPrime(n) {
    if (n < 2) return false;
    for (let i = 2; i <= Math.sqrt(n); i++) {
        if (n % i === 0) return false;
    }
    return true;
}

// Array operations
function reverseArray(arr) {
    return arr.slice().reverse();
}

function findMax(arr) {
    return Math.max(...arr);
}

function sumArray(arr) {
    return arr.reduce((sum, num) => sum + num, 0);
}

// String operations
function reverseString(str) {
    return str.split('').reverse().join('');
}

function countVowels(text) {
    const vowels = 'aeiouAEIOU';
    return text.split('').filter(char => vowels.includes(char)).length;
}

function isPalindrome(str) {
    const cleaned = str.toLowerCase().replace(/[^a-z0-9]/g, '');
    return cleaned === cleaned.split('').reverse().join('');
}

// Arrow functions and modern syntax
const square = (x) => x * x;
const cube = (x) => x * x * x;

const greet = (name) => {
    return `Hello, ${name}!`;
};

// Array methods
const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

const evens = numbers.filter(n => n % 2 === 0);
const doubled = numbers.map(n => n * 2);
const sum = numbers.reduce((acc, n) => acc + n, 0);

// Object creation
function createPerson(name, age) {
    return {
        name: name,
        age: age,
        greet() {
            return `Hi, I'm ${this.name} and I'm ${this.age} years old`;
        }
    };
}

// Export for modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        add,
        multiply,
        factorial,
        fibonacci,
        isPrime,
        reverseString,
        countVowels,
        isPalindrome
    };
}
