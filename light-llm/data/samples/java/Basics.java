/**
 * Basic Java programming examples.
 */

public class Basics {

    /**
     * Print hello world.
     */
    public static void helloWorld() {
        System.out.println("Hello, World!");
    }

    /**
     * Add two numbers.
     */
    public static int add(int a, int b) {
        return a + b;
    }

    /**
     * Multiply two numbers.
     */
    public static int multiply(int x, int y) {
        return x * y;
    }

    /**
     * Calculate factorial recursively.
     */
    public static long factorial(int n) {
        if (n <= 1) {
            return 1;
        }
        return n * factorial(n - 1);
    }

    /**
     * Calculate nth Fibonacci number.
     */
    public static long fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    /**
     * Check if a number is prime.
     */
    public static boolean isPrime(int n) {
        if (n < 2) {
            return false;
        }
        for (int i = 2; i <= Math.sqrt(n); i++) {
            if (n % i == 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Reverse a string.
     */
    public static String reverseString(String s) {
        return new StringBuilder(s).reverse().toString();
    }

    /**
     * Count vowels in a string.
     */
    public static int countVowels(String text) {
        String vowels = "aeiouAEIOU";
        int count = 0;
        for (char c : text.toCharArray()) {
            if (vowels.indexOf(c) != -1) {
                count++;
            }
        }
        return count;
    }

    /**
     * Check if string is palindrome.
     */
    public static boolean isPalindrome(String s) {
        String cleaned = s.toLowerCase().replaceAll("[^a-z0-9]", "");
        return cleaned.equals(reverseString(cleaned));
    }

    /**
     * Find maximum in array.
     */
    public static int findMax(int[] arr) {
        if (arr == null || arr.length == 0) {
            throw new IllegalArgumentException("Array is empty");
        }
        int max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
            }
        }
        return max;
    }

    /**
     * Sum all elements in array.
     */
    public static int sumArray(int[] arr) {
        int sum = 0;
        for (int num : arr) {
            sum += num;
        }
        return sum;
    }

    /**
     * Binary search in sorted array.
     */
    public static int binarySearch(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (arr[mid] == target) {
                return mid;
            } else if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return -1;
    }

    /**
     * Main method for testing.
     */
    public static void main(String[] args) {
        helloWorld();
        System.out.println("5 + 3 = " + add(5, 3));
        System.out.println("4 * 6 = " + multiply(4, 6));
        System.out.println("Factorial of 5 = " + factorial(5));
        System.out.println("Fibonacci of 7 = " + fibonacci(7));
        System.out.println("Is 17 prime? " + isPrime(17));
        System.out.println("Reverse 'hello' = " + reverseString("hello"));
    }
}
