"""
Data structures and algorithms in Python.
"""


def binary_search(arr, target):
    """
    Binary search algorithm.

    Args:
        arr: Sorted array to search
        target: Value to find

    Returns:
        Index of target or -1 if not found
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def bubble_sort(arr):
    """
    Bubble sort algorithm.

    Args:
        arr: Array to sort

    Returns:
        Sorted array
    """
    n = len(arr)
    arr = arr.copy()

    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break

    return arr


def merge_sort(arr):
    """
    Merge sort algorithm.

    Args:
        arr: Array to sort

    Returns:
        Sorted array
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)


def merge(left, right):
    """Merge two sorted arrays."""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


class LinkedListNode:
    """Node for linked list."""

    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """Simple linked list implementation."""

    def __init__(self):
        self.head = None

    def append(self, data):
        """Append data to the end."""
        new_node = LinkedListNode(data)

        if not self.head:
            self.head = new_node
            return

        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def prepend(self, data):
        """Prepend data to the beginning."""
        new_node = LinkedListNode(data)
        new_node.next = self.head
        self.head = new_node

    def delete(self, data):
        """Delete first occurrence of data."""
        if not self.head:
            return

        if self.head.data == data:
            self.head = self.head.next
            return

        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next

    def to_list(self):
        """Convert to Python list."""
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result


class Queue:
    """Simple queue implementation."""

    def __init__(self):
        self.items = []

    def enqueue(self, item):
        """Add item to queue."""
        self.items.append(item)

    def dequeue(self):
        """Remove and return first item."""
        if not self.is_empty():
            return self.items.pop(0)
        raise IndexError("dequeue from empty queue")

    def is_empty(self):
        """Check if queue is empty."""
        return len(self.items) == 0

    def size(self):
        """Get queue size."""
        return len(self.items)


if __name__ == "__main__":
    # Test binary search
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    print(f"Binary search for 7: index {binary_search(arr, 7)}")

    # Test sorting
    unsorted = [64, 34, 25, 12, 22, 11, 90]
    print(f"Bubble sort: {bubble_sort(unsorted)}")
    print(f"Merge sort: {merge_sort(unsorted)}")

    # Test linked list
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    ll.append(3)
    print(f"Linked list: {ll.to_list()}")
