/**
 * Asynchronous programming examples in JavaScript.
 */

// Basic Promise
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Async function with await
async function fetchUserData(userId) {
    try {
        console.log(`Fetching user ${userId}...`);
        await delay(1000); // Simulate API call

        return {
            id: userId,
            name: 'John Doe',
            email: 'john@example.com'
        };
    } catch (error) {
        console.error('Error fetching user:', error);
        throw error;
    }
}

// Multiple async operations
async function fetchMultipleUsers(userIds) {
    try {
        const promises = userIds.map(id => fetchUserData(id));
        const users = await Promise.all(promises);
        return users;
    } catch (error) {
        console.error('Error fetching multiple users:', error);
        throw error;
    }
}

// Retry logic
async function fetchWithRetry(url, maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            console.log(`Retry ${i + 1}/${maxRetries}...`);
            await delay(1000 * (i + 1)); // Exponential backoff
        }
    }
}

// Promise chain
function processData(data) {
    return Promise.resolve(data)
        .then(d => d.toUpperCase())
        .then(d => d.split(''))
        .then(d => d.reverse())
        .then(d => d.join(''))
        .catch(error => console.error('Error:', error));
}

// Async generator
async function* numberGenerator(max) {
    for (let i = 1; i <= max; i++) {
        await delay(100);
        yield i;
    }
}

// Using async generator
async function processNumbers() {
    for await (const num of numberGenerator(10)) {
        console.log(num);
    }
}

// Race condition example
async function raceExample() {
    const promise1 = delay(1000).then(() => 'First');
    const promise2 = delay(500).then(() => 'Second');
    const promise3 = delay(1500).then(() => 'Third');

    const winner = await Promise.race([promise1, promise2, promise3]);
    console.log('Winner:', winner);
}

// Parallel vs Sequential
async function sequentialProcessing(items) {
    const results = [];
    for (const item of items) {
        const result = await processItem(item);
        results.push(result);
    }
    return results;
}

async function parallelProcessing(items) {
    const promises = items.map(item => processItem(item));
    return await Promise.all(promises);
}

async function processItem(item) {
    await delay(100);
    return item * 2;
}

// Error handling patterns
async function safeAsyncOperation() {
    try {
        const result = await riskyOperation();
        return { success: true, data: result };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

async function riskyOperation() {
    await delay(500);
    if (Math.random() > 0.5) {
        throw new Error('Random failure');
    }
    return 'Success!';
}

// Timeout wrapper
function withTimeout(promise, ms) {
    const timeout = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Timeout')), ms)
    );
    return Promise.race([promise, timeout]);
}

// Usage example
async function main() {
    try {
        // Fetch single user
        const user = await fetchUserData(1);
        console.log('User:', user);

        // Fetch multiple users
        const users = await fetchMultipleUsers([1, 2, 3]);
        console.log('Users:', users);

        // Process with timeout
        const result = await withTimeout(
            fetchUserData(1),
            2000
        );
        console.log('Result:', result);

    } catch (error) {
        console.error('Error in main:', error);
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        delay,
        fetchUserData,
        fetchMultipleUsers,
        fetchWithRetry,
        processData,
        withTimeout
    };
}
