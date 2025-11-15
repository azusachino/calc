#include <vector>

using namespace std;

// LeetCode #191 - Number of 1 Bits (Hamming Weight)
// Count number of set bits
// Time: O(k) where k = number of 1 bits, Space: O(1)

class Solution {
public:
    int hammingWeight(uint32_t n) {
        int count = 0;
        while (n) {
            n &= (n - 1); // Clear rightmost 1 bit
            count++;
        }
        return count;
    }
};

// LeetCode #338 - Counting Bits
// Count bits for numbers 0 to n
// Time: O(n), Space: O(1) excluding output

class Solution2 {
public:
    vector<int> countBits(int n) {
        vector<int> result(n + 1);

        for (int i = 1; i <= n; i++) {
            result[i] = result[i >> 1] + (i & 1);
        }

        return result;
    }
};

// LeetCode #371 - Sum of Two Integers
// Add two integers without using + or -
// Time: O(1), Space: O(1)

class Solution3 {
public:
    int getSum(int a, int b) {
        while (b != 0) {
            unsigned int carry = (unsigned int)(a & b) << 1;
            a = a ^ b;
            b = carry;
        }
        return a;
    }
};

// LeetCode #461 - Hamming Distance
// Count positions where bits differ
// Time: O(1), Space: O(1)

class Solution4 {
public:
    int hammingDistance(int x, int y) {
        int xorResult = x ^ y;
        int count = 0;

        while (xorResult) {
            count += xorResult & 1;
            xorResult >>= 1;
        }

        return count;
    }
};

// LeetCode #190 - Reverse Bits
// Reverse bits of a 32-bit unsigned integer
// Time: O(1), Space: O(1)

class Solution5 {
public:
    uint32_t reverseBits(uint32_t n) {
        uint32_t result = 0;

        for (int i = 0; i < 32; i++) {
            result <<= 1;
            result |= (n & 1);
            n >>= 1;
        }

        return result;
    }
};

// LeetCode #231 - Power of Two
// Check if n is power of 2
// Time: O(1), Space: O(1)

class Solution6 {
public:
    bool isPowerOfTwo(int n) {
        if (n <= 0) return false;
        return (n & (n - 1)) == 0;
    }
};

// LeetCode #342 - Power of Four
// Check if n is power of 4
// Time: O(1), Space: O(1)

class Solution7 {
public:
    bool isPowerOfFour(int n) {
        if (n <= 0) return false;

        // Check if power of 2 and 1 bits at odd positions
        return (n & (n - 1)) == 0 && (n & 0x55555555) != 0;
    }
};
