#include <vector>

using namespace std;

// LeetCode #136 - Single Number
// Find element that appears once (all others appear twice)
// Time: O(n), Space: O(1)
// Using XOR: a ^ a = 0, a ^ 0 = a

class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int result = 0;
        for (int num : nums) {
            result ^= num;
        }
        return result;
    }
};

// LeetCode #137 - Single Number II
// Find element that appears once (all others appear three times)
// Time: O(n), Space: O(1)

class Solution2 {
public:
    int singleNumber(vector<int>& nums) {
        int ones = 0, twos = 0;

        for (int num : nums) {
            twos |= ones & num;
            ones ^= num;
            int threes = ones & twos;
            ones &= ~threes;
            twos &= ~threes;
        }

        return ones;
    }
};

// LeetCode #260 - Single Number III
// Find two elements that appear once (all others appear twice)
// Time: O(n), Space: O(1)

class Solution3 {
public:
    vector<int> singleNumber(vector<int>& nums) {
        // Get XOR of the two unique numbers
        long long xorSum = 0;
        for (int num : nums) {
            xorSum ^= num;
        }

        // Find rightmost set bit
        long long rightmostBit = xorSum & -xorSum;

        int num1 = 0, num2 = 0;

        // Partition numbers based on rightmost bit
        for (int num : nums) {
            if (num & rightmostBit) {
                num1 ^= num;
            } else {
                num2 ^= num;
            }
        }

        return {num1, num2};
    }
};
