#include <vector>

using namespace std;

// LeetCode #78 - Subsets (using bit manipulation)
// Generate all subsets using bitmask
// Time: O(n * 2^n), Space: O(1) excluding output

class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        int n = nums.size();
        int totalSubsets = 1 << n; // 2^n
        vector<vector<int>> result;

        for (int mask = 0; mask < totalSubsets; mask++) {
            vector<int> subset;

            for (int i = 0; i < n; i++) {
                if (mask & (1 << i)) {
                    subset.push_back(nums[i]);
                }
            }

            result.push_back(subset);
        }

        return result;
    }
};

// LeetCode #89 - Gray Code
// Generate n-bit Gray code sequence
// Time: O(2^n), Space: O(1) excluding output

class Solution2 {
public:
    vector<int> grayCode(int n) {
        int size = 1 << n;
        vector<int> result;

        for (int i = 0; i < size; i++) {
            result.push_back(i ^ (i >> 1));
        }

        return result;
    }
};

// LeetCode #751 - IP to CIDR (using bit manipulation)
// Convert IP range to CIDR blocks
// Advanced bit manipulation problem

class Solution3 {
public:
    vector<string> ipToCIDR(string ip, int n) {
        long long start = ipToLong(ip);
        vector<string> result;

        while (n > 0) {
            // Find rightmost set bit position
            int trailingZeros = 0;
            long long temp = start;

            while (temp > 0 && (temp & 1) == 0) {
                trailingZeros++;
                temp >>= 1;
            }

            // Maximum CIDR block size we can use
            int maxBlock = 1 << trailingZeros;

            // Don't exceed remaining IPs
            while (maxBlock > n) {
                maxBlock >>= 1;
                trailingZeros--;
            }

            result.push_back(longToIP(start) + "/" + to_string(32 - trailingZeros));

            start += maxBlock;
            n -= maxBlock;
        }

        return result;
    }

private:
    long long ipToLong(const string& ip) {
        long long result = 0;
        int val = 0;

        for (char c : ip) {
            if (c == '.') {
                result = result * 256 + val;
                val = 0;
            } else {
                val = val * 10 + (c - '0');
            }
        }

        result = result * 256 + val;
        return result;
    }

    string longToIP(long long val) {
        return to_string((val >> 24) & 255) + "." +
               to_string((val >> 16) & 255) + "." +
               to_string((val >> 8) & 255) + "." +
               to_string(val & 255);
    }
};

// LeetCode #201 - Bitwise AND of Numbers Range
// Find bitwise AND of all numbers in range [left, right]
// Time: O(log n), Space: O(1)

class Solution4 {
public:
    int rangeBitwiseAnd(int left, int right) {
        int shift = 0;

        // Find common prefix
        while (left < right) {
            left >>= 1;
            right >>= 1;
            shift++;
        }

        return left << shift;
    }
};
