#include <vector>
#include <unordered_map>

using namespace std;

// LeetCode #560 - Subarray Sum Equals K
// Find total number of continuous subarrays whose sum equals k
// Time: O(n), Space: O(n)

class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> prefixSumCount;
        prefixSumCount[0] = 1; // Empty prefix

        int count = 0;
        int sum = 0;

        for (int num : nums) {
            sum += num;

            // Check if there's a prefix sum that makes current subarray sum = k
            if (prefixSumCount.count(sum - k)) {
                count += prefixSumCount[sum - k];
            }

            prefixSumCount[sum]++;
        }

        return count;
    }
};

// LeetCode #523 - Continuous Subarray Sum
// Check if array has continuous subarray of size at least 2 that sums to multiple of k
// Time: O(n), Space: O(k)

class Solution2 {
public:
    bool checkSubarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> remainderIndex;
        remainderIndex[0] = -1;

        int sum = 0;

        for (int i = 0; i < nums.size(); i++) {
            sum += nums[i];
            int remainder = sum % k;

            if (remainderIndex.count(remainder)) {
                if (i - remainderIndex[remainder] >= 2) {
                    return true;
                }
            } else {
                remainderIndex[remainder] = i;
            }
        }

        return false;
    }
};
