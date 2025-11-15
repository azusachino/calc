#include <vector>
#include <algorithm>

using namespace std;

// LeetCode #53 - Maximum Subarray (Kadane's Algorithm)
// Find the contiguous subarray with the largest sum
// Time: O(n), Space: O(1)

class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int maxSum = nums[0];
        int currentSum = nums[0];

        for (int i = 1; i < nums.size(); i++) {
            currentSum = max(nums[i], currentSum + nums[i]);
            maxSum = max(maxSum, currentSum);
        }

        return maxSum;
    }

    // Alternative: Dynamic Programming approach
    int maxSubArrayDP(vector<int>& nums) {
        vector<int> dp(nums.size());
        dp[0] = nums[0];
        int maxSum = dp[0];

        for (int i = 1; i < nums.size(); i++) {
            dp[i] = max(nums[i], dp[i-1] + nums[i]);
            maxSum = max(maxSum, dp[i]);
        }

        return maxSum;
    }
};
