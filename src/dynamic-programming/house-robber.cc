#include <vector>
#include <algorithm>

using namespace std;

// LeetCode #198 - House Robber
// Rob houses to maximize money without alerting police (can't rob adjacent)
// Time: O(n), Space: O(1)

class Solution {
public:
    int rob(vector<int>& nums) {
        if (nums.empty()) return 0;
        if (nums.size() == 1) return nums[0];

        int prev2 = 0;
        int prev1 = nums[0];

        for (int i = 1; i < nums.size(); i++) {
            int current = max(prev1, prev2 + nums[i]);
            prev2 = prev1;
            prev1 = current;
        }

        return prev1;
    }

    // Alternative: DP array approach
    int robDP(vector<int>& nums) {
        if (nums.empty()) return 0;
        if (nums.size() == 1) return nums[0];

        vector<int> dp(nums.size());
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);

        for (int i = 2; i < nums.size(); i++) {
            dp[i] = max(dp[i-1], dp[i-2] + nums[i]);
        }

        return dp[nums.size() - 1];
    }
};

// LeetCode #213 - House Robber II
// Houses arranged in a circle
// Time: O(n), Space: O(1)

class Solution2 {
public:
    int rob(vector<int>& nums) {
        if (nums.size() == 1) return nums[0];

        // Either rob houses 0 to n-2 OR houses 1 to n-1
        return max(robRange(nums, 0, nums.size() - 2),
                   robRange(nums, 1, nums.size() - 1));
    }

private:
    int robRange(vector<int>& nums, int start, int end) {
        int prev2 = 0, prev1 = 0;

        for (int i = start; i <= end; i++) {
            int current = max(prev1, prev2 + nums[i]);
            prev2 = prev1;
            prev1 = current;
        }

        return prev1;
    }
};
