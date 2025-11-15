#include <vector>
#include <algorithm>

using namespace std;

// LeetCode #300 - Longest Increasing Subsequence
// Find length of longest strictly increasing subsequence
// Time: O(n^2) for DP, O(n log n) for binary search
// Space: O(n)

class Solution {
public:
    // Dynamic Programming approach
    int lengthOfLIS(vector<int>& nums) {
        if (nums.empty()) return 0;

        vector<int> dp(nums.size(), 1);
        int maxLen = 1;

        for (int i = 1; i < nums.size(); i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
            maxLen = max(maxLen, dp[i]);
        }

        return maxLen;
    }

    // Binary Search + DP approach (optimal)
    int lengthOfLISOptimal(vector<int>& nums) {
        vector<int> tails;

        for (int num : nums) {
            auto it = lower_bound(tails.begin(), tails.end(), num);

            if (it == tails.end()) {
                tails.push_back(num);
            } else {
                *it = num;
            }
        }

        return tails.size();
    }
};
