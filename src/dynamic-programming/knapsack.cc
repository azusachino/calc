#include <vector>
#include <algorithm>

using namespace std;

// 0/1 Knapsack Problem
// Choose items to maximize value without exceeding weight capacity
// Time: O(n * W), Space: O(n * W) or O(W) optimized

class Solution {
public:
    // 2D DP approach
    int knapsack(vector<int>& weights, vector<int>& values, int capacity) {
        int n = weights.size();
        vector<vector<int>> dp(n + 1, vector<int>(capacity + 1, 0));

        for (int i = 1; i <= n; i++) {
            for (int w = 0; w <= capacity; w++) {
                if (weights[i-1] <= w) {
                    dp[i][w] = max(dp[i-1][w],
                                   dp[i-1][w - weights[i-1]] + values[i-1]);
                } else {
                    dp[i][w] = dp[i-1][w];
                }
            }
        }

        return dp[n][capacity];
    }

    // Space-optimized 1D DP
    int knapsackOptimized(vector<int>& weights, vector<int>& values, int capacity) {
        vector<int> dp(capacity + 1, 0);

        for (int i = 0; i < weights.size(); i++) {
            for (int w = capacity; w >= weights[i]; w--) {
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
            }
        }

        return dp[capacity];
    }
};

// LeetCode #416 - Partition Equal Subset Sum
// Check if array can be partitioned into two equal sum subsets
// Time: O(n * sum), Space: O(sum)

class Solution2 {
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }

        if (sum % 2 != 0) return false;

        int target = sum / 2;
        vector<bool> dp(target + 1, false);
        dp[0] = true;

        for (int num : nums) {
            for (int j = target; j >= num; j--) {
                dp[j] = dp[j] || dp[j - num];
            }
        }

        return dp[target];
    }
};
