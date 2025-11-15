#include <vector>
#include <deque>
#include <algorithm>

using namespace std;

// LeetCode #1696 - Jump Game VI
// Maximum score from jumping with max k steps
// Time: O(n), Space: O(k)
// DP + Monotonic Deque optimization

class Solution {
public:
    int maxResult(vector<int>& nums, int k) {
        int n = nums.size();
        deque<int> dq; // Store indices, front has max value
        dq.push_back(0);

        for (int i = 1; i < n; i++) {
            // Remove indices outside window
            while (!dq.empty() && dq.front() < i - k) {
                dq.pop_front();
            }

            // Current max is nums[i] + dp[dq.front()]
            nums[i] = nums[i] + nums[dq.front()];

            // Maintain decreasing deque
            while (!dq.empty() && nums[dq.back()] <= nums[i]) {
                dq.pop_back();
            }

            dq.push_back(i);
        }

        return nums[n - 1];
    }
};
