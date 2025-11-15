#include <vector>
#include <deque>
#include <climits>

using namespace std;

// LeetCode #862 - Shortest Subarray with Sum at Least K
// Find shortest subarray with sum >= k (can have negative numbers)
// Time: O(n), Space: O(n)
// Advanced monotonic deque with prefix sum

class Solution {
public:
    int shortestSubarray(vector<int>& nums, int k) {
        int n = nums.size();
        vector<long long> prefix(n + 1, 0);

        // Build prefix sum
        for (int i = 0; i < n; i++) {
            prefix[i + 1] = prefix[i] + nums[i];
        }

        int result = INT_MAX;
        deque<int> dq; // Monotonic increasing deque of indices

        for (int i = 0; i <= n; i++) {
            // Check if we can form valid subarray
            while (!dq.empty() && prefix[i] - prefix[dq.front()] >= k) {
                result = min(result, i - dq.front());
                dq.pop_front();
            }

            // Maintain monotonic increasing property
            while (!dq.empty() && prefix[i] <= prefix[dq.back()]) {
                dq.pop_back();
            }

            dq.push_back(i);
        }

        return result == INT_MAX ? -1 : result;
    }
};

// LeetCode #1438 - Longest Continuous Subarray With Absolute Diff <= Limit
// Time: O(n), Space: O(n)

class Solution2 {
public:
    int longestSubarray(vector<int>& nums, int limit) {
        deque<int> maxDq, minDq; // Store indices
        int left = 0;
        int maxLen = 0;

        for (int right = 0; right < nums.size(); right++) {
            // Maintain max deque (decreasing)
            while (!maxDq.empty() && nums[maxDq.back()] < nums[right]) {
                maxDq.pop_back();
            }
            maxDq.push_back(right);

            // Maintain min deque (increasing)
            while (!minDq.empty() && nums[minDq.back()] > nums[right]) {
                minDq.pop_back();
            }
            minDq.push_back(right);

            // Shrink window if difference exceeds limit
            while (nums[maxDq.front()] - nums[minDq.front()] > limit) {
                if (maxDq.front() == left) maxDq.pop_front();
                if (minDq.front() == left) minDq.pop_front();
                left++;
            }

            maxLen = max(maxLen, right - left + 1);
        }

        return maxLen;
    }
};
