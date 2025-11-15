#include <vector>
#include <deque>

using namespace std;

// LeetCode #239 - Sliding Window Maximum
// Find maximum in each sliding window of size k
// Time: O(n), Space: O(k)
// Lee215's elegant monotonic deque approach

class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> result;
        deque<int> dq; // Store indices

        for (int i = 0; i < nums.size(); i++) {
            // Remove indices outside the window
            if (!dq.empty() && dq.front() <= i - k) {
                dq.pop_front();
            }

            // Remove smaller elements from back (they can't be max)
            while (!dq.empty() && nums[dq.back()] < nums[i]) {
                dq.pop_back();
            }

            dq.push_back(i);

            // Add to result once we have a full window
            if (i >= k - 1) {
                result.push_back(nums[dq.front()]);
            }
        }

        return result;
    }
};
