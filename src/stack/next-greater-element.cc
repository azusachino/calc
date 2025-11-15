#include <vector>
#include <stack>

using namespace std;

// LeetCode #496 - Next Greater Element I
// Find next greater element for each element
// Time: O(n + m), Space: O(m)

class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<int, int> nextGreater;
        stack<int> st;

        // Build next greater map for nums2
        for (int num : nums2) {
            while (!st.empty() && st.top() < num) {
                nextGreater[st.top()] = num;
                st.pop();
            }
            st.push(num);
        }

        vector<int> result;
        for (int num : nums1) {
            result.push_back(nextGreater.count(num) ? nextGreater[num] : -1);
        }

        return result;
    }
};

// LeetCode #503 - Next Greater Element II (Circular Array)
// Time: O(n), Space: O(n)

class Solution2 {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        int n = nums.size();
        vector<int> result(n, -1);
        stack<int> st;

        // Process array twice to handle circular
        for (int i = 0; i < 2 * n; i++) {
            int num = nums[i % n];

            while (!st.empty() && nums[st.top()] < num) {
                result[st.top()] = num;
                st.pop();
            }

            if (i < n) {
                st.push(i);
            }
        }

        return result;
    }
};

// LeetCode #739 - Daily Temperatures
// Find how many days until a warmer temperature
// Time: O(n), Space: O(n)

class Solution3 {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        vector<int> result(n, 0);
        stack<int> st; // Store indices

        for (int i = 0; i < n; i++) {
            while (!st.empty() && temperatures[st.top()] < temperatures[i]) {
                int idx = st.top();
                st.pop();
                result[idx] = i - idx;
            }
            st.push(i);
        }

        return result;
    }
};
