#include <vector>
#include <algorithm>

using namespace std;

// LeetCode #42 - Trapping Rain Water
// Calculate how much water can be trapped after raining
// Time: O(n), Space: O(1)

class Solution {
public:
    // Two pointer approach - Most optimal
    int trap(vector<int>& height) {
        if (height.empty()) return 0;

        int left = 0, right = height.size() - 1;
        int leftMax = 0, rightMax = 0;
        int water = 0;

        while (left < right) {
            if (height[left] < height[right]) {
                if (height[left] >= leftMax) {
                    leftMax = height[left];
                } else {
                    water += leftMax - height[left];
                }
                left++;
            } else {
                if (height[right] >= rightMax) {
                    rightMax = height[right];
                } else {
                    water += rightMax - height[right];
                }
                right--;
            }
        }

        return water;
    }

    // Alternative: Dynamic Programming approach
    int trapDP(vector<int>& height) {
        if (height.empty()) return 0;
        int n = height.size();

        vector<int> leftMax(n), rightMax(n);

        leftMax[0] = height[0];
        for (int i = 1; i < n; i++) {
            leftMax[i] = max(leftMax[i-1], height[i]);
        }

        rightMax[n-1] = height[n-1];
        for (int i = n - 2; i >= 0; i--) {
            rightMax[i] = max(rightMax[i+1], height[i]);
        }

        int water = 0;
        for (int i = 0; i < n; i++) {
            water += min(leftMax[i], rightMax[i]) - height[i];
        }

        return water;
    }
};
