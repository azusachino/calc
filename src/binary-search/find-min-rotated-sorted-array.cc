#include <vector>

using namespace std;

// LeetCode #153 - Find Minimum in Rotated Sorted Array
// Find the minimum element in rotated sorted array (no duplicates)
// Time: O(log n), Space: O(1)

class Solution {
public:
    int findMin(vector<int>& nums) {
        int left = 0;
        int right = nums.size() - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] > nums[right]) {
                // Minimum is in right half
                left = mid + 1;
            } else {
                // Minimum is in left half or mid
                right = mid;
            }
        }

        return nums[left];
    }
};

// LeetCode #154 - Find Minimum in Rotated Sorted Array II
// Find minimum with duplicates allowed
// Time: O(log n) average, O(n) worst case, Space: O(1)

class Solution2 {
public:
    int findMin(vector<int>& nums) {
        int left = 0;
        int right = nums.size() - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else if (nums[mid] < nums[right]) {
                right = mid;
            } else {
                // nums[mid] == nums[right], can't determine which side
                right--;
            }
        }

        return nums[left];
    }
};
