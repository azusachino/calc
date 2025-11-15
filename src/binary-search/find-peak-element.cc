#include <vector>

using namespace std;

// LeetCode #162 - Find Peak Element
// Find a peak element (element greater than neighbors)
// Time: O(log n), Space: O(1)

class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        int left = 0;
        int right = nums.size() - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] > nums[mid + 1]) {
                // Peak is on the left (or mid is peak)
                right = mid;
            } else {
                // Peak is on the right
                left = mid + 1;
            }
        }

        return left;
    }
};

// LeetCode #852 - Peak Index in a Mountain Array
// Find the peak in a mountain array (strictly increasing then decreasing)
// Time: O(log n), Space: O(1)

class Solution2 {
public:
    int peakIndexInMountainArray(vector<int>& arr) {
        int left = 0;
        int right = arr.size() - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;

            if (arr[mid] < arr[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return left;
    }
};
