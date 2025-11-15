#include <vector>
#include <queue>
#include <algorithm>

using namespace std;

// LeetCode #215 - Kth Largest Element in an Array
// Find the kth largest element in unsorted array
// Time: O(n log k) for heap, O(n) average for quickselect
// Space: O(k) for heap, O(1) for quickselect

class Solution {
public:
    // Min heap approach
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int, vector<int>, greater<int>> minHeap;

        for (int num : nums) {
            minHeap.push(num);
            if (minHeap.size() > k) {
                minHeap.pop();
            }
        }

        return minHeap.top();
    }

    // Quickselect approach (optimal)
    int findKthLargestQuickSelect(vector<int>& nums, int k) {
        return quickSelect(nums, 0, nums.size() - 1, nums.size() - k);
    }

private:
    int quickSelect(vector<int>& nums, int left, int right, int k) {
        if (left == right) return nums[left];

        int pivotIndex = partition(nums, left, right);

        if (k == pivotIndex) {
            return nums[k];
        } else if (k < pivotIndex) {
            return quickSelect(nums, left, pivotIndex - 1, k);
        } else {
            return quickSelect(nums, pivotIndex + 1, right, k);
        }
    }

    int partition(vector<int>& nums, int left, int right) {
        int pivot = nums[right];
        int i = left;

        for (int j = left; j < right; j++) {
            if (nums[j] <= pivot) {
                swap(nums[i], nums[j]);
                i++;
            }
        }

        swap(nums[i], nums[right]);
        return i;
    }
};
