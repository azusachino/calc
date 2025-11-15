#include <vector>

using namespace std;

// LeetCode #26 - Remove Duplicates from Sorted Array
// Remove duplicates in-place
// Time: O(n), Space: O(1)

class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        if (nums.empty()) return 0;

        int slow = 0;

        for (int fast = 1; fast < nums.size(); fast++) {
            if (nums[fast] != nums[slow]) {
                slow++;
                nums[slow] = nums[fast];
            }
        }

        return slow + 1;
    }
};

// LeetCode #80 - Remove Duplicates from Sorted Array II
// Each element can appear at most twice
// Time: O(n), Space: O(1)

class Solution2 {
public:
    int removeDuplicates(vector<int>& nums) {
        if (nums.size() <= 2) return nums.size();

        int slow = 2;

        for (int fast = 2; fast < nums.size(); fast++) {
            if (nums[fast] != nums[slow - 2]) {
                nums[slow] = nums[fast];
                slow++;
            }
        }

        return slow;
    }
};

// LeetCode #27 - Remove Element
// Remove all instances of val in-place
// Time: O(n), Space: O(1)

class Solution3 {
public:
    int removeElement(vector<int>& nums, int val) {
        int slow = 0;

        for (int fast = 0; fast < nums.size(); fast++) {
            if (nums[fast] != val) {
                nums[slow] = nums[fast];
                slow++;
            }
        }

        return slow;
    }
};
