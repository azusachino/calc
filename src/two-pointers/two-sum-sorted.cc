#include <vector>

using namespace std;

// LeetCode #167 - Two Sum II - Input Array Is Sorted
// Find two numbers that add up to target
// Time: O(n), Space: O(1)

class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        int left = 0;
        int right = numbers.size() - 1;

        while (left < right) {
            int sum = numbers[left] + numbers[right];

            if (sum == target) {
                return {left + 1, right + 1}; // 1-indexed
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }

        return {};
    }
};

// LeetCode #15 - 3Sum (already exists but adding here for completeness)
// Find all unique triplets that sum to zero
// Time: O(n^2), Space: O(1) excluding output

class Solution2 {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> result;
        sort(nums.begin(), nums.end());

        for (int i = 0; i < nums.size(); i++) {
            if (i > 0 && nums[i] == nums[i-1]) continue;

            int left = i + 1;
            int right = nums.size() - 1;
            int target = -nums[i];

            while (left < right) {
                int sum = nums[left] + nums[right];

                if (sum == target) {
                    result.push_back({nums[i], nums[left], nums[right]});

                    while (left < right && nums[left] == nums[left+1]) left++;
                    while (left < right && nums[right] == nums[right-1]) right--;

                    left++;
                    right--;
                } else if (sum < target) {
                    left++;
                } else {
                    right--;
                }
            }
        }

        return result;
    }
};

// LeetCode #18 - 4Sum
// Find all unique quadruplets that sum to target
// Time: O(n^3), Space: O(1) excluding output

class Solution3 {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> result;
        sort(nums.begin(), nums.end());
        int n = nums.size();

        for (int i = 0; i < n - 3; i++) {
            if (i > 0 && nums[i] == nums[i-1]) continue;

            for (int j = i + 1; j < n - 2; j++) {
                if (j > i + 1 && nums[j] == nums[j-1]) continue;

                int left = j + 1;
                int right = n - 1;
                long long sum_target = (long long)target - nums[i] - nums[j];

                while (left < right) {
                    int sum = nums[left] + nums[right];

                    if (sum == sum_target) {
                        result.push_back({nums[i], nums[j], nums[left], nums[right]});

                        while (left < right && nums[left] == nums[left+1]) left++;
                        while (left < right && nums[right] == nums[right-1]) right--;

                        left++;
                        right--;
                    } else if (sum < sum_target) {
                        left++;
                    } else {
                        right--;
                    }
                }
            }
        }

        return result;
    }
};
