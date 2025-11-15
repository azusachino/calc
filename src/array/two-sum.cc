#include <vector>
#include <unordered_map>

using namespace std;

// LeetCode #1 - Two Sum
// Given an array of integers nums and an integer target,
// return indices of the two numbers such that they add up to target.
// Time: O(n), Space: O(n)

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> map;

        for (int i = 0; i < nums.size(); i++) {
            int complement = target - nums[i];

            if (map.find(complement) != map.end()) {
                return {map[complement], i};
            }

            map[nums[i]] = i;
        }

        return {};
    }
};
