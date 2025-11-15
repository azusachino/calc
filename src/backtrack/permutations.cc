#include <vector>
#include <algorithm>

using namespace std;

// LeetCode #46 - Permutations
// Generate all permutations of distinct integers
// Time: O(n! * n), Space: O(n)

class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> result;
        vector<int> path;
        vector<bool> used(nums.size(), false);
        backtrack(nums, path, used, result);
        return result;
    }

private:
    void backtrack(vector<int>& nums, vector<int>& path,
                   vector<bool>& used, vector<vector<int>>& result) {
        if (path.size() == nums.size()) {
            result.push_back(path);
            return;
        }

        for (int i = 0; i < nums.size(); i++) {
            if (used[i]) continue;

            path.push_back(nums[i]);
            used[i] = true;

            backtrack(nums, path, used, result);

            path.pop_back();
            used[i] = false;
        }
    }
};

// LeetCode #47 - Permutations II (with duplicates)
// Time: O(n! * n), Space: O(n)

class Solution2 {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> result;
        vector<int> path;
        vector<bool> used(nums.size(), false);

        sort(nums.begin(), nums.end()); // Sort to handle duplicates

        backtrack(nums, path, used, result);
        return result;
    }

private:
    void backtrack(vector<int>& nums, vector<int>& path,
                   vector<bool>& used, vector<vector<int>>& result) {
        if (path.size() == nums.size()) {
            result.push_back(path);
            return;
        }

        for (int i = 0; i < nums.size(); i++) {
            if (used[i]) continue;

            // Skip duplicates
            if (i > 0 && nums[i] == nums[i-1] && !used[i-1]) {
                continue;
            }

            path.push_back(nums[i]);
            used[i] = true;

            backtrack(nums, path, used, result);

            path.pop_back();
            used[i] = false;
        }
    }
};

// LeetCode #31 - Next Permutation
// Find next lexicographically greater permutation
// Time: O(n), Space: O(1)

class Solution3 {
public:
    void nextPermutation(vector<int>& nums) {
        int n = nums.size();
        int i = n - 2;

        // Find first decreasing element from right
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }

        if (i >= 0) {
            // Find element just larger than nums[i]
            int j = n - 1;
            while (j >= 0 && nums[j] <= nums[i]) {
                j--;
            }
            swap(nums[i], nums[j]);
        }

        // Reverse the suffix
        reverse(nums.begin() + i + 1, nums.end());
    }
};
