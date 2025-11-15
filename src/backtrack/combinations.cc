#include <vector>

using namespace std;

// LeetCode #77 - Combinations
// Generate all combinations of k numbers from 1 to n
// Time: O(C(n,k) * k), Space: O(k)

class Solution {
public:
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> result;
        vector<int> path;
        backtrack(n, k, 1, path, result);
        return result;
    }

private:
    void backtrack(int n, int k, int start,
                   vector<int>& path, vector<vector<int>>& result) {
        if (path.size() == k) {
            result.push_back(path);
            return;
        }

        // Pruning: if remaining elements < needed elements, stop
        for (int i = start; i <= n - (k - path.size()) + 1; i++) {
            path.push_back(i);
            backtrack(n, k, i + 1, path, result);
            path.pop_back();
        }
    }
};

// LeetCode #39 - Combination Sum
// Find all combinations that sum to target (can reuse elements)
// Time: O(n^(T/M)) where T=target, M=min candidate
// Space: O(T/M)

class Solution2 {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> result;
        vector<int> path;
        backtrack(candidates, target, 0, path, result);
        return result;
    }

private:
    void backtrack(vector<int>& candidates, int target, int start,
                   vector<int>& path, vector<vector<int>>& result) {
        if (target == 0) {
            result.push_back(path);
            return;
        }

        if (target < 0) return;

        for (int i = start; i < candidates.size(); i++) {
            path.push_back(candidates[i]);
            backtrack(candidates, target - candidates[i], i, path, result);
            path.pop_back();
        }
    }
};

// LeetCode #40 - Combination Sum II
// Each number used at most once, input may have duplicates
// Time: O(2^n), Space: O(n)

class Solution3 {
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<vector<int>> result;
        vector<int> path;

        sort(candidates.begin(), candidates.end());

        backtrack(candidates, target, 0, path, result);
        return result;
    }

private:
    void backtrack(vector<int>& candidates, int target, int start,
                   vector<int>& path, vector<vector<int>>& result) {
        if (target == 0) {
            result.push_back(path);
            return;
        }

        for (int i = start; i < candidates.size(); i++) {
            if (candidates[i] > target) break;

            // Skip duplicates
            if (i > start && candidates[i] == candidates[i-1]) {
                continue;
            }

            path.push_back(candidates[i]);
            backtrack(candidates, target - candidates[i], i + 1, path, result);
            path.pop_back();
        }
    }
};

// LeetCode #216 - Combination Sum III
// Find k numbers that sum to n, using 1-9 only once
// Time: O(C(9,k)), Space: O(k)

class Solution4 {
public:
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<vector<int>> result;
        vector<int> path;
        backtrack(k, n, 1, path, result);
        return result;
    }

private:
    void backtrack(int k, int n, int start,
                   vector<int>& path, vector<vector<int>>& result) {
        if (path.size() == k) {
            if (n == 0) {
                result.push_back(path);
            }
            return;
        }

        for (int i = start; i <= 9; i++) {
            if (i > n) break;

            path.push_back(i);
            backtrack(k, n - i, i + 1, path, result);
            path.pop_back();
        }
    }
};
