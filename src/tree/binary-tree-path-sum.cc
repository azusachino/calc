#include "../include/tree.h"
#include <vector>

using namespace std;

// LeetCode #112 - Path Sum
// Determine if tree has root-to-leaf path with given sum
// Time: O(n), Space: O(h)

class Solution {
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if (root == nullptr) return false;

        if (root->left == nullptr && root->right == nullptr) {
            return root->val == targetSum;
        }

        int remaining = targetSum - root->val;
        return hasPathSum(root->left, remaining) ||
               hasPathSum(root->right, remaining);
    }
};

// LeetCode #113 - Path Sum II
// Find all root-to-leaf paths with given sum
// Time: O(n), Space: O(n)

class Solution2 {
public:
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        vector<vector<int>> result;
        vector<int> path;
        findPaths(root, targetSum, path, result);
        return result;
    }

private:
    void findPaths(TreeNode* root, int targetSum,
                   vector<int>& path, vector<vector<int>>& result) {
        if (root == nullptr) return;

        path.push_back(root->val);

        if (root->left == nullptr && root->right == nullptr &&
            root->val == targetSum) {
            result.push_back(path);
        }

        findPaths(root->left, targetSum - root->val, path, result);
        findPaths(root->right, targetSum - root->val, path, result);

        path.pop_back();
    }
};

// LeetCode #437 - Path Sum III
// Find number of paths that sum to targetSum (not necessarily root-to-leaf)
// Time: O(n^2) worst case, Space: O(h)

class Solution3 {
public:
    int pathSum(TreeNode* root, long long targetSum) {
        if (root == nullptr) return 0;

        return countPaths(root, targetSum) +
               pathSum(root->left, targetSum) +
               pathSum(root->right, targetSum);
    }

private:
    int countPaths(TreeNode* node, long long targetSum) {
        if (node == nullptr) return 0;

        int count = (node->val == targetSum) ? 1 : 0;

        count += countPaths(node->left, targetSum - node->val);
        count += countPaths(node->right, targetSum - node->val);

        return count;
    }
};
