#include "../include/tree.h"
#include <climits>

// LeetCode #98 - Validate Binary Search Tree
// Check if a binary tree is a valid BST
// Time: O(n), Space: O(h)

class Solution {
public:
    bool isValidBST(TreeNode* root) {
        return validate(root, LLONG_MIN, LLONG_MAX);
    }

private:
    bool validate(TreeNode* node, long long minVal, long long maxVal) {
        if (node == nullptr) return true;

        if (node->val <= minVal || node->val >= maxVal) {
            return false;
        }

        return validate(node->left, minVal, node->val) &&
               validate(node->right, node->val, maxVal);
    }
};

// Alternative: In-order traversal approach
class Solution2 {
public:
    bool isValidBST(TreeNode* root) {
        long long prev = LLONG_MIN;
        return inorder(root, prev);
    }

private:
    bool inorder(TreeNode* node, long long& prev) {
        if (node == nullptr) return true;

        if (!inorder(node->left, prev)) return false;

        if (node->val <= prev) return false;
        prev = node->val;

        return inorder(node->right, prev);
    }
};
