#include "../include/tree.h"

// LeetCode #226 - Invert Binary Tree
// Invert a binary tree (swap left and right children)
// Time: O(n), Space: O(h) where h is height

class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (root == nullptr) {
            return nullptr;
        }

        // Swap left and right children
        TreeNode* temp = root->left;
        root->left = root->right;
        root->right = temp;

        // Recursively invert subtrees
        invertTree(root->left);
        invertTree(root->right);

        return root;
    }
};
