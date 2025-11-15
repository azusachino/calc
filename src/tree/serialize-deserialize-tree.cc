#include "../include/tree.h"
#include <string>
#include <sstream>
#include <queue>

using namespace std;

// LeetCode #297 - Serialize and Deserialize Binary Tree
// Encode a tree to a string and decode back to tree structure
// Time: O(n), Space: O(n)

class Codec {
public:
    // Encodes a tree to a single string
    string serialize(TreeNode* root) {
        if (root == nullptr) return "null";

        return to_string(root->val) + "," +
               serialize(root->left) + "," +
               serialize(root->right);
    }

    // Decodes your encoded data to tree
    TreeNode* deserialize(string data) {
        queue<string> nodes;
        stringstream ss(data);
        string item;

        while (getline(ss, item, ',')) {
            nodes.push(item);
        }

        return deserializeHelper(nodes);
    }

private:
    TreeNode* deserializeHelper(queue<string>& nodes) {
        string val = nodes.front();
        nodes.pop();

        if (val == "null") {
            return nullptr;
        }

        TreeNode* root = new TreeNode(stoi(val));
        root->left = deserializeHelper(nodes);
        root->right = deserializeHelper(nodes);

        return root;
    }
};

// Alternative: Level-order serialization
class Codec2 {
public:
    string serialize(TreeNode* root) {
        if (root == nullptr) return "";

        string result;
        queue<TreeNode*> q;
        q.push(root);

        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();

            if (node == nullptr) {
                result += "null,";
            } else {
                result += to_string(node->val) + ",";
                q.push(node->left);
                q.push(node->right);
            }
        }

        return result;
    }

    TreeNode* deserialize(string data) {
        if (data.empty()) return nullptr;

        stringstream ss(data);
        string item;
        getline(ss, item, ',');

        TreeNode* root = new TreeNode(stoi(item));
        queue<TreeNode*> q;
        q.push(root);

        while (!q.empty() && getline(ss, item, ',')) {
            TreeNode* node = q.front();
            q.pop();

            if (item != "null") {
                node->left = new TreeNode(stoi(item));
                q.push(node->left);
            }

            if (getline(ss, item, ',') && item != "null") {
                node->right = new TreeNode(stoi(item));
                q.push(node->right);
            }
        }

        return root;
    }
};
