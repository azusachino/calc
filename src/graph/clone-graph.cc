#include <vector>
#include <unordered_map>
#include <queue>

using namespace std;

// Definition for a Node
class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node() {
        val = 0;
        neighbors = vector<Node*>();
    }
    Node(int _val) {
        val = _val;
        neighbors = vector<Node*>();
    }
    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};

// LeetCode #133 - Clone Graph
// Deep copy of an undirected graph
// Time: O(V + E), Space: O(V)

class Solution {
public:
    // DFS approach
    Node* cloneGraph(Node* node) {
        if (node == nullptr) return nullptr;

        unordered_map<Node*, Node*> visited;
        return dfs(node, visited);
    }

private:
    Node* dfs(Node* node, unordered_map<Node*, Node*>& visited) {
        if (visited.count(node)) {
            return visited[node];
        }

        Node* clone = new Node(node->val);
        visited[node] = clone;

        for (Node* neighbor : node->neighbors) {
            clone->neighbors.push_back(dfs(neighbor, visited));
        }

        return clone;
    }
};

// Alternative: BFS approach
class Solution2 {
public:
    Node* cloneGraph(Node* node) {
        if (node == nullptr) return nullptr;

        unordered_map<Node*, Node*> visited;
        queue<Node*> q;

        Node* clone = new Node(node->val);
        visited[node] = clone;
        q.push(node);

        while (!q.empty()) {
            Node* curr = q.front();
            q.pop();

            for (Node* neighbor : curr->neighbors) {
                if (!visited.count(neighbor)) {
                    visited[neighbor] = new Node(neighbor->val);
                    q.push(neighbor);
                }
                visited[curr]->neighbors.push_back(visited[neighbor]);
            }
        }

        return clone;
    }
};
