#include "../../include/union-find.h"
#include <vector>

using namespace std;

// LeetCode #684 - Redundant Connection
// Find edge that creates a cycle in undirected graph
// Time: O(n * α(n)), Space: O(n)

class Solution {
public:
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        UnionFind uf(n + 1); // 1-indexed

        for (const auto& edge : edges) {
            int u = edge[0];
            int v = edge[1];

            if (!uf.unite(u, v)) {
                return edge; // This edge creates a cycle
            }
        }

        return {};
    }
};

// LeetCode #685 - Redundant Connection II (Directed Graph)
// Time: O(n), Space: O(n)

class Solution2 {
public:
    vector<int> findRedundantDirectedConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        vector<int> parent(n + 1, 0);
        vector<int> candidate1, candidate2;

        // Find node with two parents
        for (const auto& edge : edges) {
            int u = edge[0], v = edge[1];

            if (parent[v] != 0) {
                candidate1 = {parent[v], v};
                candidate2 = edge;
                edge[1] = 0; // Mark as invalid
            } else {
                parent[v] = u;
            }
        }

        // Check for cycle
        UnionFind uf(n + 1);
        for (const auto& edge : edges) {
            if (edge[1] == 0) continue;

            int u = edge[0], v = edge[1];
            if (!uf.unite(u, v)) {
                return candidate1.empty() ? edge : candidate1;
            }
        }

        return candidate2;
    }
};
