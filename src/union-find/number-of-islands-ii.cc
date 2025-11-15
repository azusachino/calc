#include "../../include/union-find.h"
#include <vector>

using namespace std;

// LeetCode #305 - Number of Islands II (Premium)
// Dynamically add islands and track count
// Time: O(k * α(m*n)), Space: O(m*n)
// where k = number of operations

class Solution {
public:
    vector<int> numIslands2(int m, int n, vector<vector<int>>& positions) {
        UnionFind uf(m * n);
        vector<int> result;
        vector<vector<bool>> grid(m, vector<bool>(n, false));

        int dirs[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int count = 0;

        for (const auto& pos : positions) {
            int x = pos[0];
            int y = pos[1];

            if (grid[x][y]) {
                result.push_back(count);
                continue;
            }

            grid[x][y] = true;
            count++;

            int id = x * n + y;
            uf.parent[id] = id; // Initialize

            // Check 4 neighbors
            for (auto& dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];

                if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny]) {
                    int nid = nx * n + ny;

                    if (uf.unite(id, nid)) {
                        count--;
                    }
                }
            }

            result.push_back(count);
        }

        return result;
    }
};

// Manual UnionFind implementation for this specific problem
class Solution2 {
private:
    class UnionFind {
    public:
        vector<int> parent;
        vector<int> rank;

        UnionFind(int n) : parent(n, -1), rank(n, 0) {}

        int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }

        bool unite(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);

            if (rootX == rootY) return false;

            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }

            return true;
        }
    };

public:
    vector<int> numIslands2(int m, int n, vector<vector<int>>& positions) {
        UnionFind uf(m * n);
        vector<int> result;
        vector<vector<bool>> grid(m, vector<bool>(n, false));

        int dirs[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int count = 0;

        for (const auto& pos : positions) {
            int x = pos[0];
            int y = pos[1];

            if (grid[x][y]) {
                result.push_back(count);
                continue;
            }

            grid[x][y] = true;
            count++;

            int id = x * n + y;
            uf.parent[id] = id;

            for (auto& dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];

                if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny]) {
                    int nid = nx * n + ny;
                    if (uf.unite(id, nid)) {
                        count--;
                    }
                }
            }

            result.push_back(count);
        }

        return result;
    }
};
