#include <vector>
#include <queue>

using namespace std;

// LeetCode #200 - Number of Islands
// Count number of islands (connected 1s) in 2D grid
// Time: O(m * n), Space: O(m * n) worst case for DFS/BFS

class Solution {
public:
    // DFS approach
    int numIslands(vector<vector<char>>& grid) {
        if (grid.empty()) return 0;

        int count = 0;
        int m = grid.size();
        int n = grid[0].size();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    dfs(grid, i, j);
                }
            }
        }

        return count;
    }

private:
    void dfs(vector<vector<char>>& grid, int i, int j) {
        if (i < 0 || i >= grid.size() ||
            j < 0 || j >= grid[0].size() ||
            grid[i][j] != '1') {
            return;
        }

        grid[i][j] = '0'; // Mark as visited

        dfs(grid, i + 1, j);
        dfs(grid, i - 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i, j - 1);
    }
};

// Alternative: BFS approach
class Solution2 {
public:
    int numIslands(vector<vector<char>>& grid) {
        if (grid.empty()) return 0;

        int count = 0;
        int m = grid.size();
        int n = grid[0].size();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    bfs(grid, i, j);
                }
            }
        }

        return count;
    }

private:
    void bfs(vector<vector<char>>& grid, int i, int j) {
        queue<pair<int, int>> q;
        q.push({i, j});
        grid[i][j] = '0';

        int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        while (!q.empty()) {
            auto [x, y] = q.front();
            q.pop();

            for (auto& dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];

                if (nx >= 0 && nx < grid.size() &&
                    ny >= 0 && ny < grid[0].size() &&
                    grid[nx][ny] == '1') {
                    grid[nx][ny] = '0';
                    q.push({nx, ny});
                }
            }
        }
    }
};
