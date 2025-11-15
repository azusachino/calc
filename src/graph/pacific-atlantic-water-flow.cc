#include <vector>
#include <queue>

using namespace std;

// LeetCode #417 - Pacific Atlantic Water Flow
// Find cells where water can flow to both Pacific and Atlantic oceans
// Time: O(m * n), Space: O(m * n)

class Solution {
public:
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
        if (heights.empty()) return {};

        int m = heights.size();
        int n = heights[0].size();

        vector<vector<bool>> pacific(m, vector<bool>(n, false));
        vector<vector<bool>> atlantic(m, vector<bool>(n, false));

        // DFS from Pacific border (top and left)
        for (int i = 0; i < m; i++) {
            dfs(heights, pacific, i, 0);
            dfs(heights, atlantic, i, n - 1);
        }

        for (int j = 0; j < n; j++) {
            dfs(heights, pacific, 0, j);
            dfs(heights, atlantic, m - 1, j);
        }

        // Find cells reachable by both
        vector<vector<int>> result;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (pacific[i][j] && atlantic[i][j]) {
                    result.push_back({i, j});
                }
            }
        }

        return result;
    }

private:
    void dfs(vector<vector<int>>& heights, vector<vector<bool>>& visited,
             int i, int j) {
        visited[i][j] = true;

        int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        for (auto& dir : dirs) {
            int ni = i + dir[0];
            int nj = j + dir[1];

            if (ni >= 0 && ni < heights.size() &&
                nj >= 0 && nj < heights[0].size() &&
                !visited[ni][nj] &&
                heights[ni][nj] >= heights[i][j]) {
                dfs(heights, visited, ni, nj);
            }
        }
    }
};
