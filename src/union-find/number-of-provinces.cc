#include "../../include/union-find.h"
#include <vector>

using namespace std;

// LeetCode #547 - Number of Provinces
// Find number of connected components in graph
// Time: O(n^2 * α(n)), Space: O(n)

class Solution {
public:
    int findCircleNum(vector<vector<int>>& isConnected) {
        int n = isConnected.size();
        UnionFind uf(n);

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (isConnected[i][j] == 1) {
                    uf.unite(i, j);
                }
            }
        }

        return uf.getCount();
    }
};
