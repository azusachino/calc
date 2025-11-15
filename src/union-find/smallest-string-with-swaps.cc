#include "../../include/union-find.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace std;

// LeetCode #1202 - Smallest String With Swaps
// Find lexicographically smallest string with allowed swaps
// Time: O(n log n + E * α(n)), Space: O(n)

class Solution {
public:
    string smallestStringWithSwaps(string s, vector<vector<int>>& pairs) {
        int n = s.length();
        UnionFind uf(n);

        // Union connected indices
        for (const auto& pair : pairs) {
            uf.unite(pair[0], pair[1]);
        }

        // Group indices by component
        unordered_map<int, vector<int>> components;
        for (int i = 0; i < n; i++) {
            components[uf.find(i)].push_back(i);
        }

        string result = s;

        // Sort characters within each component
        for (const auto& [root, indices] : components) {
            // Get characters at these indices
            string chars;
            for (int idx : indices) {
                chars += s[idx];
            }

            // Sort characters
            sort(chars.begin(), chars.end());

            // Assign sorted characters back
            vector<int> sortedIndices = indices;
            sort(sortedIndices.begin(), sortedIndices.end());

            for (int i = 0; i < sortedIndices.size(); i++) {
                result[sortedIndices[i]] = chars[i];
            }
        }

        return result;
    }
};
