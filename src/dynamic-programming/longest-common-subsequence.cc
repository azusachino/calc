#include <string>
#include <vector>
#include <algorithm>

using namespace std;

// LeetCode #1143 - Longest Common Subsequence
// Find length of longest common subsequence between two strings
// Time: O(m * n), Space: O(m * n)

class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.length();
        int n = text2.length();

        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1[i-1] == text2[j-1]) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                } else {
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }

        return dp[m][n];
    }

    // Space-optimized version
    int longestCommonSubsequenceOptimized(string text1, string text2) {
        int m = text1.length();
        int n = text2.length();

        vector<int> prev(n + 1, 0);
        vector<int> curr(n + 1, 0);

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1[i-1] == text2[j-1]) {
                    curr[j] = prev[j-1] + 1;
                } else {
                    curr[j] = max(prev[j], curr[j-1]);
                }
            }
            swap(prev, curr);
        }

        return prev[n];
    }
};

// LeetCode #72 - Edit Distance
// Minimum operations to convert word1 to word2
// Time: O(m * n), Space: O(m * n)

class Solution2 {
public:
    int minDistance(string word1, string word2) {
        int m = word1.length();
        int n = word2.length();

        vector<vector<int>> dp(m + 1, vector<int>(n + 1));

        // Initialize base cases
        for (int i = 0; i <= m; i++) dp[i][0] = i;
        for (int j = 0; j <= n; j++) dp[0][j] = j;

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1[i-1] == word2[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                } else {
                    dp[i][j] = 1 + min({dp[i-1][j],    // delete
                                        dp[i][j-1],    // insert
                                        dp[i-1][j-1]}); // replace
                }
            }
        }

        return dp[m][n];
    }
};
