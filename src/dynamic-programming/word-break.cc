#include <string>
#include <vector>
#include <unordered_set>

using namespace std;

// LeetCode #139 - Word Break
// Check if string can be segmented into dictionary words
// Time: O(n^2 * m) where m is average word length
// Space: O(n)

class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> dict(wordDict.begin(), wordDict.end());
        vector<bool> dp(s.length() + 1, false);
        dp[0] = true;

        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && dict.count(s.substr(j, i - j))) {
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[s.length()];
    }
};

// LeetCode #140 - Word Break II
// Return all possible sentences
// Time: O(2^n) in worst case, Space: O(2^n)

class Solution2 {
public:
    vector<string> wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> dict(wordDict.begin(), wordDict.end());
        vector<string> result;
        string current;
        backtrack(s, 0, dict, current, result);
        return result;
    }

private:
    void backtrack(const string& s, int start, unordered_set<string>& dict,
                   string& current, vector<string>& result) {
        if (start == s.length()) {
            result.push_back(current.substr(0, current.length() - 1)); // Remove trailing space
            return;
        }

        for (int end = start + 1; end <= s.length(); end++) {
            string word = s.substr(start, end - start);
            if (dict.count(word)) {
                int len = current.length();
                current += word + " ";
                backtrack(s, end, dict, current, result);
                current.resize(len);
            }
        }
    }
};
