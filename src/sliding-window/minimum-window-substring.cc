#include <string>
#include <unordered_map>
#include <climits>

using namespace std;

// LeetCode #76 - Minimum Window Substring
// Find minimum window in s that contains all characters in t
// Time: O(m + n), Space: O(k) where k = unique chars
// Lee215's template for sliding window

class Solution {
public:
    string minWindow(string s, string t) {
        if (s.empty() || t.empty()) return "";

        unordered_map<char, int> need, window;

        for (char c : t) {
            need[c]++;
        }

        int left = 0, right = 0;
        int valid = 0;
        int start = 0, minLen = INT_MAX;

        while (right < s.length()) {
            char c = s[right];
            right++;

            // Update window
            if (need.count(c)) {
                window[c]++;
                if (window[c] == need[c]) {
                    valid++;
                }
            }

            // Shrink window
            while (valid == need.size()) {
                // Update result
                if (right - left < minLen) {
                    start = left;
                    minLen = right - left;
                }

                char d = s[left];
                left++;

                if (need.count(d)) {
                    if (window[d] == need[d]) {
                        valid--;
                    }
                    window[d]--;
                }
            }
        }

        return minLen == INT_MAX ? "" : s.substr(start, minLen);
    }
};
