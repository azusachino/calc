#include <string>

using namespace std;

// LeetCode #5 - Longest Palindromic Substring
// Find the longest palindromic substring in s
// Time: O(n^2), Space: O(1)

class Solution {
public:
    string longestPalindrome(string s) {
        if (s.empty()) return "";

        int start = 0, maxLen = 0;

        for (int i = 0; i < s.length(); i++) {
            // Check for odd length palindrome
            int len1 = expandAroundCenter(s, i, i);
            // Check for even length palindrome
            int len2 = expandAroundCenter(s, i, i + 1);

            int len = max(len1, len2);
            if (len > maxLen) {
                maxLen = len;
                start = i - (len - 1) / 2;
            }
        }

        return s.substr(start, maxLen);
    }

private:
    int expandAroundCenter(const string& s, int left, int right) {
        while (left >= 0 && right < s.length() && s[left] == s[right]) {
            left--;
            right++;
        }
        return right - left - 1;
    }
};
