#include <string>
#include <unordered_set>
#include <algorithm>

using namespace std;

// LeetCode #3 - Longest Substring Without Repeating Characters
// Find the length of the longest substring without repeating characters
// Time: O(n), Space: O(min(m,n)) where m is charset size

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_set<char> charSet;
        int left = 0;
        int maxLength = 0;

        for (int right = 0; right < s.length(); right++) {
            while (charSet.find(s[right]) != charSet.end()) {
                charSet.erase(s[left]);
                left++;
            }
            charSet.insert(s[right]);
            maxLength = max(maxLength, right - left + 1);
        }

        return maxLength;
    }

    // Alternative: Using sliding window with map
    int lengthOfLongestSubstringMap(string s) {
        unordered_map<char, int> charIndex;
        int maxLength = 0;
        int left = 0;

        for (int right = 0; right < s.length(); right++) {
            if (charIndex.find(s[right]) != charIndex.end()) {
                left = max(left, charIndex[s[right]] + 1);
            }
            charIndex[s[right]] = right;
            maxLength = max(maxLength, right - left + 1);
        }

        return maxLength;
    }
};
