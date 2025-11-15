#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

// Lee215's Sliding Window Template
// Works for many sliding window problems

class SlidingWindowTemplate {
public:
    // Template for finding substring
    string slidingWindow(string s, string t) {
        unordered_map<char, int> need, window;

        for (char c : t) need[c]++;

        int left = 0, right = 0;
        int valid = 0;

        while (right < s.length()) {
            char c = s[right];
            right++;

            // Update window data
            if (need.count(c)) {
                window[c]++;
                if (window[c] == need[c]) {
                    valid++;
                }
            }

            // Shrink window
            while (/* window needs shrinking */) {
                // Update result here if needed

                char d = s[left];
                left++;

                // Update window data
                if (need.count(d)) {
                    if (window[d] == need[d]) {
                        valid--;
                    }
                    window[d]--;
                }
            }
        }

        return "";
    }
};

// LeetCode #438 - Find All Anagrams in a String
// Time: O(n), Space: O(1)

class Solution1 {
public:
    vector<int> findAnagrams(string s, string p) {
        unordered_map<char, int> need, window;
        for (char c : p) need[c]++;

        int left = 0, right = 0;
        int valid = 0;
        vector<int> result;

        while (right < s.length()) {
            char c = s[right];
            right++;

            if (need.count(c)) {
                window[c]++;
                if (window[c] == need[c]) {
                    valid++;
                }
            }

            // Shrink when window size equals p length
            while (right - left >= p.length()) {
                if (valid == need.size()) {
                    result.push_back(left);
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

        return result;
    }
};

// LeetCode #567 - Permutation in String
// Time: O(n), Space: O(1)

class Solution2 {
public:
    bool checkInclusion(string s1, string s2) {
        unordered_map<char, int> need, window;
        for (char c : s1) need[c]++;

        int left = 0, right = 0;
        int valid = 0;

        while (right < s2.length()) {
            char c = s2[right];
            right++;

            if (need.count(c)) {
                window[c]++;
                if (window[c] == need[c]) {
                    valid++;
                }
            }

            while (right - left >= s1.length()) {
                if (valid == need.size()) {
                    return true;
                }

                char d = s2[left];
                left++;

                if (need.count(d)) {
                    if (window[d] == need[d]) {
                        valid--;
                    }
                    window[d]--;
                }
            }
        }

        return false;
    }
};

// LeetCode #424 - Longest Repeating Character Replacement
// Time: O(n), Space: O(1)

class Solution3 {
public:
    int characterReplacement(string s, int k) {
        unordered_map<char, int> count;
        int left = 0;
        int maxCount = 0;
        int maxLen = 0;

        for (int right = 0; right < s.length(); right++) {
            maxCount = max(maxCount, ++count[s[right]]);

            // If replacements needed > k, shrink window
            while (right - left + 1 - maxCount > k) {
                count[s[left]]--;
                left++;
            }

            maxLen = max(maxLen, right - left + 1);
        }

        return maxLen;
    }
};
