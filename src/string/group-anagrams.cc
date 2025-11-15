#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

using namespace std;

// LeetCode #49 - Group Anagrams
// Group strings that are anagrams of each other
// Time: O(n * k log k) where k is max length of string
// Space: O(n * k)

class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> groups;

        for (const string& str : strs) {
            string key = str;
            sort(key.begin(), key.end());
            groups[key].push_back(str);
        }

        vector<vector<string>> result;
        for (auto& pair : groups) {
            result.push_back(pair.second);
        }

        return result;
    }

    // Alternative: Using character count as key
    vector<vector<string>> groupAnagramsCount(vector<string>& strs) {
        unordered_map<string, vector<string>> groups;

        for (const string& str : strs) {
            string key = getKey(str);
            groups[key].push_back(str);
        }

        vector<vector<string>> result;
        for (auto& pair : groups) {
            result.push_back(pair.second);
        }

        return result;
    }

private:
    string getKey(const string& str) {
        vector<int> count(26, 0);
        for (char c : str) {
            count[c - 'a']++;
        }

        string key;
        for (int i = 0; i < 26; i++) {
            if (count[i] > 0) {
                key += string(count[i], 'a' + i);
            }
        }
        return key;
    }
};
