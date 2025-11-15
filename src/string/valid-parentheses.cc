#include <string>
#include <stack>
#include <unordered_map>

using namespace std;

// LeetCode #20 - Valid Parentheses
// Determine if the input string has valid parentheses
// Time: O(n), Space: O(n)

class Solution {
public:
    bool isValid(string s) {
        stack<char> st;
        unordered_map<char, char> pairs = {
            {')', '('},
            {']', '['},
            {'}', '{'}
        };

        for (char c : s) {
            if (pairs.find(c) != pairs.end()) {
                // Closing bracket
                if (st.empty() || st.top() != pairs[c]) {
                    return false;
                }
                st.pop();
            } else {
                // Opening bracket
                st.push(c);
            }
        }

        return st.empty();
    }
};
