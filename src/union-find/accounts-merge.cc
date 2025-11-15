#include "../../include/union-find.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <algorithm>

using namespace std;

// LeetCode #721 - Accounts Merge
// Merge accounts with common emails
// Time: O(n * k * α(n)), Space: O(n * k)
// where n = accounts, k = emails per account

class Solution {
public:
    vector<vector<string>> accountsMerge(vector<vector<string>>& accounts) {
        int n = accounts.size();
        UnionFind uf(n);

        unordered_map<string, int> emailToAccount;

        // Union accounts with same email
        for (int i = 0; i < n; i++) {
            for (int j = 1; j < accounts[i].size(); j++) {
                const string& email = accounts[i][j];

                if (emailToAccount.count(email)) {
                    uf.unite(i, emailToAccount[email]);
                } else {
                    emailToAccount[email] = i;
                }
            }
        }

        // Group emails by root account
        unordered_map<int, set<string>> components;
        for (const auto& [email, account] : emailToAccount) {
            int root = uf.find(account);
            components[root].insert(email);
        }

        // Build result
        vector<vector<string>> result;
        for (const auto& [root, emails] : components) {
            vector<string> merged;
            merged.push_back(accounts[root][0]); // Name

            for (const string& email : emails) {
                merged.push_back(email);
            }

            result.push_back(merged);
        }

        return result;
    }
};
