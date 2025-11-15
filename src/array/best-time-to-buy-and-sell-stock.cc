#include <vector>
#include <algorithm>

using namespace std;

// LeetCode #121 - Best Time to Buy and Sell Stock
// Find the maximum profit from buying and selling stock once
// Time: O(n), Space: O(1)

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.empty()) return 0;

        int minPrice = prices[0];
        int maxProfit = 0;

        for (int i = 1; i < prices.size(); i++) {
            maxProfit = max(maxProfit, prices[i] - minPrice);
            minPrice = min(minPrice, prices[i]);
        }

        return maxProfit;
    }
};
