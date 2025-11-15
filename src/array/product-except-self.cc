#include <vector>

using namespace std;

// LeetCode #238 - Product of Array Except Self
// Return an array where each element is the product of all other elements
// Time: O(n), Space: O(1) - output array doesn't count
// Note: Cannot use division

class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        vector<int> result(n, 1);

        // Calculate left products
        int leftProduct = 1;
        for (int i = 0; i < n; i++) {
            result[i] = leftProduct;
            leftProduct *= nums[i];
        }

        // Calculate right products and multiply with left
        int rightProduct = 1;
        for (int i = n - 1; i >= 0; i--) {
            result[i] *= rightProduct;
            rightProduct *= nums[i];
        }

        return result;
    }
};
