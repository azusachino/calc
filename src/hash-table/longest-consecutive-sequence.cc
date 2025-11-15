#include <vector>
#include <unordered_set>
#include <algorithm>

using namespace std;

// LeetCode #128 - Longest Consecutive Sequence
// Find length of longest consecutive sequence in unsorted array
// Time: O(n), Space: O(n)

class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        if (nums.empty()) return 0;

        unordered_set<int> numSet(nums.begin(), nums.end());
        int maxLength = 0;

        for (int num : numSet) {
            // Only start counting from the beginning of a sequence
            if (numSet.find(num - 1) == numSet.end()) {
                int currentNum = num;
                int currentLength = 1;

                while (numSet.find(currentNum + 1) != numSet.end()) {
                    currentNum++;
                    currentLength++;
                }

                maxLength = max(maxLength, currentLength);
            }
        }

        return maxLength;
    }
};
