#include <vector>
#include <algorithm>

using namespace std;

// LeetCode #75 - Sort Colors (Dutch National Flag)
// Sort array with values 0, 1, 2 in one pass
// Time: O(n), Space: O(1)

class Solution {
public:
    void sortColors(vector<int>& nums) {
        int left = 0;       // Next position for 0
        int right = nums.size() - 1;  // Next position for 2
        int curr = 0;

        while (curr <= right) {
            if (nums[curr] == 0) {
                swap(nums[left], nums[curr]);
                left++;
                curr++;
            } else if (nums[curr] == 2) {
                swap(nums[curr], nums[right]);
                right--;
                // Don't increment curr, need to check swapped value
            } else {
                curr++;
            }
        }
    }
};

// LeetCode #125 - Valid Palindrome
// Check if string is palindrome (alphanumeric only)
// Time: O(n), Space: O(1)

class Solution2 {
public:
    bool isPalindrome(string s) {
        int left = 0;
        int right = s.length() - 1;

        while (left < right) {
            while (left < right && !isalnum(s[left])) {
                left++;
            }
            while (left < right && !isalnum(s[right])) {
                right--;
            }

            if (tolower(s[left]) != tolower(s[right])) {
                return false;
            }

            left++;
            right--;
        }

        return true;
    }
};

// LeetCode #344 - Reverse String
// Reverse string in-place
// Time: O(n), Space: O(1)

class Solution3 {
public:
    void reverseString(vector<char>& s) {
        int left = 0;
        int right = s.size() - 1;

        while (left < right) {
            swap(s[left], s[right]);
            left++;
            right--;
        }
    }
};
