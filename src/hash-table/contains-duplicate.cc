#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cmath>

using namespace std;

// LeetCode #217 - Contains Duplicate
// Check if array contains any duplicates
// Time: O(n), Space: O(n)

class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        unordered_set<int> seen;

        for (int num : nums) {
            if (seen.count(num)) {
                return true;
            }
            seen.insert(num);
        }

        return false;
    }
};

// LeetCode #219 - Contains Duplicate II
// Check if there are duplicates within k distance
// Time: O(n), Space: O(min(n,k))

class Solution2 {
public:
    bool containsNearbyDuplicate(vector<int>& nums, int k) {
        unordered_map<int, int> indexMap;

        for (int i = 0; i < nums.size(); i++) {
            if (indexMap.count(nums[i]) && i - indexMap[nums[i]] <= k) {
                return true;
            }
            indexMap[nums[i]] = i;
        }

        return false;
    }
};

// LeetCode #220 - Contains Duplicate III
// Check if there are two distinct indices i and j such that
// abs(nums[i] - nums[j]) <= valueDiff and abs(i - j) <= indexDiff
// Time: O(n log k), Space: O(k)

class Solution3 {
public:
    bool containsNearbyAlmostDuplicate(vector<int>& nums, int indexDiff, int valueDiff) {
        if (indexDiff <= 0 || valueDiff < 0) return false;

        unordered_map<long long, long long> buckets;
        long long bucketSize = (long long)valueDiff + 1;

        for (int i = 0; i < nums.size(); i++) {
            long long bucket = getBucket(nums[i], bucketSize);

            // Check current bucket
            if (buckets.count(bucket)) {
                return true;
            }

            // Check adjacent buckets
            if (buckets.count(bucket - 1) &&
                abs(nums[i] - buckets[bucket - 1]) <= valueDiff) {
                return true;
            }

            if (buckets.count(bucket + 1) &&
                abs(nums[i] - buckets[bucket + 1]) <= valueDiff) {
                return true;
            }

            buckets[bucket] = nums[i];

            // Remove old bucket
            if (i >= indexDiff) {
                buckets.erase(getBucket(nums[i - indexDiff], bucketSize));
            }
        }

        return false;
    }

private:
    long long getBucket(long long num, long long bucketSize) {
        return num >= 0 ? num / bucketSize : (num + 1) / bucketSize - 1;
    }
};
