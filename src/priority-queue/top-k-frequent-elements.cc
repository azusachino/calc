#include <vector>
#include <unordered_map>
#include <queue>
#include <algorithm>

using namespace std;

// LeetCode #347 - Top K Frequent Elements
// Return the k most frequent elements
// Time: O(n log k), Space: O(n)

class Solution {
public:
    // Heap approach
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> freq;
        for (int num : nums) {
            freq[num]++;
        }

        auto comp = [](pair<int, int>& a, pair<int, int>& b) {
            return a.second > b.second;
        };

        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(comp)> minHeap(comp);

        for (auto& p : freq) {
            minHeap.push(p);
            if (minHeap.size() > k) {
                minHeap.pop();
            }
        }

        vector<int> result;
        while (!minHeap.empty()) {
            result.push_back(minHeap.top().first);
            minHeap.pop();
        }

        return result;
    }

    // Bucket sort approach - O(n)
    vector<int> topKFrequentBucket(vector<int>& nums, int k) {
        unordered_map<int, int> freq;
        for (int num : nums) {
            freq[num]++;
        }

        vector<vector<int>> buckets(nums.size() + 1);
        for (auto& p : freq) {
            buckets[p.second].push_back(p.first);
        }

        vector<int> result;
        for (int i = buckets.size() - 1; i >= 0 && result.size() < k; i--) {
            for (int num : buckets[i]) {
                result.push_back(num);
                if (result.size() == k) break;
            }
        }

        return result;
    }
};
