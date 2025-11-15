#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>

using namespace std;

// LeetCode #621 - Task Scheduler
// Find minimum time needed to execute all tasks with cooldown
// Time: O(n), Space: O(1) - at most 26 different tasks

class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
        // Count frequency of each task
        unordered_map<char, int> freq;
        for (char task : tasks) {
            freq[task]++;
        }

        // Max heap based on frequency
        priority_queue<int> maxHeap;
        for (auto& p : freq) {
            maxHeap.push(p.second);
        }

        int time = 0;

        while (!maxHeap.empty()) {
            vector<int> temp;
            int cycle = n + 1;

            // Execute tasks in current cycle
            for (int i = 0; i < cycle && !maxHeap.empty(); i++) {
                int count = maxHeap.top();
                maxHeap.pop();

                if (count > 1) {
                    temp.push_back(count - 1);
                }

                time++;
            }

            // Put remaining tasks back
            for (int count : temp) {
                maxHeap.push(count);
            }

            // If heap is empty, we're done
            // Otherwise, add idle time if needed
            if (!maxHeap.empty()) {
                time += max(0, cycle - (int)temp.size() - 1);
            }
        }

        return time;
    }

    // Alternative: Math approach
    int leastIntervalMath(vector<char>& tasks, int n) {
        vector<int> freq(26, 0);
        for (char task : tasks) {
            freq[task - 'A']++;
        }

        int maxFreq = *max_element(freq.begin(), freq.end());
        int maxCount = count(freq.begin(), freq.end(), maxFreq);

        return max((int)tasks.size(), (maxFreq - 1) * (n + 1) + maxCount);
    }
};
