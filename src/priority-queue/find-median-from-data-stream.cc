#include <queue>

using namespace std;

// LeetCode #295 - Find Median from Data Stream
// Design a data structure that supports adding numbers and finding median
// Time: O(log n) for addNum, O(1) for findMedian
// Space: O(n)

class MedianFinder {
private:
    priority_queue<int> maxHeap; // Left half (smaller numbers)
    priority_queue<int, vector<int>, greater<int>> minHeap; // Right half (larger numbers)

public:
    MedianFinder() {}

    void addNum(int num) {
        // Add to max heap first
        maxHeap.push(num);

        // Balance: move largest from max heap to min heap
        minHeap.push(maxHeap.top());
        maxHeap.pop();

        // Maintain size property: maxHeap.size() >= minHeap.size()
        if (maxHeap.size() < minHeap.size()) {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
    }

    double findMedian() {
        if (maxHeap.size() > minHeap.size()) {
            return maxHeap.top();
        }
        return (maxHeap.top() + minHeap.top()) / 2.0;
    }
};
