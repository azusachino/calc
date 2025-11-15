#include "../include/list.h"
#include <vector>
#include <queue>

using namespace std;

// LeetCode #23 - Merge k Sorted Lists (moved from linked-list to priority-queue)
// This is primarily a heap problem, so keeping it here as well
// Time: O(N log k) where N is total nodes, k is number of lists
// Space: O(k)

class Solution {
public:
    struct Compare {
        bool operator()(ListNode* a, ListNode* b) {
            return a->val > b->val;
        }
    };

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        priority_queue<ListNode*, vector<ListNode*>, Compare> minHeap;

        for (ListNode* list : lists) {
            if (list != nullptr) {
                minHeap.push(list);
            }
        }

        ListNode dummy(0);
        ListNode* tail = &dummy;

        while (!minHeap.empty()) {
            ListNode* node = minHeap.top();
            minHeap.pop();

            tail->next = node;
            tail = tail->next;

            if (node->next != nullptr) {
                minHeap.push(node->next);
            }
        }

        return dummy.next;
    }
};
