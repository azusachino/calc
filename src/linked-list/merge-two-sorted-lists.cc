#include "../include/list.h"

// LeetCode #21 - Merge Two Sorted Lists
// Merge two sorted linked lists into one sorted list
// Time: O(n + m), Space: O(1) for iterative, O(n + m) for recursive

class Solution {
public:
    // Iterative approach
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode dummy(0);
        ListNode* tail = &dummy;

        while (list1 != nullptr && list2 != nullptr) {
            if (list1->val <= list2->val) {
                tail->next = list1;
                list1 = list1->next;
            } else {
                tail->next = list2;
                list2 = list2->next;
            }
            tail = tail->next;
        }

        tail->next = (list1 != nullptr) ? list1 : list2;

        return dummy.next;
    }

    // Recursive approach
    ListNode* mergeTwoListsRecursive(ListNode* list1, ListNode* list2) {
        if (list1 == nullptr) return list2;
        if (list2 == nullptr) return list1;

        if (list1->val <= list2->val) {
            list1->next = mergeTwoListsRecursive(list1->next, list2);
            return list1;
        } else {
            list2->next = mergeTwoListsRecursive(list1, list2->next);
            return list2;
        }
    }
};
