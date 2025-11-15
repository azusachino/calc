#include "../include/list.h"

// LeetCode #206 - Reverse Linked List
// Reverse a singly linked list
// Time: O(n), Space: O(1) for iterative, O(n) for recursive

class Solution {
public:
    // Iterative approach
    ListNode* reverseList(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* curr = head;

        while (curr != nullptr) {
            ListNode* next = curr->next;
            curr->next = prev;
            prev = curr;
            curr = next;
        }

        return prev;
    }

    // Recursive approach
    ListNode* reverseListRecursive(ListNode* head) {
        if (head == nullptr || head->next == nullptr) {
            return head;
        }

        ListNode* newHead = reverseListRecursive(head->next);
        head->next->next = head;
        head->next = nullptr;

        return newHead;
    }
};
