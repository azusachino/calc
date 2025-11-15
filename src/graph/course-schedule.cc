#include <vector>
#include <queue>

using namespace std;

// LeetCode #207 - Course Schedule
// Check if you can finish all courses (detect cycle in directed graph)
// Time: O(V + E), Space: O(V + E)

class Solution {
public:
    // Topological Sort using Kahn's Algorithm (BFS)
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> graph(numCourses);
        vector<int> indegree(numCourses, 0);

        // Build graph
        for (const auto& prereq : prerequisites) {
            graph[prereq[1]].push_back(prereq[0]);
            indegree[prereq[0]]++;
        }

        // Find all courses with no prerequisites
        queue<int> q;
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {
                q.push(i);
            }
        }

        int completed = 0;

        while (!q.empty()) {
            int course = q.front();
            q.pop();
            completed++;

            for (int next : graph[course]) {
                indegree[next]--;
                if (indegree[next] == 0) {
                    q.push(next);
                }
            }
        }

        return completed == numCourses;
    }
};

// Alternative: DFS cycle detection
class Solution2 {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> graph(numCourses);
        for (const auto& prereq : prerequisites) {
            graph[prereq[1]].push_back(prereq[0]);
        }

        vector<int> state(numCourses, 0); // 0: unvisited, 1: visiting, 2: visited

        for (int i = 0; i < numCourses; i++) {
            if (hasCycle(graph, i, state)) {
                return false;
            }
        }

        return true;
    }

private:
    bool hasCycle(vector<vector<int>>& graph, int course, vector<int>& state) {
        if (state[course] == 1) return true;  // Cycle detected
        if (state[course] == 2) return false; // Already processed

        state[course] = 1; // Mark as visiting

        for (int next : graph[course]) {
            if (hasCycle(graph, next, state)) {
                return true;
            }
        }

        state[course] = 2; // Mark as visited
        return false;
    }
};

// LeetCode #210 - Course Schedule II
// Return the ordering of courses to take
// Time: O(V + E), Space: O(V + E)

class Solution3 {
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> graph(numCourses);
        vector<int> indegree(numCourses, 0);

        for (const auto& prereq : prerequisites) {
            graph[prereq[1]].push_back(prereq[0]);
            indegree[prereq[0]]++;
        }

        queue<int> q;
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {
                q.push(i);
            }
        }

        vector<int> order;

        while (!q.empty()) {
            int course = q.front();
            q.pop();
            order.push_back(course);

            for (int next : graph[course]) {
                indegree[next]--;
                if (indegree[next] == 0) {
                    q.push(next);
                }
            }
        }

        return order.size() == numCourses ? order : vector<int>();
    }
};
