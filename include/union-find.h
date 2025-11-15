#ifndef UNION_FIND_H
#define UNION_FIND_H

#include <vector>
#include <numeric>

using namespace std;

// Union Find (Disjoint Set Union) Data Structure
// With path compression and union by rank optimization
// Time: O(α(n)) ≈ O(1) per operation, where α is inverse Ackermann

class UnionFind {
private:
    vector<int> parent;
    vector<int> rank;
    int count; // Number of connected components

public:
    UnionFind(int n) : parent(n), rank(n, 0), count(n) {
        iota(parent.begin(), parent.end(), 0); // parent[i] = i
    }

    // Find with path compression
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]); // Path compression
        }
        return parent[x];
    }

    // Union by rank
    bool unite(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);

        if (rootX == rootY) {
            return false; // Already in same set
        }

        // Union by rank
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }

        count--;
        return true;
    }

    // Check if x and y are connected
    bool connected(int x, int y) {
        return find(x) == find(y);
    }

    // Get number of connected components
    int getCount() {
        return count;
    }
};

#endif
