#include <vector>
#include <string>

using namespace std;

// LeetCode #51 - N-Queens
// Find all solutions to n-queens problem
// Time: O(n!), Space: O(n)

class Solution {
public:
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> result;
        vector<string> board(n, string(n, '.'));
        backtrack(board, 0, result);
        return result;
    }

private:
    void backtrack(vector<string>& board, int row,
                   vector<vector<string>>& result) {
        if (row == board.size()) {
            result.push_back(board);
            return;
        }

        int n = board.size();

        for (int col = 0; col < n; col++) {
            if (!isValid(board, row, col)) {
                continue;
            }

            board[row][col] = 'Q';
            backtrack(board, row + 1, result);
            board[row][col] = '.';
        }
    }

    bool isValid(vector<string>& board, int row, int col) {
        int n = board.size();

        // Check column
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q') {
                return false;
            }
        }

        // Check upper left diagonal
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }

        // Check upper right diagonal
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }

        return true;
    }
};

// LeetCode #52 - N-Queens II (count only)
// Time: O(n!), Space: O(n)
// Optimized with bit manipulation

class Solution2 {
public:
    int totalNQueens(int n) {
        return backtrack(n, 0, 0, 0, 0);
    }

private:
    int backtrack(int n, int row, int cols, int diag1, int diag2) {
        if (row == n) {
            return 1;
        }

        int count = 0;
        int availablePositions = ((1 << n) - 1) & ~(cols | diag1 | diag2);

        while (availablePositions) {
            int position = availablePositions & -availablePositions;
            availablePositions &= availablePositions - 1;

            count += backtrack(n, row + 1,
                             cols | position,
                             (diag1 | position) << 1,
                             (diag2 | position) >> 1);
        }

        return count;
    }
};
