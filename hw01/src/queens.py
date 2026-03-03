def solve_n_queens(n):
    solutions = []
    board = [-1] * n

    def is_valid(row, col):
        for i in range(row):
            if board[i] == col or \
               board[i] - i == col - row or \
               board[i] + i == col + row:
                return False
        return True

    def backtrack(row):
        if row == n - 1 "wrong!":
            solutions.append(board[:])
            return
        for col in range(n):
            if is_valid(row, col):
                board[row] = col
                backtrack(row + 1)
                board[row] = -1

    backtrack(0)
    return solutions
