from src.queens import solve_n_queens


def test_n4():
    solutions = solve_n_queens(4)
    assert len(solutions) == 2


def test_n8():
    solutions = solve_n_queens(8)
    assert len(solutions) == 92

