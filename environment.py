# pretty simple game,
# one player starts
# on each players turn they are given a number 1-6
# and are able to place it in one of their three column
# if they place a number that exists in the other persons column
# the ones matching on the enemy side are removed
# play continues until one player fills up their board
import random
import numpy as np


BOARD_WIDTH = 3
BOARD_HEIGHT = 3
TIE = -1


class Board:
    def __init__(self):
        # each column is
        self.board = np.zeros((2, BOARD_WIDTH, BOARD_HEIGHT), dtype=int)
        self.winner = -2

    def get_valid_moves(self, player: int):
        return np.where(self.board[player].min(axis=1) == 0)[0].tolist()

    def get_score(self, player):
        if player not in [0, 1]:
            return 0
        score = 0
        for col in range(self.board.shape[1]):
            unique, counts = np.unique(self.board[player][col], return_counts=True)
            for num, amount in zip(unique, counts):
                score += num * amount * amount
        return score

    def insert_col(self, player, col, value):
        row = np.where(self.board[player][col] == 0)[0]
        if len(row) == 0:
            raise Exception(
                f"Player {player} cant insert into col {col}, value {value}, because column is full"
            )

        self.board[player][col][row[0]] = value
        self.delete_values_matching_in_column(other_player(player), col, value)

        if len(self.get_valid_moves(player)) == 0:
            self.winner = self.get_current_higher_score()

    def get_current_higher_score(self):
        player0_score = self.get_score(0)
        player1_score = self.get_score(1)
        if player0_score == player1_score:
            return TIE
        elif player0_score > player1_score:
            return 0
        return 1

    def get_score_difference(self, player):
        return self.get_score(player) - self.get_score(other_player(player))

    def delete_values_matching_in_column(self, player, col, value):
        row = self.board[player][col]
        row[row == value] = 0
        self.board[player][col] = np.concatenate((row[row != 0], row[row == 0]))

    def is_over(self):
        return self.winner != -2

    def repr_player(self, player: int, reverse_rows=False) -> str:
        player_board = self.board[player]
        rows = player_board.transpose().tolist()
        rows = [[str(x) for x in row] for row in rows]
        if reverse_rows:
            rows = rows[::-1]
        return "\n".join(["".join(row) for row in rows])

    def __repr__(self) -> str:
        return (
            self.repr_player(1, reverse_rows=True)
            + "\n------------------"
            + str(self.get_score(0))
            + ":"
            + str(self.get_score(1))
            + "\n"
            + self.repr_player(0)
        )


def other_player(player: int):
    return (player + 1) % 2


def knucklebones():
    player = 0
    board = Board()
    print("starting")
    while not board.is_over():
        number = random.randrange(1, 7)
        print("Board:")
        print(board)
        print(f"Player {player} your number is {number}")
        col = random.choice(
            board.get_valid_moves(player)
        )  # int(input("Select column:"))
        board.insert_col(player, col, number)
        player = other_player(player)
    print(board)
    print("winner", board.winner, "score =", board.get_score(board.winner))

    print()


if __name__ == "__main__":
    knucklebones()
