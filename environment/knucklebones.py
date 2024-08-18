# pretty simple game,
# one player starts
# on each players turn they are given a number 1-6
# and are able to place it in one of their three column
# if they place a number that exists in the other persons column
# the ones matching on the enemy side are removed
# play continues until one player fills up their board
import random
from collections import Counter
import copy
import numpy as np
from typing import List


BOARD_WIDTH = 3
BOARD_HEIGHT = 3
TIE = -1


class KnuckleBonesUtils:
    """
    Collection of static methods to create knuclebones game
    """

    EMPTY_BOARD = [
        [[0 for row in range(BOARD_HEIGHT)] for col in range(BOARD_WIDTH)]
        for player_board in range(2)
    ]

    @staticmethod
    def get_initial_board():
        return copy.deepcopy(KnuckleBonesUtils.EMPTY_BOARD)

    @staticmethod
    def other_player(player: int) -> int:
        return (player + 1) % 2

    @staticmethod
    def other_player_or_tie(player: int) -> int:
        if player == 2:
            return player
        return (player + 1) % 2

    @staticmethod
    def get_valid_moves(board:List[List[List[int]]], player: int) -> List[int]:
        valid_cols = set()
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                if board[player][row][col] == 0:
                    valid_cols.add(col)
        return list(valid_cols)

    @staticmethod
    def get_score(board:List[List[List[int]]], player) -> int:
        score = 0
        for col in range(BOARD_WIDTH):
            counter = Counter(board[player][col])
            for num, amount in counter.items():
                score += num * amount * amount
        return score

    @staticmethod
    def insert_col(board:List[List[List[int]]], player, col, value) -> List[List[List[int]]]:
        # it's got to be put in the first non zero position
        for row in range(BOARD_HEIGHT):
            if board[player][row][col] == 0:
                board[player][row][col] = value
                break
        else:
            # can't insert into this col
            raise Exception(
                f"Player {player} cant insert into col {col}, value {value}"
            )

        # update the enemy side
        board = KnuckleBonesUtils.delete_values_matching_in_column(
            board, KnuckleBonesUtils.other_player(player), col, value
        )
        return board

    @staticmethod
    def get_score_difference(board:List[List[List[int]]], player) -> int:
        player0_score = KnuckleBonesUtils.get_score(board, 0)
        player1_score = KnuckleBonesUtils.get_score(board, 1)
        difference = player0_score - player1_score
        if player:
            return -difference
        return difference

    @staticmethod
    def delete_values_matching_in_column(board:List[List[List[int]]], player, col, value) -> List[List[List[int]]]:
        # only hard part is sliding the rest down
        slide_amount = 0
        for row in range(BOARD_HEIGHT):
            if board[player][row][col] == value:
                slide_amount += 1
                board[player][row][col] = 0
            elif slide_amount > 0:
                board[player][col][row - slide_amount] = board[player][row][col]
                board[player][row][col] = 0
        return board

    @staticmethod
    def is_over(board:List[List[List[int]]]) -> bool:
        """
        One person has a fully filled board
        """
        if not KnuckleBonesUtils.get_valid_moves(
            board, 0
        ) or not KnuckleBonesUtils.get_valid_moves(board, 1):
            return True
        return False

    @staticmethod
    def get_winner(board:List[List[List[int]]]) -> int:
        if not KnuckleBonesUtils.is_over(board):
            return 3  # shouldn't be called
        score_difference = KnuckleBonesUtils.get_score_difference(board, 0)
        if score_difference == 0:
            return 2
        if score_difference > 0:
            return 0
        return 1

    @staticmethod
    def repr_player(board:List[List[List[int]]], player, reverse_rows=False) -> str:
        player_board = board[player]
        # 2d array we want all the 0's then all the 1's
        rows = []

        for row in range(BOARD_HEIGHT):
            str_repr = ""
            for col in range(BOARD_WIDTH):
                str_repr += str(player_board[row][col])
            rows.append(str_repr)
        if reverse_rows:
            return "\n".join(rows[::-1])
        return "\n".join(rows)

    @staticmethod
    def __repr__(board:List[List[List[int]]]) -> str:
        # print the top player as player 0
        return (
            KnuckleBonesUtils.repr_player(board, 1, reverse_rows=True)
            + "\n------------------"
            + str(KnuckleBonesUtils.get_score(board, 0))
            + ":"
            + str(KnuckleBonesUtils.get_score(board, 1))
            + "\n"
            + KnuckleBonesUtils.repr_player(board, 0)
        )

    @staticmethod
    def get_dice_roll() -> int:
        return random.randrange(1, 7)

    @staticmethod
    def flatten_board(board:List[List[List[int]]]) -> List[int]:
        # turns [3,3,2] to [18,1]
        return list(np.array(board).flatten())

    @staticmethod
    def flip_board(board:List[List[List[int]]]):
        temp = board[0]
        board[0] = board[1]
        board[1] = temp
        return board


class Board:
    """
    Handy class with state that uses the staticmethods from above
    """

    def __init__(self):
        self.board = KnuckleBonesUtils.get_initial_board()

    def upload_board(self, board):
        self.board = board

    def get_valid_moves(self, player: int):
        return KnuckleBonesUtils.get_valid_moves(self.board, player)

    def get_score(self, player):
        return KnuckleBonesUtils.get_score(self.board, player)

    def insert_col(self, player, col, value):
        self.board = KnuckleBonesUtils.insert_col(self.board, player, col, value)
        return self.board

    def get_score_difference(self, player):
        return KnuckleBonesUtils.get_score_difference(self.board, player)

    def delete_values_matching_in_column(self, player, col, value):
        self.board = KnuckleBonesUtils.delete_values_matching_in_column(
            self.board, player, col, value
        )
        return self.board

    @property
    def winner(self):
        return KnuckleBonesUtils.get_winner(self.board)

    def get_winner(self):
        return KnuckleBonesUtils.get_winner(self.board)

    def is_over(self):
        return KnuckleBonesUtils.is_over(self.board)

    def repr_player(self, player: int, reverse_rows=False) -> str:
        return KnuckleBonesUtils.repr_player(self.board, player, reverse_rows)

    def __repr__(self) -> str:
        return KnuckleBonesUtils.__repr__(self.board)

    def get_observation(self):
        # we also return which die was rolled
        # we have to prevent people getting observation
        # multiple times to get a better dice roll
        return [self.board, random.randrange(1, 7)]
