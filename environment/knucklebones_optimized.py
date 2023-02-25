import numpy as np
from typing import List
import copy
from numba import jit


BOARD_WIDTH = 7
BOARD_HEIGHT = 6


class KnuckleBonesUtils:
    """
    Collection of static methods to create knuclebones game
    """

    EMPTY_BOARD = np.zeros((2, BOARD_WIDTH, BOARD_HEIGHT), dtype=np.int32)

    @staticmethod
    def other_player(player: int) -> int:
        return (player + 1) % 2

    @staticmethod
    def other_player_or_tie(player: int) -> int:
        if player == 2:
            return player
        return (player + 1) % 2

    @staticmethod
    def get_valid_moves(board, player: int) -> List[int]:
        valid_cols = []
        for col in range(BOARD_WIDTH):
            if 0 in board[player, col]:
                valid_cols.append(col)
        return valid_cols

    @staticmethod
    @jit(nopython=True)
    def get_score(board, player) -> int:
        score = 0
        for col in range(BOARD_WIDTH):
            col_values = board[player, col]
            num_zeros = np.count_nonzero(col_values == 0)
            col_values = col_values[: BOARD_HEIGHT - num_zeros]
            counts = np.bincount(col_values)
            for num, amount in enumerate(counts):
                score += num * amount * amount
        return score

    @staticmethod
    def insert_col(board, player, col, value) -> np.ndarray:
        # it's got to be put in the first non zero position
        row = np.argmin(board[player, col] != 0)
        if board[player, col, row] != 0:
            # can't insert into this col
            raise Exception(
                f"Player {player} cant insert into col {col}, value {value}"
            )
        board[player, col, row] = value

        # update the enemy side
        board = KnuckleBonesUtils.delete_values_matching_in_column(
            board, KnuckleBonesUtils.other_player(player), col, value
        )
        return board

    @staticmethod
    @jit(nopython=True)
    def get_score_difference(board, player) -> int:
        player0_score = KnuckleBonesUtils.get_score(board, 0)
        player1_score = KnuckleBonesUtils.get_score(board, 1)
        difference = player0_score - player1_score
        if player:
            return -difference
