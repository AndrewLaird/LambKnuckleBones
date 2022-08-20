# pretty simple game,
# one player starts
# on each players turn they are given a number 1-6
# and are able to place it in one of their three column
# if they place a number that exists in the other persons column
# the ones matching on the enemy side are removed
# play continues until one player fills up their board
import random
from collections import Counter, defaultdict
from enum import Enum
import copy
import numpy as np

import sys

BOARD_WIDTH = 3
BOARD_HEIGHT = 3
TIE = -1


class Board:
    def __init__(self):
        self.board = [
            [[0 for row in range(BOARD_HEIGHT)] for col in range(BOARD_WIDTH)]
            for player_board in range(2)
        ]
        self.winner = -2

    def get_valid_moves(self, player: int):
        valid_cols = []
        for col in range(BOARD_WIDTH):
            if 0 in self.board[player][col]:
                valid_cols.append(col)
        return valid_cols

    def get_score(self, player):
        score = 0
        for col in range(BOARD_WIDTH):
            counter = Counter(self.board[player][col])
            for num, amount in counter.items():
                score += num * amount * amount
        return score

    def insert_col(self, player, col, value):
        # it's got to be put in the first non zero position
        for row in range(BOARD_HEIGHT):
            if self.board[player][col][row] == 0:
                self.board[player][col][row] = value
                break
        else:
            # can't insert into this col
            raise Exception(
                f"Player {player} cant insert into col {col}, value {value}"
            )

        # update the enemy side
        self.delete_values_matching_in_column(other_player(player), col, value)
        # update the winner
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
        player0_score = self.get_score(0)
        player1_score = self.get_score(1)
        difference = player0_score-player1_score
        if(player):
            return - difference
        return difference

    def delete_values_matching_in_column(self, player, col, value):
        # only hard part is sliding the rest down
        slide_amount = 0
        for row in range(BOARD_HEIGHT):
            if self.board[player][col][row] == value:
                slide_amount += 1
                self.board[player][col][row] = 0
            elif slide_amount > 0:
                self.board[player][col][row - slide_amount] = self.board[player][col][
                    row
                ]
                self.board[player][col][row] = 0

    def is_over(self):
        return self.winner != -2

    def repr_player(self, player: int, reverse_rows=False) -> str:
        player_board = self.board[player]
        # 2d array we want all the 0's then all the 1's
        rows = []

        for row in range(BOARD_HEIGHT):
            str_repr = ""
            for col in range(BOARD_WIDTH):
                str_repr += str(player_board[col][row])
            rows.append(str_repr)
        if reverse_rows:
            return "\n".join(rows[::-1])
        return "\n".join(rows)

    def __repr__(self) -> str:
        # print the top player as player 0
        return (
            self.repr_player(1, reverse_rows=True)
            + "\n------------------"
            + str(self.get_score(0))
            + ":"
            + str(self.get_score(1))
            + "\n"
            + self.repr_player(0)
        )


def knucklebones():
    player = 0
    board = Board()
    print("starting")
    while not board.is_over():
        number = random.randrange(1, 7)
        print("Board:")
        print(board)
        print(f"Player {player} your number is {number}")
        col = int(input("Select column:"))
        board.insert_col(player, col, number)
        player = other_player(player)


class Outcomes(int, Enum):
    TIE = -1
    PLAYER0 = 0
    PLAYER1 = 1

class MemoizeAverageValue:
    def __init__(self, f):
        self.f = f
        self.memo = {}
        self.seen = set()

    def __call__(self, board: Board, player: int, depth: int):
        # ignore depth for identifier
        identifier = str(player) + str(board)
        if identifier in self.seen:
            return 0
        # ones we are working on right now, this is an infinite loop
        self.seen.add(identifier)
        if identifier not in self.memo:
            self.memo[identifier] = self.f(board, player, depth)
        # Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[identifier]




class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
        self.seen = set()

    def __call__(self, board: Board, player: int, depth: int):
        # ignore depth for identifier
        identifier = str(player) + str(board)
        if identifier in self.seen:
            return {Outcomes.TIE: 0, Outcomes.PLAYER0: 0, Outcomes.PLAYER1: 0}
        # ones we are working on right now, this is an infinite loop
        self.seen.add(identifier)
        if identifier not in self.memo:
            self.memo[identifier] = self.f(board, player, depth)
        # Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[identifier]


def update_array(to_update, source):
    for key in source.keys():
        to_update[key] += source[key]
    return to_update


MAX_DEPTH = 4

@MemoizeAverageValue
def knuclebones_recursive_average_value(board: Board, player: int, depth: int) -> dict:
    average_value = 0
    if depth > MAX_DEPTH:
        # say the one with the higher score won
        score_difference_from_player_0 = board.get_score_difference(0)
        return score_difference_from_player_0

    if board.is_over():
        score_difference_from_player_0 = board.get_score_difference(0)
        return score_difference_from_player_0
    # make all the moves with all the dice possible
    # at max this is 6 * 3, branching factor of 18, not that bad
    # the dice is known when making the move so it will
    # be our first level
    values_below = []
    for roll in range(1, 7):
        for move in board.get_valid_moves(player):
            new_board = copy.deepcopy(board)
            new_board.insert_col(player, move, roll)
            move_roll_value = knuclebones_recursive_average_value(
                new_board, other_player(player), depth + 1
            )
            values_below.append(move_roll_value)

    # return average of all values below this point
    return sum(values_below)/(len(values_below)+1)



@Memoize
def knuclebones_recursive(board: Board, player: int, depth: int) -> dict:
    outcomes = {Outcomes.TIE: 0, Outcomes.PLAYER0: 0, Outcomes.PLAYER1: 0}
    if depth > MAX_DEPTH:
        # say the one with the higher score won
        current_winning_player = board.get_current_higher_score()
        outcomes[current_winning_player] += 1
        return outcomes
    if board.is_over():
        outcomes[board.winner] += 1
        return outcomes
    # make all the moves with all the dice possible
    # at max this is 6 * 3, branching factor of 18, not that bad
    # the dice is known when making the move so it will
    # be our first level
    for roll in range(1, 7):
        for move in board.get_valid_moves(player):
            new_board = copy.deepcopy(board)
            new_board.insert_col(player, move, roll)
            move_roll_outcomes = knuclebones_recursive(
                new_board, other_player(player), depth + 1
            )
            update_array(outcomes, move_roll_outcomes)

    return outcomes


def other_player(player: int) -> int:
    return (player + 1) % 2

def get_best_move_average_value(board: Board, player: int, number_rolled: int):
    moves = {}
    for move in board.get_valid_moves(player):
        new_board = copy.deepcopy(board)
        new_board.insert_col(player, move, number_rolled)
        moves[move] = knuclebones_recursive_average_value(new_board, other_player(player), 0)

    print(moves)
    # find the move that has the highest average value
    best_move = 0
    best_move_value = moves[0]
    for move, value in moves.items():
        if value > best_move_value:
            best_move = move
            best_move_value = value

    print(f"Pick {best_move} to have an average value of {best_move_value}")
    return best_move


def get_best_move(board: Board, player: int, number_rolled: int):
    results = {}
    for move in board.get_valid_moves(player):
        new_board = copy.deepcopy(board)
        new_board.insert_col(player, move, number_rolled)
        results[move] = knuclebones_recursive(new_board, other_player(player), 0)
    # find the move that lost the least

    best_lost_percentage = 2
    best_move = list(results.keys())[0]
    for move, outcomes in results.items():
        lost_percentage = outcomes[other_player(player)] / (
            outcomes[player] + outcomes[TIE] + 1
        )
        if lost_percentage < best_lost_percentage:
            best_move = move
            best_lost_percentage = lost_percentage

    print(f"Pick {best_move} to win/tie {1-best_lost_percentage}")
    return best_move


def play_against_knucklebones_ai():
    player = 0
    board = Board()
    while not board.is_over():
        number_rolled = random.randrange(1, 7)
        print("Board:")
        print(board)
        # print(f"Player {player} your die is {number_rolled}")
        if player == 0:
            # print("AI is thinking .....")
            col = get_best_move(board, player, number_rolled)
        else:
            # print("AI is thinking .....")
            col = get_best_move(board, player, number_rolled)
            # col = int(input("Select column:"))
        board.insert_col(player, col, number_rolled)
        player = other_player(player)

    player0_score = board.get_score(0)
    player1_score = board.get_score(1)
    print(f"We have a winner: {board.winner}, {player0_score}: {player1_score}")
    return board.winner


def collect_data():
    outcomes = {Outcomes.TIE: 0, Outcomes.PLAYER1: 0, Outcomes.PLAYER1: 0}
    #5% 65% 30%
    for i in range(1000):
        winner = play_against_knucklebones_ai()
        outcomes[winner] += 1
        print(outcomes)


# knucklebones()
# knucklebones_ai()
# play_against_knucklebones_ai()
