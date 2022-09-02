from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
import numpy as np
from model import DefaultModel, ValueModel
import random
from knucklebones import Board, KnuckleBonesUtils
import torch
from datapoint import DataPoint

# abstract class
class Agent(ABC):
    @abstractmethod
    def get_action(
        self, player: int, board: list[list[list[int]]], number_rolled: int
    ) -> int:
        """
        player: 0 or 1 denoting which side of the board you're on
        board: shape(2,3,3), board[0] is player0's 3x3 board
        number_rolled: 1-6 the dice you get to place this turn
        """
        pass


"""
[
    [ Player 0
        [0,0,0], column 0
        [0,0,0], column 1
        [0,0,0] colum 2
    ],
    [ Player 1
        [0,0,0],
        [0,0,0],
        [0,0,0]
    ]

]

"""


class RandomAgent(Agent):
    def get_action(self, player: int, board: list[list[list[int]]], number_rolled: int):
        board_obj = Board(board)
        valid_moves = board_obj.get_valid_moves(player)
        return valid_moves[random.randint(0, len(valid_moves) - 1)]


class DepthAgent(Agent):
    def __init__(self):
        self.visited = {}
        self.max_depth = 3

    def get_position_value(self, player, board, depth):
        pass

    def get_action(self, player: int, board: list[list[list[int]]], number_rolled: int):
        # Create function to get the average score of position
        # Look at all 18 actions that can come off this action and
        pass


class BellmanAgent(Agent):
    def __init__(self):
        pass

    def get_action(self, player: int, board: list[list[list[int]]], number_rolled: int):

        state = KnuckleBonesUtils.flatten_board(board)
        state.append(number_rolled)
        # state is now [19,1]

        pass


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class ModelAgent(Agent):
    def __init__(self):
        self.my_default_model = DefaultModel()
        pass

    def get_action(self, player: int, board: list[list[list[int]]], number_rolled: int):

        # [0,2]
        valid_moves = KnuckleBonesUtils.get_valid_moves(board, player)
        # [3,1]
        action_probs = self.my_default_model.forward(
            torch.tensor(board), number_rolled
        ).detach()
        # normalize action probs to zero - 1
        action_probs = normalize_data(list(action_probs))
        action_probs = [x + 1 for x in action_probs]
        action_probs = [
            float(prob) * int(i in valid_moves) for i, prob in enumerate(action_probs)
        ]

        action = np.argmax(action_probs)
        return action

    def train(self, training_data: list[DataPoint]):

        # for each datapoint
        # run it through the network to get what it thinks it should be
        # get it's true value form bellman (hear me out, just use value for now)
        # set loss equal to MSE between those
        # [ ]
        pass


class ValueAgent(Agent):
    def __init__(self):
        self.value_model = ValueModel()
        pass

    def save(self, name):
        torch.save(self.value_model.state_dict(), name)

    def load(self, name):
        self.value_model.load_state_dict(torch.load(name))

    def get_possible_states_from_state_action(
        self, player: int, board: list[list[list[int]]], number_rolled: int, action: int
    ):
        """Get all S' for (s,a)"""
        next_board = KnuckleBonesUtils.insert_col(board, player, action, number_rolled)
        for i in range(1, 7):
            yield (deepcopy(next_board), i)

    def get_action(self, player: int, board: list[list[list[int]]], number_rolled: int):

        # [0,2]
        valid_moves = KnuckleBonesUtils.get_valid_moves(board, player)
        # [3,1]
        move_expected_values = defaultdict(float)
        for valid_move in valid_moves:
            all_s_primes = self.get_possible_states_from_state_action(
                player, board, number_rolled, valid_move
            )
            # get expected value for all_s_prime
            s_prime_values = [
                self.value_model.forward(
                    torch.tensor(board), number_rolled
                ).detach()  # captures actual value
                for board, number_rolled in all_s_primes
            ]
            # return average value of them
            move_expected_values[valid_move] = sum(
                [float(x) for x in s_prime_values]
            ) / len(s_prime_values)

        move = max(move_expected_values, key=move_expected_values.get)
        if random.random() < 0.1:
            # explore by randomly selecting a move
            move = valid_moves[random.randint(0, len(valid_moves) - 1)]

        return move

    def train(self, training_data: list[DataPoint]):
        self.value_model.train(training_data)
