from abc import ABC, abstractmethod
import math
from collections import defaultdict
from copy import deepcopy
import numpy as np
import random
import torch
import os.path
from typing import List
from environment.knucklebones import Board, KnuckleBonesUtils

from environment.datapoint import DataPoint



# abstract class
class Agent(ABC):
    @abstractmethod
    def get_action(
        self, player: int, board: List[List[List[int]]], number_rolled: int
    ) -> int:
        """
        player: 0 or 1 denoting which side of the board you're on
        board: shape(2,3,3), board[0] is player0's 3x3 board
        number_rolled: 1-6 the dice you get to place this turn
        """


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
    def get_action(self, player: int, board: List[List[List[int]]], number_rolled: int):
        board_obj = Board()
        board_obj.upload_board(board)
        valid_moves = board_obj.get_valid_moves(player)
        return valid_moves[random.randint(0, len(valid_moves) - 1)]


class DepthAgent(Agent):
    def __init__(self, max_depth=5):
        self.visited = {}
        self.max_depth = max_depth

    def get_position_value(self, player, board):
        return KnuckleBonesUtils.get_score(board, player)

    def get_action(self, player: int, board: List[List[List[int]]], number_rolled: int):
        best_action = None
        best_score = float('-inf')

        for action in KnuckleBonesUtils.get_valid_moves(board, player):
            next_board = KnuckleBonesUtils.insert_col(deepcopy(board), player, action, number_rolled)
            average_score = 0
            for next_roll in range(1, 7):
                score = self.minimax(player, next_board, self.max_depth - 1, float('-inf'), float('inf'), False, next_roll)
                average_score += score / 6

            if average_score > best_score:
                best_score = average_score
                best_action = action

        return best_action

    def minimax(self, player, board, depth, alpha, beta, is_maximizing, number_rolled):
        if depth == 0 or KnuckleBonesUtils.is_over(board):
            return self.get_position_value(player, board)

        if is_maximizing:
            best_score = float('-inf')
            for action in KnuckleBonesUtils.get_valid_moves(board, player):
                next_board = KnuckleBonesUtils.insert_col(deepcopy(board), player, action, number_rolled)
                average_score = 0
                for next_roll in range(1, 7):
                    score = self.minimax(player, next_board, depth - 1, alpha, beta, False, next_roll)
                    average_score += score / 6
                best_score = max(best_score, average_score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = float('inf')
            opponent = (player + 1) % 2
            for action in KnuckleBonesUtils.get_valid_moves(board, opponent):
                next_board = KnuckleBonesUtils.insert_col(deepcopy(board), opponent, action, number_rolled)
                average_score = 0
                for next_roll in range(1, 7):
                    score = self.minimax(player, next_board, depth - 1, alpha, beta, True, next_roll)
                    average_score += score / 6
                best_score = min(best_score, average_score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score

class MCTSNodeAgent(Agent):
    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations
        # root_node will be the node we are simulating from in the game
        # example first move it will be MCTSNode(player, KnuckleBonesUtils.get_initial_board(), number_rolled)
        # with player being the int representing our player and number_rolled being the int representing the number have to place 
        # in one of three columns
        self.root_node = None
        self.previous_board = None

    def get_opponent_move(self, opponent, old_board, new_board):
        for col in range(len(old_board[opponent])):
            for row in range(len(old_board[opponent][col])):
                if old_board[opponent][col][row] != new_board[opponent][col][row]:
                    return col, new_board[opponent][col][row]

    def apply_agent_move(self, best_action, number_rolled):
        for child in self.root_node.children:
            if child.action == best_action and child.number_rolled == number_rolled:
                child.parent = None
                self.root_node = child
                break

    def update_root_node(self, player, new_board, number_rolled):
        opponent = (player + 1) % 2
        if self.root_node is None:
            self.root_node = MCTSNode(player, KnuckleBonesUtils.get_initial_board(), number_rolled)
        else:
            action, number_rolled = self.get_opponent_move(opponent, self.previous_board, new_board)
            new_child_found = False
            for child in self.root_node.children:
                if child.action == action and child.number_rolled == number_rolled:
                    child.parent = None
                    self.root_node = child
                    new_child_found = True
                    break

            if not new_child_found:
                child = MCTSNode(player, new_board, number_rolled, parent=self.root_node, action=action)
                self.root_node.children.append(child)
                self.root_node = child

    def get_action(self, player: int, board: List[List[List[int]]], number_rolled: int):
        if self.root_node is None or self.root_node.board != board:
            self.update_root_node(player, board, number_rolled)
        self.previous_board = board

        for _ in range(self.num_simulations):
            selected_node = self.root_node.select()
            reward = selected_node.rollout()
            selected_node.backpropagate(reward)

        best_action = self.root_node.get_best_action()
        self.apply_agent_move(best_action, number_rolled)
        return best_action

class MCTSNode:
    def __init__(self, player: int, board: List[List[List[int]]], number_rolled: int, parent=None, action=None):
        self.player = player
        self.board = board
        self.number_rolled = number_rolled
        self.parent = parent
        self.action = action
        self.children = []
        self.visit_count = 0
        self.total_reward = 0

    def is_fully_expanded(self):
        return len(self.children) == len(KnuckleBonesUtils.get_valid_moves(self.board, self.player)) * 6

    def expand(self):
        valid_moves = KnuckleBonesUtils.get_valid_moves(self.board, self.player)
        for move in valid_moves:
            for number_rolled in range(1, 7):
                child_board = deepcopy(self.board)
                child_board = KnuckleBonesUtils.insert_col(child_board, self.player, move, self.number_rolled)
                child_player = (self.player + 1) % 2
                child = MCTSNode(child_player, child_board, number_rolled, parent=self, action=move)
                self.children.append(child)

    def select(self):
        if not self.children:
            if not self.is_fully_expanded():
                self.expand()
            return self

        max_ucb = float('-inf')
        selected_node = None
        for child in self.children:
            ucb = child.get_ucb()
            if ucb > max_ucb:
                max_ucb = ucb
                selected_node = child

        if not selected_node.is_fully_expanded():
            return selected_node
        else:
            return selected_node.select()

    def get_ucb(self):
        if self.visit_count == 0:
            return float('inf')
        else:
            return self.total_reward / self.visit_count + np.sqrt(2 * np.log(self.parent.visit_count) / self.visit_count)

    def rollout(self):
        board_copy = deepcopy(self.board)
        player_copy = self.player
        while not KnuckleBonesUtils.is_over(board_copy):
            valid_moves = KnuckleBonesUtils.get_valid_moves(board_copy, player_copy)
            action = valid_moves[random.randint(0, len(valid_moves) - 1)]
            number_rolled = random.randint(1, 6)
            board_copy = KnuckleBonesUtils.insert_col(board_copy, player_copy, action, number_rolled)
            player_copy = (player_copy + 1) % 2

        return KnuckleBonesUtils.get_score(board_copy, self.player)

    def backpropagate(self, reward):
        self.visit_count += 1
        self.total_reward += reward
        if self.parent is not None:
            self.parent.backpropagate(reward)

    def get_best_action(self):
        best_child = max(self.children, key=lambda c: c.visit_count)
        return best_child.action

    def __str__(self):
        return f"Node(player={self.player}, action={self.action}, number_rolled={self.number_rolled}, visit_count={self.visit_count}, total_reward={self.total_reward})"


class BellmanAgent(Agent):
    def __init__(self):
        pass

    def get_action(self, player: int, board: List[List[List[int]]], number_rolled: int):

        state = KnuckleBonesUtils.flatten_board(board)
        state.append(number_rolled)
        # state is now [19,1]



def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class ModelAgent(Agent):
    def __init__(self):
        self.my_default_model = DefaultModel()

    def get_action(self, player: int, board: List[List[List[int]]], number_rolled: int):

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

    def train(self, training_data: List[DataPoint]):

        # for each datapoint
        # run it through the network to get what it thinks it should be
        # get it's true value form bellman (hear me out, just use value for now)
        # set loss equal to MSE between those
        # [ ]
        pass


class ValueAgent(Agent):
    def __init__(self):
        self.value_model = ValueModel()
        self.action_requests = {}
        self.all_actions = {}

    def save(self, name):
        torch.save(self.value_model.state_dict(), name)

    def load(self, name):
        if os.path.exists(name):
            self.value_model.load_state_dict(torch.load(name))

    def get_possible_states_from_state_action(
        self, player: int, board: List[List[List[int]]], number_rolled: int, action: int
    ):
        """Get all S' for (s,a)"""
        next_board = KnuckleBonesUtils.insert_col(board, player, action, number_rolled)
        for i in range(1, 7):
            yield (deepcopy(next_board), i)

    def get_action(self, player: int, board: List[List[List[int]]], number_rolled: int):
        valid_moves = KnuckleBonesUtils.get_valid_moves(board, player)
        move_expected_values = defaultdict(float)
        for valid_move in valid_moves:
            all_s_primes = self.get_possible_states_from_state_action(
                player, board, number_rolled, valid_move
            )
            # get expected value for all_s_prime
            s_prime_values = self.value_model.forward(
                torch.stack(
                    [
                        state_to_tensor(board, number_rolled)
                        for board, number_rolled in all_s_primes
                    ]
                )
            ).detach()  # captures actual value
            # return average value of them
            move_expected_values[valid_move] = sum(
                [float(x) for x in s_prime_values]
            ) / len(s_prime_values)

        move = max(move_expected_values, key=move_expected_values.get)
        if random.random() < 0.1:
            # explore by randomly selecting a move
            move = valid_moves[random.randint(0, len(valid_moves) - 1)]

        return move

    def request_action(
        self,
        game_num: int,
        player: int,
        board: List[List[List[int]]],
        number_rolled: int,
    ):
        self.action_requests[game_num] = (player, board, number_rolled)

    def exectute_all_requests(self):
        pass

    def read_action(self, game_num):
        return self.all_actions[game_num]

    def train_with_data(self, training_data: List[DataPoint]):
        self.value_model.train_with_data(training_data)
        

