from agents.agents import Agent, KnuckleBonesUtils
import uuid
from typing import List
import numpy as np
import random
from copy import deepcopy


class MCTSAgent(Agent):
    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations
        self.root_node_key = None
        self.nodes = {}
        self.previous_board = None

    def get_opponent_move(self, opponent, old_board, new_board):
        for col in range(len(old_board[opponent])):
            for row in range(len(old_board[opponent][col])):
                if old_board[opponent][col][row] != new_board[opponent][col][row]:
                    return col, new_board[opponent][col][row]

    def apply_agent_move(self, best_action, number_rolled):
        for node_key, node in self.nodes.items():
            if (
                node["action"] == best_action
                and node["number_rolled"] == number_rolled
                and node["parent_key"] == self.root_node_key
            ):
                self.root_node_key = node_key
                break

    def update_root_node(self, player, new_board, number_rolled):
        opponent = (player + 1) % 2
        if self.root_node_key is None:
            node_key = self.create_node_key()
            self.nodes[node_key] = self.create_node(
                player, KnuckleBonesUtils.get_initial_board(), number_rolled
            )
            self.root_node_key = node_key
        else:
            action, number_rolled = self.get_opponent_move(
                opponent, self.previous_board, new_board
            )
            new_child_found = False
            for node_key, node in self.nodes.items():
                if (
                    node["action"] == action
                    and node["number_rolled"] == number_rolled
                    and node["parent_key"] == self.root_node_key
                ):
                    self.root_node_key = node_key
                    new_child_found = True
                    break

            if not new_child_found:
                child_key = self.create_node_key()
                child = self.create_node(
                    player,
                    new_board,
                    number_rolled,
                    parent_key=self.root_node_key,
                    action=action,
                )
                self.nodes[child_key] = child
                self.root_node_key = child_key

    def get_action(self, player: int, board: List[List[List[int]]], number_rolled: int):
        if self.root_node_key is None or self.nodes[self.root_node_key]['board'] != board:
            self.update_root_node(player, board, number_rolled)

        for _ in range(self.num_simulations):
            selected_node_key = self.select(self.root_node_key)
            reward = self.rollout(selected_node_key)
            self.backpropagate(selected_node_key, reward)

        best_action = self.get_best_action(self.root_node_key)

        self.apply_agent_move(best_action, number_rolled)
        previous_board = deepcopy(board)
        previous_board = KnuckleBonesUtils.insert_col(previous_board, player, best_action, number_rolled)

        self.previous_board = previous_board
        return best_action

    def create_node_key(self):
        return str(uuid.uuid4())

    def create_node(self, player: int, board: List[List[List[int]]], number_rolled: int, parent_key=None, action=None):
        return {
            'player': player,
            'board': board,
            'number_rolled': number_rolled,
            'parent_key': parent_key,
            'action': action,
            'children_keys': [],
            'visit_count': 0,
            'total_reward': 0
        }

    def is_fully_expanded(self, node_key):
        node = self.nodes[node_key]
        num_valid_moves = len(KnuckleBonesUtils.get_valid_moves(node['board'], node['player']))
        return len(node['children_keys']) == num_valid_moves * 6

    def expand(self, node_key):
        node = self.nodes[node_key]
        valid_moves = KnuckleBonesUtils.get_valid_moves(node['board'], node['player'])
        for move in valid_moves:
            for number_rolled in range(1, 7):
                child_board = deepcopy(node['board'])
                child_board = KnuckleBonesUtils.insert_col(child_board, node['player'], move, number_rolled)
                child_player = (node['player'] + 1) % 2
                child_key = self.create_node_key()
                child = self.create_node(child_player, child_board, number_rolled, parent_key=node_key, action=move)
                self.nodes[child_key] = child
                node['children_keys'].append(child_key)


    def select(self, node_key):
        current_key = node_key

        while True:
            current_node = self.nodes[current_key]
            if not current_node['children_keys']:
                if not self.is_fully_expanded(current_key):
                    self.expand(current_key)
                return current_key

            max_ucb = float('-inf')
            selected_node_key = None
            for child_key in current_node['children_keys']:
                child = self.nodes[child_key]
                ucb = self.get_ucb(child_key)
                if ucb > max_ucb:
                    max_ucb = ucb
                    selected_node_key = child_key

            if not self.is_fully_expanded(selected_node_key):
                return selected_node_key
            else:
                current_key = selected_node_key

    def get_ucb(self, node_key):
        node = self.nodes[node_key]
        if node['visit_count'] == 0:
            return float('inf')
        else:
            return node['total_reward'] / node['visit_count'] + np.sqrt(2 * np.log(self.nodes[node['parent_key']]['visit_count']) / node['visit_count'])

    def rollout(self, node_key):
        node = self.nodes[node_key]
        board_copy = deepcopy(node['board'])
        player_copy = node['player']
        while not KnuckleBonesUtils.is_over(board_copy):
            valid_moves = KnuckleBonesUtils.get_valid_moves(board_copy, player_copy)
            action = valid_moves[random.randint(0, len(valid_moves) - 1)]
            number_rolled = random.randint(1, 6)
            board_copy = KnuckleBonesUtils.insert_col(board_copy, player_copy, action, number_rolled)
            player_copy = (player_copy + 1) % 2

        return KnuckleBonesUtils.get_score(board_copy, node['player'])


    def backpropagate(self, node_key, reward):
        current_key = node_key
        while current_key is not None:
            node = self.nodes[current_key]
            node['visit_count'] += 1
            node['total_reward'] += reward
            current_key = node['parent_key']

    def get_best_action(self, node_key):
        node = self.nodes[node_key]
        best_child_key = max(node['children_keys'], key=lambda child_key: self.nodes[child_key]['visit_count'])
        return self.nodes[best_child_key]['action']
