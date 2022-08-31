from collections import namedtuple
from copy import deepcopy
from typing import Any, NamedTuple
from knucklebones import Board, KnuckleBonesUtils
from agents import Agent, RandomAgent


class DataPoint:
    state: Any
    action: int
    next_state: Any
    reward: float

    def __init__(
        self,
        state: Any,
        action: int,
        next_state: Any,
        reward: int,
    ):
        self.state=state
        self.action=action
        self.next_state=next_state
        self.reward=reward

    def __repr__(self):
        board_print = KnuckleBonesUtils.__repr__(self.state[0])
        next_board_print = KnuckleBonesUtils.__repr__(self.next_state[0])
        return f"""
        die:{self.state[1]}
        {board_print}
        action:{self.action}\n
        next_die:{self.next_state[1]}
        {next_board_print}\n
        reward:{self.reward}\n
        """

def update_training_data(training_data: list[DataPoint], reward: float):
    for training_point in training_data[::-1]:
        training_point.reward = reward
        reward = -reward
    return training_data
    

def run_arena(player0: Agent, player1: Agent, render=False):
    training_data: list[DataPoint] = []
    board = Board()
    current_player = 0
    players = {0: player0, 1: player1}
    board_dict, number_rolled = board.get_observation()
    while not board.is_over():
        old_state = deepcopy((board_dict, number_rolled))
        player_action = players[current_player].get_action(
            player=current_player, board=board_dict, number_rolled=number_rolled
        )
        board.insert_col(current_player, player_action, number_rolled)
        board_dict, number_rolled = board.get_observation()
        training_data.append(
            DataPoint(old_state, player_action, deepcopy((board_dict, number_rolled)), 0)
        )
        if render:
            print(board)

        current_player = KnuckleBonesUtils.other_player(current_player)
    reward = board.get_winner()
    if(reward == 0):
        # do we train on ties?
        return reward, []
    training_data = update_training_data(training_data, reward)
    return board.get_winner(), training_data


if __name__ == "__main__":
    random_agent = RandomAgent()

    winner, training_data = run_arena(random_agent, random_agent, render=True)
    print(f"winner {winner}")
    print("training data", training_data)
