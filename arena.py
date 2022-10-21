from collections import namedtuple
from copy import deepcopy
from typing import Any, NamedTuple


def parallel_arena(num_games, player0: Agent, player1: Agent, render=False):
    pass


def run_arena(player0: Agent, player1: Agent, render=False):
    training_data: list[DataPoint] = []
    board = Board()
    current_player = 0
    players = {0: player0, 1: player1}
    board_dict, number_rolled = board.get_observation()
    while not board.is_over():
        old_state = deepcopy([board_dict, number_rolled])
        player_action = players[current_player].get_action(
            player=current_player,
            board=deepcopy(board_dict),
            number_rolled=number_rolled,
        )
        board.insert_col(current_player, player_action, number_rolled)
        board_dict, new_number_rolled = board.get_observation()
        training_data.append(
            DataPoint(
                old_state,
                player_action,
                deepcopy([board_dict, new_number_rolled]),
                current_player,
                0,
            )
        )
        if render:
            print(f"Player: {current_player}, Die: {number_rolled}")
            print(board)
            print("~~~~~~~~~~~~~")

        current_player = KnuckleBonesUtils.other_player(current_player)
        number_rolled = new_number_rolled
    reward = board.get_winner()
    if reward == 2:
        # do we train on ties?
        return reward, []
    training_data = update_training_data(training_data, reward)

    return board.get_winner(), training_data


if __name__ == "__main__":
    random_agent = RandomAgent()
    default_model_agent = ModelAgent()
    value_agent = ValueAgent()
    value_agent.load("value_agent.pt")

    winning = {0: 0, 1: 0, 2: 0}
    for i in range(300):
        if i % 2 == 0:
            winner, training_data = run_arena(value_agent, random_agent, render=i == 0)
        else:
            winner, training_data = run_arena(random_agent, value_agent, render=i == 0)
            # map winner to the other winner
            winner = {0: 1, 1: 0, 2: 2}[winner]
        if(i ==0):
            print(training_data)

        winning[winner] += 1
    print(winning)
