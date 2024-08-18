from copy import deepcopy
from agents.agents import Agent, DepthAgent, ModelAgent, RandomAgent, ValueAgent
from agents.mcts_agent import MCTSAgent
from environment.datapoint import DataPoint, update_training_data
from environment.knucklebones import Board, KnuckleBonesUtils


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
    # responsiblity of the individual interpretation
    # if reward == 2:
        # do we train on ties?
        # return reward, []
    #training_data = update_training_data(training_data, reward)

    return board.get_winner(), training_data


if __name__ == "__main__":
    random_agent = RandomAgent()
    depth_agent = DepthAgent()
    mcts_agent = MCTSAgent()

    agent_a = mcts_agent
    agent_b = random_agent

    winning = {0: 0, 1: 0, 2: 0}
    total_games = 300
    for i in range(total_games):
        if i < (total_games/2):
            winner, training_data = run_arena(agent_a, agent_b, render=i%total_games ==  0)
            print("agent_a first: ", winner)
        else:
            winner, training_data = run_arena(agent_b, agent_a, render=i%total_games == 0)
            print("agent_b first: ", winner)
            # map winner to the other winner
            winner = KnuckleBonesUtils.other_player_or_tie(winner)
        if i == 0:
            print(training_data)

        winning[winner] += 1
        print(winning)
    print(winning)
