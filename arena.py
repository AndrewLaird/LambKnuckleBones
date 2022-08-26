from knucklebones import Board, KnuckleBonesUtils
from agents import Agent, RandomAgent

def run_arena(player0: Agent, player1: Agent, render=False):

    board = Board()
    current_player = 0
    players = {0: player0, 1: player1}
    while(not board.is_over()):
        board_dict, number_rolled = board.get_observation()
        player_action = players[current_player].get_action(player=current_player, board=board_dict, number_rolled=number_rolled)
        board.insert_col(current_player, player_action, number_rolled)
        if(render):
            print(board)

        current_player = KnuckleBonesUtils.other_player(current_player)
    return board.get_winner()

if __name__ == "__main__":
    random_agent = RandomAgent()

    winner = run_arena(random_agent, random_agent, render=True)
    print(f"winner {winner}")

    
