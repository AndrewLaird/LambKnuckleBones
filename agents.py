from abc import ABC, abstractmethod
import random
from knucklebones import Board

# abstract class
class Agent(ABC):

    @abstractmethod
    def get_action(self, player: int, board: list[list[list[int]]], number_rolled: int):
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
        return valid_moves[random.randint(0,len(valid_moves)-1)]

class DepthAgent(Agent):

    def __init__(self):
        self.visited = {}
        self.max_depth=3

    def get_position_value(self, player, board, depth):
        pass

    def get_action(self, player: int, board: list[list[list[int]]], number_rolled: int):
        # Create function to get the average score of position
        # Look at all 18 actions that can come off this action and 
        pass

class MctsAgent(Agent):
    """
    MonteCarlo Tree Search
    """
    def 
