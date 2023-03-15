import gym
from gym import spaces
import random
import numpy as np


BOARD_WIDTH = 3
BOARD_HEIGHT = 3
DIE = 6
TIE = -1


def other_player(player):
    return 1 - player


class KnuckleBonesEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(BOARD_WIDTH)
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=6, shape=(2, BOARD_WIDTH, BOARD_HEIGHT), dtype=int), spaces.Discrete(DIE)))
        self.board = np.zeros((2, BOARD_WIDTH, BOARD_HEIGHT), dtype=int)
        self.current_die = random.randint(0,DIE)
        self.current_player = 0
        self.winner = None
        self.score = [0, 0]

    def reset(self):
        self.board = np.zeros((2, BOARD_WIDTH, BOARD_HEIGHT), dtype=int)
        self.current_player = 0
        self.current_die = random.randint(0,DIE)
        self.winner = None
        self.score = [0, 0]
        return (self.board, self.current_die)

    def step(self, action: int):
        player = self.current_player
        col = action 
        value = self.current_die

        try:
            self.insert_col(player, col, value)
        except Exception:
            return self.board, -10, True, {}

        reward = self.get_score_difference(player)
        done = self.is_over()

        if done:
            if self.winner == player:
                reward += 100
            elif self.winner == other_player(player):
                reward -= 100

        self.current_player = other_player(player)

        self.current_die = random.randint(0,DIE)
        return (self.board, self.current_die), reward, done, {}

    def insert_col(self, player, col, value):
        row = np.where(self.board[player][col] == 0)[0]
        if len(row) == 0:
            raise Exception(f"Player {player} cant insert into col {col}, value {value}, because column is full")

        self.board[player][col][row[0]] = value
        self.delete_values_matching_in_column(other_player(player), col, value)

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


    
    def get_valid_moves(self, player):
        return [col for col in range(BOARD_WIDTH) if self.board[player][col].min() == 0]

    def get_score(self, player):
        if player not in [0, 1]:
            return 0
        score = 0
        for col in range(self.board.shape[1]):
            unique, counts = np.unique(self.board[player][col], return_counts=True)
            for num, amount in zip(unique, counts):
                score += num * amount * amount
        return score

    def get_score_difference(self, player):
        return self.get_score(player) - self.get_score(other_player(player))

    def delete_values_matching_in_column(self, player, col, value):
        row = self.board[player][col]
        row[row == value] = 0
        self.board[player][col] = np.concatenate((row[row != 0], row[row == 0]))

    def is_over(self):
        return self.winner != -2

    def repr_player(self, player: int, reverse_rows=False) -> str:
        player_board = self.board[player]
        rows = player_board.transpose().tolist()
        rows = [[str(x) for x in row] for row in rows]
        if reverse_rows:
            rows = rows[::-1]
        return "\n".join(["".join(row) for row in rows])

    def __repr__(self) -> str:
        return (
            self.repr_player(1, reverse_rows=True)
            + "\n------------------"
            + str(self.get_score(0))
            + ":"
            + str(self.get_score(1))
            + "\n"
            + self.repr_player(0)
        )



