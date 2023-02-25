from environment.knucklebones import KnuckleBonesUtils
from typing import Any, List


class DataPoint:
    state: Any
    action: int
    next_state: Any
    current_player: int
    reward: float

    def __init__(
        self,
        state: Any,
        action: int,
        next_state: Any,
        current_player: int,
        reward: int,
    ):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.current_player = current_player
        self.reward = reward

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


def update_training_data(
    training_data: List[DataPoint], reward: float
) -> List[DataPoint]:
    # training data is guarnteed to be an ordered list of the states made throughout the game
    # replace with real belman
    # currently it is just 1 for winnning moves and -1 for lossing moves, no ties given
    for training_point in training_data[::-1]:
        winner = KnuckleBonesUtils.get_winner(training_point.next_state[0])

        if winner == training_point.current_player:
            training_point.reward = 1000
        else:
            training_point.reward = KnuckleBonesUtils.get_score_difference(
                training_point.next_state[0], training_point.current_player
            )

        # flip the reward and board for player 1
        if training_point.current_player:
            training_point.state[0] = KnuckleBonesUtils.flip_board(
                training_point.state[0]
            )
            training_point.next_state[0] = KnuckleBonesUtils.flip_board(
                training_point.next_state[0]
            )

    return training_data
