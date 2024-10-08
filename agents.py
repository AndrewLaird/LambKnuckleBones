from copy import deepcopy
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
device = "cpu"  # gpu_0
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    device = torch.device("cuda")


def state_to_tensor(board: Any, number_rolled: int):
    x = torch.Tensor(board)
    x = torch.reshape(x, (-1,))
    # reshape to [18,1]
    # add dice
    x = torch.cat([x, torch.tensor([number_rolled], dtype=torch.float)])
    return x


class DefaultModel(nn.Module):
    def __init__(self):
        super(DefaultModel, self).__init__()
        # Input size is [2,3,3]
        self.fc1 = nn.Linear(19, 128)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor, die: int):
        # Input size is [2,3,3] + 1 die
        # reshape to [9,2]
        x = torch.reshape(x, (-1,))
        # reshape to [18,1]
        # add dice
        x = torch.cat([x, torch.tensor([die], dtype=torch.float)])
        # 19,1
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


class ValueModel(nn.Module):
    def __init__(self):
        super(ValueModel, self).__init__()
        # Input size is [2,3,3]
        self.fc1 = nn.Linear(19, 128)
        self.norm1 = nn.BatchNorm1d(128, affine=False)
        self.fc2 = nn.Linear(128, 1024)
        self.norm2 = nn.BatchNorm1d(1024, affine=False)
        self.fc3 = nn.Linear(1024, 128)
        self.norm3 = nn.BatchNorm1d(128, affine=False)
        self.fc4 = nn.Linear(128, 1)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.gamma = 0.9
        self.to(device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor):
        x = x.to(device)
        x = self.norm1(self.fc1(x))
        x = self.norm2(self.fc2(x))
        x = self.norm3(self.fc3(x))
        x = self.fc4(x)
        return x

    def average_all_possible_next_rolls(self, board: Any):
        possible_next_boards = [state_to_tensor(board, i) for i in range(1, 7)]
        return sum(possible_next_boards) / len(possible_next_boards)

    def get_all_next_move_q_values(self, datapoint: DataPoint):
        # take all the moves
        result = []
        board, number_rolled = datapoint.state
        for move in KnuckleBonesUtils.get_valid_moves(board, datapoint.current_player):
            if not KnuckleBonesUtils.is_over(board):
                next_board = KnuckleBonesUtils.insert_col(
                    deepcopy(board), datapoint.current_player, move, number_rolled
                )
                result.append(self.average_all_possible_next_rolls(next_board))
        return self.forward(torch.stack(result)).detach()

    def calculate_targets(self, training_data: list[DataPoint]):
        self.train(False)
        target_values = torch.Tensor().to(device)
        for datapoint in training_data:
            # board, number_rolled = datapoint.state

            # run it through the network to get what it thinks it should be
            # model_value = self.forward(state_to_tensor(board, number_rolled))
            # get it's true value form bellman (hear me out, just use value for now)
            q_target = torch.tensor([datapoint.reward], dtype=torch.float).to(device)

            q_s_prime_a_prime = self.get_all_next_move_q_values(datapoint)

            q_target = q_target + self.gamma * torch.max(q_s_prime_a_prime)
            target_values = torch.cat((target_values, q_target))
        self.train(True)
        target_values = target_values.unsqueeze(-1)
        return target_values

    def train_with_data(self, training_data: list[DataPoint]):
        # start loss at zero
        target_values = self.calculate_targets(training_data)

        for i in range(100):
            self.optimizer.zero_grad()
            total_loss = 0
            all_known_states = torch.stack(
                [state_to_tensor(*datapoint.state) for datapoint in training_data]
            )

            model_values = self.forward(all_known_states)
            # set loss equal to MSE between those
            # total_loss += self.loss(model_value, true_value)
            total_loss = self.loss(model_values, target_values)

            # recompute the weights
            print("total loss:", total_loss)
            if torch.is_tensor(total_loss):
                total_loss.backward()
                self.optimizer.step()
