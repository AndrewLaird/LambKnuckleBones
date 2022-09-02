import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from datapoint import DataPoint

device = "cpu"  # gpu_0


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
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(128, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 128)
        self.fc5 = nn.Linear(128, 1)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

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
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)

    def train(self, training_data: list[DataPoint]):
        # start loss at zero
        for i in range(10):
            self.optimizer.zero_grad()
            total_loss = 0
            model_values, true_values = torch.Tensor(), torch.Tensor()
            for datapoint in training_data:
                board, number_rolled = datapoint.state

                # run it through the network to get what it thinks it should be
                model_value = self.forward(torch.tensor(board), number_rolled)
                # get it's true value form bellman (hear me out, just use value for now)
                true_value = torch.tensor([datapoint.reward], dtype=torch.float)
                model_values = torch.cat((model_values, model_value))
                true_values = torch.cat((true_values, true_value))

            # set loss equal to MSE between those
            # total_loss += self.loss(model_value, true_value)
            total_loss = self.loss(model_values, true_values)

            # recompute the weights
            print("total loss:", total_loss)
            if torch.is_tensor(total_loss):
                total_loss.backward()
                self.optimizer.step()


if __name__ == "__main__":
    board = [[[0 for row in range(3)] for col in range(3)] for player_board in range(2)]
    my_torch_tensor = torch.tensor(board, dtype=torch.float)
    print(my_torch_tensor)
    print(my_torch_tensor.shape)
    my_default = DefaultModel()
    print(my_default.forward(my_torch_tensor, 5))

    # produce outputs for every state and compare them against reward
    # to produce loss
    # then backprop
