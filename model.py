import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = "cpu"  # gpu_0


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
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
        print(x, x.shape)
        # add dice
        x = torch.cat([x, torch.tensor([die], dtype=torch.float)])
        # 19,1
        print(x, x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        # flatten  to 2*3*3 + 1 = (18 + 1) = 19
        return self.fc3(x)



if __name__ == "__main__":
    board = [[[0 for row in range(3)] for col in range(3)] for player_board in range(2)]
    my_torch_tensor = torch.tensor(board, dtype=torch.float)
    print(my_torch_tensor)
    print(my_torch_tensor.shape)
    my_dqn = DQN()
    print(my_dqn.forward(my_torch_tensor, 5))

    # produce outputs for every state and compare them against reward
    # to produce loss
    # then backprop 

