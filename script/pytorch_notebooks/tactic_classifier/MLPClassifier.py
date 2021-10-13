import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        hidden_layer_input = self.fc1(input)
        weighted_sum = self.fc2(hidden_layer_input)
        output = F.softmax(weighted_sum, dim=1)
        return output
