import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleClassifier(nn.Module):

    def __init__(self, input_size, output_size):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, input):
        weighted_sum = self.fc1(input)
        output = F.softmax(weighted_sum, dim=1)
        return output
