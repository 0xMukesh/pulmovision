import torch
from torch import nn
import torch.utils.data as data

from medmnist import Evaluator

from constants import N_EPOCHS, DATASET_NAME


class Model(nn.Module):
    def __init__(self, in_channels, num_classes):
        # calls the parent class constructor
        super(Model, self).__init__()

        self.layer_one = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer_two = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer_three = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer_four = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer_five = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fully_connected = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer_one(x)
        x = self.layer_two(x)
        x = self.layer_three(x)
        x = self.layer_four(x)
        x = self.layer_five(x)
        x = x.view(x.size(0), -1)  # flattens the feature maps
        x = self.fully_connected(x)
        return x


def train_model(model, data_loader, optimizer, loss_function):
    for epoch in range(N_EPOCHS):
        model.train()

        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            targets = targets.to(torch.float32)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
