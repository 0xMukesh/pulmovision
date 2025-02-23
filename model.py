import torch
from torch import nn
from medmnist import Evaluator
from constants import DATASET_NAME


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


def train_model(model, train_data_loader, optimizer, loss_function, device, num_epochs=3):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print(
                    f'[epoch {epoch + 1}, batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0


def test_model(model, split, test_data_loader, device):
    model.eval()

    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for inputs, targets in test_data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            targets = targets.to(torch.float32)
            outputs = outputs.softmax(dim=-1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.cpu().numpy()

        evaluator = Evaluator(DATASET_NAME, split)
        metrics = evaluator.evaluate(y_score)

        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))
