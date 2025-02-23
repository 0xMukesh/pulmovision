import torch
from torch import nn
import numpy as np
from medmnist import Evaluator
from sklearn.metrics import f1_score
import torchvision.models as models

from constants import DATASET_NAME, DATASET_CLASSES


class Model(nn.Module):
    def __init__(self, num_classes=14, grayscale_input=True):
        super(Model, self).__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')

        for name, param in self.resnet.named_parameters():
            if not name.startswith("layer4") and not name.startswith("fc"):
                param.requires_grad = False

        if grayscale_input:
            self.resnet.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


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

        # converting probabilities to binary predictions with 0.5 as the threshold
        y_pred = (y_score >= 0.5).astype(np.float32)

        evaluator = Evaluator(DATASET_NAME, split)
        auc_acc_metrics = evaluator.evaluate(y_score)

        # aggregates all classes together
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        # calculates f1 for each class and takes the average
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        per_class_f1 = f1_score(y_true, y_pred, average=None)

        print(
            f"{split}  auc: {auc_acc_metrics[0]:.3f}  acc:{auc_acc_metrics[1]:.3f}")
        print(f"{split}  micro_f1: {micro_f1:.3f}  macro_f1: {macro_f1:.3f}\n")
        print("per-class f1:")

        for i, (class_name, f1) in enumerate(zip(DATASET_CLASSES, per_class_f1)):
            print(f"{class_name}: {f1:.3f}")

        return auc_acc_metrics, micro_f1, macro_f1, per_class_f1
