import torchvision.transforms as transforms

import torch
from torch import nn
from torch import optim
import torch.utils.data as data

from medmnist import INFO, ChestMNIST, Evaluator

from constants import DATASET_NAME, SHOULD_DOWNLOAD, BATCH_SIZE, LEARNING_RATE, N_EPOCHS
from model import Model, train_model


class DataClass(ChestMNIST):
    def __init__(self, split='train', transform=None, target_transform=None, download=False):
        super().__init__(split=split, transform=transform,
                         target_transform=target_transform, download=download)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

info = INFO[DATASET_NAME]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

data_transform = transforms.Compose([
    # converts the raw PIL images to pytorch tensor and normalizes the RGB values to [0, 1]
    transforms.ToTensor(),
    # transforms the [0, 1] range to [-1, 1]
    # output = (input - mean)/std
    transforms.Normalize(mean=[.5], std=[.5])
])

train_dataset = DataClass(
    split='train', transform=data_transform, download=SHOULD_DOWNLOAD)
test_dataset = DataClass(
    split="test", transform=data_transform, download=SHOULD_DOWNLOAD)

train_data_loader = data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_eval_data_loader = data.DataLoader(
    dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_data_loader = data.DataLoader(
    dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)


model = Model(in_channels=n_channels, num_classes=n_classes).to(device=device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

train_model(model, train_data_loader, optimizer, loss_function)


def test(split):
    model.eval()

    y_true = torch.tensor([])
    y_score = torch.tensor([])

    data_loader = train_eval_data_loader if split == "train" else test_data_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            if task == "multi-label, binary-class":
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()

        evaluator = Evaluator(DATASET_NAME, split)
        metrics = evaluator.evaluate(y_score)

        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))


print("Testing the model...")
test(split="train")
