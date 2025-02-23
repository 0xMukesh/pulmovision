import torch
from torch import nn
from torch import optim
import torch.utils.data as data
import torchvision.transforms as transforms
from medmnist import ChestMNIST
from model import Model, train_model, test_model


class DataClass(ChestMNIST):
    def __init__(self, split="train", transform=None, target_transform=None, download=False):
        super().__init__(split=split, transform=transform,
                         target_transform=target_transform, download=download)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5]),
    # randomly position the image during training
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])

test_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

train_dataset = DataClass(
    split="train", transform=train_data_transform, download=True)
test_dataset = DataClass(
    split="test", transform=test_data_transform, download=True)

train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

model = Model(in_channels=1, num_classes=14).to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, optimizer,
            loss_function, device, num_epochs=3)
test_model(model, "test", test_loader, device)

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "model.pth")
