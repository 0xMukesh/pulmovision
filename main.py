import torchvision.transforms as transforms
import torch.utils.data as data

from medmnist import INFO, ChestMNIST
from constants import DATASET_NAME, SHOULD_DOWNLOAD, BATCH_SIZE

import matplotlib.pyplot as plt


class DataClass(ChestMNIST):
    def __init__(self, split='train', transform=None, target_transform=None, download=False):
        super().__init__(split=split, transform=transform,
                         target_transform=target_transform, download=download)


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

train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_eval_loader = data.DataLoader(
    dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(
    dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

montage = train_dataset.montage(length=20)

plt.figure(figsize=(10, 10))
plt.imshow(montage, cmap='gray')
plt.axis('off')
plt.show()
