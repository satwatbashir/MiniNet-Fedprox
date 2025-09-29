# fedge/task.py

from collections import OrderedDict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from flwr_datasets.partitioner import DirichletPartitioner
import datasets as hfds  # HuggingFace Datasets (labels-only for partitioning)

# ───────────────────── Global seeding ─────────────────────
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Keep non-deterministic CuDNN for speed; flip to deterministic if you need
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True

# ───────────────────── Supported flags ─────────────────────
DATA_FLAGS = ["cifar10"]

# ───────────────────────── LeNet ──────────────────────────
class Net(nn.Module):
    def __init__(self, in_ch: int = 3, img_h: int = 32, img_w: int = 32, n_class: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 6, kernel_size=5)     # 3x32x32 -> 6x28x28
        self.pool  = nn.MaxPool2d(2, 2)                     # -> 6x14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)        # -> 16x10x10 -> pool -> 16x5x5
        # compute flatten dim dynamically
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(torch.zeros(1, in_ch, img_h, img_w))))
            x = self.pool(F.relu(self.conv2(x)))
            flat = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ─────────────── CIFAR-10 transforms (Flower style) ───────────────
# Standard mean/std for CIFAR-10
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

TFM_TRAIN = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
])

TFM_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
])

# ─────────────────────── load_data ───────────────────────
def load_data(dataset_flag: str,
              partition_id: int,
              num_partitions: int,
              batch_size: int = 32,
              alpha: float = 0.3,
              seed: int = 0):
    """
    Return (trainloader, testloader, n_classes) for a Dirichlet split of CIFAR-10.

    Args:
        dataset_flag: must be "cifar10"
        partition_id: integer in [0..num_partitions-1]
        num_partitions: number of clients
        batch_size: dataloader batch size
        alpha: Dirichlet concentration (heterogeneity)
        seed: random seed controlling partitioning and RNG for this run
    """
    if dataset_flag.lower() != "cifar10":
        raise ValueError("This loader only supports dataset_flag='cifar10'")

    # 1) Load full CIFAR-10 (download once)
    train_full = datasets.CIFAR10(root="./data", train=True, download=True, transform=TFM_TRAIN)
    test_full  = datasets.CIFAR10(root="./data", train=False, download=True, transform=TFM_TEST)

    # 2) Build HF labels datasets for partitioner (labels only)
    train_labels = [int(y) for y in train_full.targets]  # 50k
    test_labels  = [int(y) for y in test_full.targets]   # 10k
    hf_train = hfds.Dataset.from_dict({"label": train_labels}).cast_column("label", hfds.Value("int64"))
    hf_test  = hfds.Dataset.from_dict({"label":  test_labels}).cast_column("label",  hfds.Value("int64"))

    # 3) Partition train/test indices with DirichletPartitioner
    # Ensure deterministic partitioning per run
    set_global_seed(seed)
    train_partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=alpha,
        min_partition_size=100,
        self_balancing=False,
        shuffle=True,
        seed=seed,
    )
    train_partitioner.dataset = hf_train
    train_partitioner._determine_partition_id_to_indices_if_needed()
    train_indices = train_partitioner._partition_id_to_indices[partition_id]

    test_partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=alpha,
        min_partition_size=20,
        self_balancing=False,
        shuffle=True,
        seed=seed,
    )
    test_partitioner.dataset = hf_test
    test_partitioner._determine_partition_id_to_indices_if_needed()
    test_indices = test_partitioner._partition_id_to_indices[partition_id]

    # 4) Wrap with Subset + DataLoader
    trainloader = DataLoader(Subset(train_full, train_indices), batch_size=batch_size, shuffle=True,  num_workers=0)
    testloader  = DataLoader(Subset(test_full,  test_indices),  batch_size=batch_size, shuffle=False, num_workers=0)

    # 5) CIFAR-10 classes
    n_class = 10
    return trainloader, testloader, n_class

# ─────────────────── train / test / weights ───────────────────
def test(net: nn.Module, loader: DataLoader, device: torch.device):
    net.to(device)
    ce  = nn.CrossEntropyLoss()
    net.eval()
    total, correct, loss_sum = 0, 0, 0.0
    if len(loader) == 0:
        return 0.0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().view(-1)
            if labels.size(0) != images.size(0):
                continue
            out = net(images)
            loss_sum += ce(out, labels).item() * labels.size(0)

            correct  += (out.argmax(1) == labels).sum().item()
            total    += labels.size(0)
    if total == 0:
        return 0.0, 0.0
    return (loss_sum / max(1, total), correct / max(1, total))


def get_weights(net: nn.Module):
    return [v.cpu().numpy() for v in net.state_dict().values()]

def set_weights(net: nn.Module, w):
    net.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), w)}))
