import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

def calculate_mean_std(dataset_root='./workspace/datasets/cifar10'):
    """
    Calculates the mean and standard deviation of the CIFAR10 training dataset.
    """
    # Transform to convert images to tensors (scaled to [0, 1])
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the training dataset WITHOUT normalization
    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_root,
        train=True,
        download=True,
        transform=transform
    )

    # Use a DataLoader to iterate efficiently
    # Use a larger batch size for faster computation if memory allows
    loader = DataLoader(train_dataset, batch_size=1024, shuffle=False, num_workers=4)

    mean = 0.
    std = 0.
    nb_samples = 0.

    print("Calculating mean and std... This might take a few minutes.")

    # Use torch.no_grad() for efficiency
    with torch.no_grad():
        for data, _ in loader:
            batch_samples = data.size(0)
            # Reshape data: (batch_size, channels, height, width) -> (batch_size, channels, height*width)
            data = data.view(batch_samples, data.size(1), -1)
            # Calculate sum and sum of squares per batch
            mean += data.mean(2).sum(0) # Sum means across pixels, then sum across batch
            std += data.std(2).sum(0)   # Sum stds across pixels, then sum across batch
            nb_samples += batch_samples

    # Calculate overall mean and std
    mean /= nb_samples
    std /= nb_samples

    print(f"Calculated Mean: {mean.tolist()}")
    print(f"Calculated Std: {std.tolist()}")
    print(f"Based on {int(nb_samples)} training samples.")

if __name__ == '__main__':
    calculate_mean_std() 