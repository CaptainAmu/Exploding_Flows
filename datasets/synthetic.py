import torch
import numpy as np
from sklearn.datasets import make_moons, make_swiss_roll
from PIL import Image
from os.path import join

def sample_moons(n_samples=256, noise=0.05):
    x = torch.tensor(make_moons(n_samples=n_samples, noise=noise)[0], dtype=torch.float32)
    return x

def sample_swiss_roll(n_samples=256, noise=0.5, scale=5):
    X, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
    x = torch.tensor(X[:, [0, 2]], dtype=torch.float32) / scale
    return x


def manual_sample(filename, n_samples=256, noise=0.5, scale=5, white_background=True):
    path = join('datasets', filename)
    img = Image.open(path).convert("L")   # Convert to grayscale
    arr = np.array(img)
    mask = arr < 127 if white_background else arr > 127 # Convert to BW
    coords = np.argwhere(mask)   # (row, col)
    if len(coords) == 0:
        raise ValueError("No non-zero pixels. Please check if image has enough contrast.")
    idx = np.random.choice(len(coords), size=n_samples, replace=True)
    sampled = coords[idx].astype(np.float32)
    sampled += np.random.randn(*sampled.shape) * noise
    sampled = (sampled - sampled.mean(0)) / sampled.std(0) # Scale
    sampled = sampled / scale
    sampled = sampled[:, [1, 0]]  # Flip coordinates and adjust
    sampled[:,1] = -sampled[:,1]
    x = torch.tensor(sampled, dtype=torch.float32)
    return x


def sample_dataset(n_samples=256, noise=0.05, scale=5, dataset = 'moons', filename = None):
    assert dataset in ['moons', 'swiss_roll', 'manual'], 'Dataset Undefined. Must select between moons, swiss_roll, and manual'
    scale = 1/scale
    if dataset == 'moons':
        x = sample_moons(n_samples=n_samples, noise=noise)
        x = x / scale
    elif dataset == 'swiss_roll':
        x = sample_swiss_roll(n_samples=n_samples, noise=noise, scale=scale)
    elif dataset == 'manual':
        assert filename, 'Filename is none. Please provide filename'
        x = manual_sample(filename, n_samples=n_samples, noise=noise, scale=scale)
    return x


def draw_compare(x1):
    '''
    Given sampled points from target distribution, draw same size from initial distribution, plot paired plot.

    Args: 
        x1: Torch tensor (of shape (#samples, 2))
    '''
    import torch
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
    pts0 = torch.randn_like(x1).numpy()
    pts1 = x1.numpy()
    axes[0].scatter(pts0[:,0], pts0[:,1], s=10, alpha=0.6)
    axes[0].axis("equal")
    axes[0].set_title("Point cloud sampled from initial normal")
    axes[1].scatter(pts1[:,0], pts1[:,1], s=10, alpha=0.6)
    axes[1].axis("equal")
    axes[1].set_title("Point cloud sampled from target")
    plt.show()
