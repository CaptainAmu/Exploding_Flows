import torch
import torch.nn as nn
from torch.optim import Adam
from models.flow import Flowmodel
from datasets.synthetic import sample_dataset

def train_flow(dataset="moons", 
               filename=None, 
               n_samples=512,
               noise=0.5,
               scale=5,
               n_iters=10000, 
               lr=0.01):
    model = Flowmodel(dim=2, hidden_dim=64)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for iter in range(n_iters):
        x1 = sample_dataset(n_samples, noise, scale, dataset, filename)
        x0 = torch.randn_like(x1)
        t = torch.rand(x0.shape[0], 1)

        v_target = x1 - x0
        xt = x0 + v_target * t
        v_pred = model(xt, t)

        loss_value = loss_fn(v_pred, v_target)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        if iter % 1000 == 0:
            print(f"Step {iter}, Loss: {loss_value.item()}")

    return model
