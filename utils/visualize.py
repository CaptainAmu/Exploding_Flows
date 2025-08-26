import matplotlib.pyplot as plt
import torch
import math
    
def visualize_flow_evolution(flow_model, x0, n_steps=8, xlim=(-3,3), ylim=(-3,3), max_cols=10):
    '''
    Plot sampled points from the trained flow_model using forward_Euler.
    '''
    time_steps = torch.linspace(0, 1, n_steps + 1).view(-1, 1)
    
    total_plots = n_steps + 1 
    n_cols = min(total_plots, max_cols)
    n_rows = math.ceil(total_plots / max_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), sharex=True, sharey=True)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    x = x0
    for idx in range(total_plots):
        row = idx // max_cols
        col = idx % max_cols
        ax = axes[row, col]
        
        if idx == 0:
            ax.scatter(x0[:, 0], x0[:, 1], s=10, alpha=0.6)
            ax.set_title("Initial State")
        else:
            x = flow_model.sample(x, time_steps[idx-1], time_steps[idx])
            ax.scatter(x[:, 0].detach(), x[:, 1].detach(), s=10, alpha=0.6)
            ax.set_title(f"Step {idx}")
        
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    
    for idx in range(total_plots, n_rows * n_cols):
        row = idx // max_cols
        col = idx % max_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()