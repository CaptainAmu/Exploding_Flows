import torch
from torch import nn, Tensor

class Flowmodel(nn.Module):
    def __init__(self, dim = 2, hidden_dim = 64):
        super(Flowmodel, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim), # +1 for time dimension 
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        ''' 
        Predicts the velocity at x_t at time t
        
        Args:
        x_t: tensor of 2 dim positions
        t: tensor of times
        '''
        return(self.layers(torch.cat((x_t, t), dim=-1)))
    
    def sample(self, x: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1,1).expand(x.shape[0], 1)
        t_end = t_end.view(1,1).expand(x.shape[0], 1)
        dt = t_end - t_start
        x_end = x + self.forward(x_t = x, t = t_start) * dt # Use forward Euler step
        return x_end
    
    def sample_mid_euler(self, x: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1,1).expand(x.shape[0], 1)
        t_end = t_end.view(1,1).expand(x.shape[0], 1)
        dt = t_end - t_start
        x_mid = x + self.forward(x_t = x, t = t_start + dt / 2) * dt
        return x_mid
    
    def sample_rk2(self, x: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        # Runge-Kutta 2nd order method
        t_start = t_start.view(1,1).expand(x.shape[0], 1)
        t_end = t_end.view(1,1).expand(x.shape[0], 1)
        dt = t_end - t_start
        return (x + dt * self.forward(x_t = x + self(x_t=x, t=t_start) * (t_end - t_start) / 2,
                                      t=t_start + (t_end - t_start) / 2))



