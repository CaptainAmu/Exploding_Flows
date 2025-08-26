import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_variation(vft, 
                      xrange, 
                      yrange, 
                      grid_size, 
                      t, 
                      verbose = False):
    """
    Given a trained flowmodel instance and time point (0 to 1), 
    capture the vector field, compute mean and std vect length, divergence and curl vect field.

    Args:
        vft: Flowmodel instance
        xrange, yrange: (min, max)
        grid_size: (nx, ny)
        t: float, time
        verbose: print description

    Return:
        ( vect_t, 
          stat_t = (vectt_len_mean, vectt_len_std), 
          div_t,
          curl_t)
    """
    nx, ny = grid_size
    xs = np.linspace(xrange[0], xrange[1], nx)
    ys = np.linspace(yrange[0], yrange[1], ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    grid_points = np.stack([X, Y], axis=-1).reshape(-1, 2)

    def eval_field(time):
        pts = torch.tensor(grid_points, dtype=torch.float32)
        tt = torch.full((pts.shape[0], 1), time, dtype=torch.float32)
        v = vft(pts, tt).detach().numpy()   # (N,2)
        return v.reshape(nx, ny, 2)

    # 1. vector field at t and t+dt
    V_t   = eval_field(t)

    # --- Part 1: norm statistics ---
    norms_t   = np.linalg.norm(V_t.reshape(-1, 2), axis=1)
    stat_t    = (norms_t.mean(), norms_t.std())

    # --- Part 2: divergence and curl ---
    def finite_diff(V):
        # V shape (nx, ny, 2)
        # component split
        Vx, Vy = V[..., 0], V[..., 1]
        # central differences (ignore boundary -> shape (nx-2, ny-2))
        dVx_dx = (Vx[2:,1:-1] - Vx[:-2,1:-1]) / (xs[2:,None] - xs[:-2,None])
        dVy_dy = (Vy[1:-1,2:] - Vy[1:-1,:-2]) / (ys[None,2:] - ys[None,:-2])
        div = dVx_dx + dVy_dy

        dVy_dx = (Vy[2:,1:-1] - Vy[:-2,1:-1]) / (xs[2:,None] - xs[:-2,None])
        dVx_dy = (Vx[1:-1,2:] - Vx[1:-1,:-2]) / (ys[None,2:] - ys[None,:-2])
        curl = dVy_dx - dVx_dy
        return div, curl

    div_t, curl_t     = finite_diff(V_t)

    if verbose:
        print(f'Time t vec_length mean {stat_t[0]}, std {stat_t[1]}')
        print(f'Time t div mean {div_t.mean()}, std {div_t.std()}')
        print(f'Time t curl mean {curl_t.mean()}, std {curl_t.std()}')

    out = dict(V_t=V_t, 
               stat_t=stat_t, 
               div_t=div_t, 
               curl_t=curl_t)
    return out



def analyze_flow_over_time(vft, xrange, yrange, grid_size, time_steps=100, verbose=False):
    """
    Given a trained flowmodel instance and time point (0 to 1), plot the 
    1.1 mean length 
    1.2 std length 
    2.1 mean divergence 
    2.2 mean curl
    of the time-dependent vector field at times t=0 to t=1, over region xrange*yrange.

    Args:
        vft: Flowmodel instance
        xrange: (minx, maxx)
        yrange: (miny, maxy)
        grid_size: (nx, ny)
        time_steps: number of time samples between 0 and 1
    """
    from tqdm import tqdm  # optional, for progress bar
    
    times = np.linspace(0, 1, time_steps)
    
    mean_lengths = []
    std_lengths  = []
    mean_divs    = []
    mean_curls   = []

    for t in tqdm(times):
        out = compute_variation(vft, xrange, yrange, grid_size, t, verbose=verbose)
        stat_t = out["stat_t"]   # (mean, std)
        div_t  = out["div_t"]
        curl_t = out["curl_t"]

        mean_lengths.append(stat_t[0])
        std_lengths.append(stat_t[1])
        mean_divs.append(div_t.mean())
        mean_curls.append(curl_t.mean())

    # ---- Plot 1: mean length and std length ----
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(times, mean_lengths, label="Mean length")
    plt.plot(times, std_lengths, label="Std length")
    plt.xlabel("Time t")
    plt.ylabel("Vector norm")
    plt.title("Vector length statistics vs. time")
    plt.legend()
    
    # ---- Plot 2: mean divergence and mean curl ----
    plt.subplot(1,2,2)
    plt.plot(times, mean_divs, label="Mean divergence")
    plt.plot(times, mean_curls, label="Mean curl")
    plt.xlabel("Time t")
    plt.ylabel("Value")
    plt.title("Mean vector field Divergence and Curl vs. time")
    plt.legend()
    
    plt.tight_layout()
    plt.show()