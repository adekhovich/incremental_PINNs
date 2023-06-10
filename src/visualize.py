"""
Visualize outputs.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

def exact_u(Exact, task_id, x, t, nu, beta, rho, layers, N_f, L, source, u0_str, system, path, save=False):
    """Visualize exact solution."""
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=24)

    line = np.linspace(x.min(), x.max(), 2)[:,None]

    ax.set_xlabel('t', fontweight='bold', size=42)
    ax.set_ylabel('x', fontweight='bold', size=42)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 36}
    )

    if system == 'rd':
        ax.set_title(fr"$\rho$ = {rho[task_id]}, $\nu$ = {nu[task_id]}", fontsize=46)
    elif system == 'reaction':    
        ax.set_title(fr"$\rho$ = {rho[task_id]}", fontsize=46)
    elif system == 'diffusion':    
        ax.set_title(fr"$\nu$ = {nu[task_id]}", fontsize=46)  
    elif system == 'convection':    
        ax.set_title(fr"$\beta$ = {beta[task_id]}", fontsize=46)  

    ax.tick_params(labelsize=32)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f"{path}/uexact_{system}_nu{nu[task_id]}_beta{beta[task_id]}_rho{rho[task_id]}_Nf{N_f}_{layers}_L{L}_source{source}_{u0_str}.pdf")
        
    #plt.close()

    return None

def u_diff(Exact, U_pred, task_id, x, t, nu, beta, rho, seed, layers, N_f, L, source, lr, u0_str, system, path, relative_error = False, save=False):
    """Visualize abs(u_pred - u_exact)."""

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    if relative_error:
        h = ax.imshow(np.abs(Exact.T - U_pred.T)/np.abs(Exact.T), interpolation='nearest', cmap='binary',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    else:
        err = np.abs(Exact.T - U_pred.T)
        h = ax.imshow(err, interpolation='nearest', vmin=0, vmax=err.max(), cmap='binary',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=24)

    line = np.linspace(x.min(), x.max(), 2)[:,None]

    ax.set_xlabel('t', fontweight='bold', size=42)
    ax.set_ylabel('x', fontweight='bold', size=42)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 36}
    )
    
    if system == 'rd':
        ax.set_title(fr"$\rho$ = {rho[task_id]}, $\nu$ = {nu[task_id]}", fontsize=46)
    elif system == 'reaction':    
        ax.set_title(fr"$\rho$ = {rho[task_id]}", fontsize=46)
    elif system == 'diffusion':    
        ax.set_title(fr"$\nu$ = {nu[task_id]}", fontsize=46)  
    elif system == 'convection':    
        ax.set_title(fr"$\beta$ = {beta[task_id]}", fontsize=46)  

    ax.tick_params(labelsize=32)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f"{path}/udiff_{system}_nu{nu[task_id]}_beta{beta[task_id]}_rho{rho[task_id]}_Nf{N_f}_{layers}_L{L}_seed{seed}_source{source}_{u0_str}_lr{lr}.pdf")

    return None

def u_predict(U_pred, task_id, x, t, nu, beta, rho, layers, N_f, L, source, u0_str, system, path, save=False):
    """Visualize u_predicted."""

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    # colorbar for prediction: set min/max to ground truth solution.
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto', vmin=u_vals.min(0), vmax=u_vals.max(0))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=24)

    line = np.linspace(x.min(), x.max(), 2)[:,None]

    ax.set_xlabel('t', fontweight='bold', size=42)
    ax.set_ylabel('x', fontweight='bold', size=42)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 36}
    )
    
    if system == 'rd':
        ax.set_title(fr"$\rho$ = {rho[task_id]}, $\nu$ = {nu[task_id]}", fontsize=46)
    elif system == 'reaction':    
        ax.set_title(fr"$\rho$ = {rho[task_id]}", fontsize=46)
    elif system == 'diffusion':    
        ax.set_title(fr"$\nu$ = {nu[task_id]}", fontsize=46)  
    elif system == 'convection':    
        ax.set_title(fr"$\beta$ = {beta[task_id]}", fontsize=46)  

    ax.tick_params(labelsize=32)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f"{path}/upredicted_{system}_nu{nu[task_id]}_beta{beta[task_id]}_rho{rho[task_id]}_Nf{N_f}_{layers}_L{L}_seed{seed}_source{source}_{u0_str}_lr{lr}.pdf")

    plt.close()
    return None
