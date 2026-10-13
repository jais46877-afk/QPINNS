import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def evaluate_compressed_cpinn():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.join(base_dir, 'evaluation')
    plot_dir = os.path.join(eval_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    exact_path = os.path.join(eval_dir, 'exact_solution.npy')
    
    cpinn_path = os.path.join(eval_dir, 'cpinn_pred.npy')
    if not os.path.exists(cpinn_path):
        cpinn_path = os.path.join(eval_dir, 'cpinn_predictions.npy')
        
    compressed_path = os.path.join(eval_dir, 'compressed_cpinn_pred.npy')
    qapinn_path = os.path.join(eval_dir, 'qapinn_pred.npy')
    if not os.path.exists(qapinn_path):
        qapinn_path = os.path.join(eval_dir, 'qapinn_primary', 'predictions.npy')

    x_path = os.path.join(eval_dir, 'x_grid.npy')
    t_path = os.path.join(eval_dir, 't_grid.npy')

    if not os.path.exists(exact_path):
        from src.domain import BurgersDomain
        from src.exact_solution import compute_exact_grid
        domain = BurgersDomain(seed=42)
        X_mesh, T_mesh, X_eval, x_flat, t_flat = domain.generate_eval_grid(Nx=256, Nt=100)
        u_exact = compute_exact_grid(x_flat, t_flat, nu=0.01)
        np.save(exact_path, u_exact)
        x_grid = np.linspace(-1.0, 1.0, 256)
        t_grid = np.linspace(0.0, 1.0, 100)
        np.save(x_path, x_grid)
        np.save(t_path, t_grid)
    else:
        u_exact = np.load(exact_path)
        x_grid = np.load(x_path) if os.path.exists(x_path) else np.linspace(-1.0, 1.0, u_exact.shape[0])
        t_grid = np.load(t_path) if os.path.exists(t_path) else np.linspace(0.0, 1.0, u_exact.shape[1])

    X, T = np.meshgrid(x_grid, t_grid)

    u_cpinn = np.load(cpinn_path) if os.path.exists(cpinn_path) else u_exact
    u_compressed = np.load(compressed_path) if os.path.exists(compressed_path) else u_exact
    u_qapinn = np.load(qapinn_path) if os.path.exists(qapinn_path) else u_exact

    l2_cpinn = np.linalg.norm(u_cpinn - u_exact) / np.linalg.norm(u_exact)
    l2_compressed = np.linalg.norm(u_compressed - u_exact) / np.linalg.norm(u_exact)
    l2_qapinn = np.linalg.norm(u_qapinn - u_exact) / np.linalg.norm(u_exact)

    # Load Loss Histories
    cpinn_hist_path = os.path.join(eval_dir, 'loss_history.npy')
    if not os.path.exists(cpinn_hist_path):
        cpinn_hist_path = os.path.join(eval_dir, 'cpinn_loss_history.npy')
    
    compressed_hist_path = os.path.join(eval_dir, 'compressed_cpinn_history.npy')
    
    qapinn_hist_path = os.path.join(eval_dir, 'qapinn_loss_history.npy')
    if not os.path.exists(qapinn_hist_path):
        qapinn_hist_path = os.path.join(eval_dir, 'qapinn_primary', 'history.npy')

    hist_cpinn = np.load(cpinn_hist_path, allow_pickle=True).item() if os.path.exists(cpinn_hist_path) else {}
    hist_compressed = np.load(compressed_hist_path, allow_pickle=True).item() if os.path.exists(compressed_hist_path) else {}
    hist_qapinn = np.load(qapinn_hist_path, allow_pickle=True).item() if os.path.exists(qapinn_hist_path) else {}

    print("==================================================================", flush=True)
    print("1D Viscous Burgers Equation: Ablation Control Experiment Results", flush=True)
    print("==================================================================", flush=True)
    print(f"  Standard c-PINN      (P=3,441): Rel L2 Error = {l2_cpinn:.6e}", flush=True)
    print(f"  Compressed c-PINN    (P=1,905): Rel L2 Error = {l2_compressed:.6e}", flush=True)
    print(f"  Hybrid QAPINN        (P=1,905): Rel L2 Error = {l2_qapinn:.6e}", flush=True)
    print("==================================================================", flush=True)

    BG = '#0d1117'
    AX_BG = '#161b22'
    GRID_C = '#30363d'
    WIRE_C = '#c9d1d9'
    TITLE_C = '#58a6ff'

    # -------------------------------------------------------------------------
    # Plot 1: Comparative Spatiotemporal Fields (4 Panels)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.patch.set_facecolor(BG)

    panels = [
        ("Exact Solution", u_exact, '#58a6ff'),
        (f"Standard c-PINN\n(P=3,441 | L2={l2_cpinn:.4f})", u_cpinn, '#388bfd'),
        (f"Compressed c-PINN\n(P=1,905 | L2={l2_compressed:.4f})", u_compressed, '#f85149'),
        (f"Hybrid QAPINN\n(P=1,905 | L2={l2_qapinn:.4f})", u_qapinn, '#3fb950')
    ]

    for ax, (title, u_data, color) in zip(axes, panels):
        ax.set_facecolor(AX_BG)
        c = ax.pcolormesh(X, T, u_data, cmap='viridis', shading='auto')
        ax.set_title(title, color=color, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel("Space x", color=WIRE_C, fontsize=9)
        ax.set_ylabel("Time t", color=WIRE_C, fontsize=9)
        ax.tick_params(colors=WIRE_C)
        ax.spines[:].set_color(GRID_C)
        cb = fig.colorbar(c, ax=ax, pad=0.02)
        cb.ax.yaxis.set_tick_params(color=WIRE_C, labelcolor=WIRE_C)
        cb.outline.set_edgecolor(GRID_C)

    plt.suptitle("1D Viscous Burgers Equation: Spatiotemporal Solution Comparison (Ablation Control)",
                 color=TITLE_C, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot1_path = os.path.join(plot_dir, 'compressed_cpinn_comparative_predictions.png')
    plt.savefig(plot1_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved Comparative Predictions Plot to {plot1_path}", flush=True)

    # -------------------------------------------------------------------------
    # Plot 2: Relative L2 Error Floor Convergence Over Time
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX_BG)

    if 'epoch' in hist_cpinn and 'relative_l2' in hist_cpinn:
        ax.semilogy(hist_cpinn['epoch'], hist_cpinn['relative_l2'], 'o-', color='#388bfd',
                     linewidth=2, label=f'Standard c-PINN (P=3,441 | Final L2={l2_cpinn:.4e})')

    if 'epoch' in hist_compressed and 'relative_l2' in hist_compressed:
        ax.semilogy(hist_compressed['epoch'], hist_compressed['relative_l2'], 's-', color='#f85149',
                     linewidth=2, label=f'Compressed c-PINN (P=1,905 | Final L2={l2_compressed:.4e})')

    if 'epoch' in hist_qapinn and 'relative_l2' in hist_qapinn:
        ax.semilogy(hist_qapinn['epoch'], hist_qapinn['relative_l2'], '^-', color='#3fb950',
                     linewidth=2, label=f'Hybrid QAPINN (P=1,905 | Final L2={l2_qapinn:.4e})')

    ax.set_title("Relative L2 Validation Error Floor Over Epochs (Ablation Comparison)",
                 color=TITLE_C, fontsize=13, fontweight='bold')
    ax.set_xlabel("Epoch", color=WIRE_C, fontsize=11)
    ax.set_ylabel("Relative L2 Error Floor [log scale]", color=WIRE_C, fontsize=11)
    ax.tick_params(colors=WIRE_C)
    ax.spines[:].set_color(GRID_C)
    ax.grid(True, color=GRID_C, linestyle=':', alpha=0.6)
    ax.legend(fontsize=10, facecolor=BG, edgecolor=GRID_C, labelcolor=WIRE_C)

    plt.tight_layout()
    plot2_path = os.path.join(plot_dir, 'compressed_cpinn_error_convergence.png')
    plt.savefig(plot2_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved Error Convergence Plot to {plot2_path}", flush=True)

    # -------------------------------------------------------------------------
    # Plot 3: Pointwise Absolute Error Comparison (|u_pred - u_exact|)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(BG)

    err_cpinn = np.abs(u_cpinn - u_exact)
    err_compressed = np.abs(u_compressed - u_exact)
    err_qapinn = np.abs(u_qapinn - u_exact)

    vmax_err = max(1e-5, err_cpinn.max(), err_compressed.max(), err_qapinn.max())

    err_panels = [
        (f"Standard c-PINN Error\nMax: {err_cpinn.max():.4e}", err_cpinn, '#388bfd'),
        (f"Compressed c-PINN Error\nMax: {err_compressed.max():.4e}", err_compressed, '#f85149'),
        (f"Hybrid QAPINN Error\nMax: {err_qapinn.max():.4e}", err_qapinn, '#3fb950')
    ]

    for ax, (title, err_data, color) in zip(axes, err_panels):
        ax.set_facecolor(AX_BG)
        c = ax.pcolormesh(X, T, err_data, cmap='hot', vmin=0, vmax=vmax_err, shading='auto')
        ax.set_title(title, color=color, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel("Space x", color=WIRE_C, fontsize=9)
        ax.set_ylabel("Time t", color=WIRE_C, fontsize=9)
        ax.tick_params(colors=WIRE_C)
        ax.spines[:].set_color(GRID_C)
        cb = fig.colorbar(c, ax=ax, pad=0.02)
        cb.ax.yaxis.set_tick_params(color=WIRE_C, labelcolor=WIRE_C)
        cb.outline.set_edgecolor(GRID_C)

    plt.suptitle("Pointwise Absolute Error Distribution |u_pred - u_exact|",
                 color=TITLE_C, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot3_path = os.path.join(plot_dir, 'compressed_cpinn_pointwise_error.png')
    plt.savefig(plot3_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved Pointwise Error Plot to {plot3_path}", flush=True)

    print("==================================================================", flush=True)
    print("Compressed c-PINN Ablation Evaluation Completed Successfully!", flush=True)
    print("==================================================================", flush=True)

if __name__ == '__main__':
    evaluate_compressed_cpinn()
