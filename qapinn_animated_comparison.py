"""
╔══════════════════════════════════════════════════════════════════════════╗
║    QAPINN ANIMATED COMPARISON — Side-by-Side Time Evolution (Idea 2)     ║
║    For QUEST-AI 2026 Pitch Demonstration                                ║
║    Velamala Manikanta et al., Andhra University, 2026                   ║
╚══════════════════════════════════════════════════════════════════════════╝

Creates high-impact animated side-by-side GIFs & MP4s showing QAPINN vs c-PINN vs Exact
solution evolution across time t = 0.0 to 1.0 for Burgers', Heat, and Wave equations.
"""

import sys
import os
import warnings
import time
warnings.filterwarnings("ignore")

# Fix Windows PowerShell UTF-8 encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.qapinn import HybridQAPINN
from src.cpinn import ClassicalPINN
from src.exact_solution import cole_hopf_exact_point

# ─── Exact Analytical Solutions ──────────────────────────────────────────────
def exact_heat(x, t, alpha=0.05):
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)

def exact_wave(x, t, c=1.0):
    return np.sin(np.pi * x) * np.cos(c * np.pi * t)

def exact_burgers(x, t, nu=0.01):
    if t == 0.0:
        return -np.sin(np.pi * x)
    return cole_hopf_exact_point(x, t, nu=nu)

EQ_CONFIG = {
    'burgers': {
        'name': "1D Viscous Burgers' Equation (Shock Wave Formation)",
        'exact': exact_burgers,
        'domain_x': (-1.0, 1.0),
        'domain_t': (0.0, 1.0),
        'ylim': (-1.1, 1.1),
        'unit': "Velocity u(x,t)",
    },
    'heat': {
        'name': "1D Parabolic Heat Equation (Thermal Dissipation)",
        'exact': exact_heat,
        'domain_x': (-1.0, 1.0),
        'domain_t': (0.0, 1.0),
        'ylim': (-0.1, 1.1),
        'unit': "Temperature T(x,t)",
    },
    'wave': {
        'name': "1D Hyperbolic Wave Equation (Standing Wave Reflection)",
        'exact': exact_wave,
        'domain_x': (-1.0, 1.0),
        'domain_t': (0.0, 1.0),
        'ylim': (-1.2, 1.2),
        'unit': "Displacement u(x,t)",
    }
}

MODEL_PATHS = {
    'qapinn': 'evaluation/qapinn_primary/model.pt',
    'cpinn':  'evaluation/cpinn_model.pt',
}

def load_models():
    print("  Loading trained QAPINN and c-PINN models...")
    qapinn = HybridQAPINN(n_qubits=4, n_layers=2, position='first',
                         measurement_mode='expectation', encoding='angle',
                         entanglement='linear', hidden_features=40)
    if os.path.exists(MODEL_PATHS['qapinn']):
        qapinn.load_state_dict(torch.load(MODEL_PATHS['qapinn'], map_location='cpu', weights_only=True))
        qapinn.eval()
        print("  ✓ QAPINN loaded successfully.")

    cpinn = ClassicalPINN(hidden_features=40, num_layers=3)
    if os.path.exists(MODEL_PATHS['cpinn']):
        cpinn.load_state_dict(torch.load(MODEL_PATHS['cpinn'], map_location='cpu', weights_only=True))
        cpinn.eval()
        print("  ✓ c-PINN loaded successfully.")

    return qapinn, cpinn

def predict_batch(model, x_arr, t_arr):
    X = torch.tensor(np.stack([x_arr, t_arr], axis=1), dtype=torch.float32)
    with torch.no_grad():
        return model(X).cpu().numpy().flatten()

def generate_animation(equation='burgers', n_frames=60, fps=15, save_gif=True, show_live=False):
    if not show_live:
        matplotlib.use('Agg')
    else:
        try:
            matplotlib.use('TkAgg')
        except Exception:
            pass

    config = EQ_CONFIG[equation]
    qapinn, cpinn = load_models()

    x_arr = np.linspace(config['domain_x'][0], config['domain_x'][1], 200)
    t_frames = np.linspace(config['domain_t'][0], config['domain_t'][1], n_frames)

    print(f"\n  Pre-computing {n_frames} spacetime frames for {config['name']}...")
    
    # Pre-calculate predictions across frames for super-smooth animation
    u_qapinn_frames = []
    u_cpinn_frames  = []
    u_exact_frames  = []
    l2_qapinn_list  = []
    l2_cpinn_list   = []

    for idx, t_val in enumerate(t_frames):
        t_arr = np.full_like(x_arr, t_val)
        u_q = predict_batch(qapinn, x_arr, t_arr)
        u_c = predict_batch(cpinn,  x_arr, t_arr)
        u_e = np.array([config['exact'](x, t_val) for x in x_arr])

        l2_q = np.linalg.norm(u_q - u_e) / (np.linalg.norm(u_e) + 1e-10) * 100
        l2_c = np.linalg.norm(u_c - u_e) / (np.linalg.norm(u_e) + 1e-10) * 100

        u_qapinn_frames.append(u_q)
        u_cpinn_frames.append(u_c)
        u_exact_frames.append(u_e)
        l2_qapinn_list.append(l2_q)
        l2_cpinn_list.append(l2_c)

    # ── Figure Setup ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#0f172a')
    ax1.set_facecolor('#1e293b')
    ax2.set_facecolor('#1e293b')

    # Styling left panel (Solution Profiles)
    ax1.set_xlim(config['domain_x'])
    ax1.set_ylim(config['ylim'])
    ax1.set_xlabel('Spatial Coordinate (x)', color='#e2e8f0', fontsize=11, fontweight='bold')
    ax1.set_ylabel(config['unit'], color='#e2e8f0', fontsize=11, fontweight='bold')
    ax1.tick_params(colors='#94a3b8')
    ax1.grid(True, linestyle='--', alpha=0.3, color='#475569')

    line_exact, = ax1.plot([], [], 'w-',  lw=2.5, label='Exact Analytical', zorder=3)
    line_qapinn, = ax1.plot([], [], '#38bdf8', lw=2.2, linestyle='--', label='QAPINN (Ours)', zorder=4)
    line_cpinn,  = ax1.plot([], [], '#f43f5e', lw=2.0, linestyle=':',  label='Classical c-PINN', zorder=2)
    time_text_1  = ax1.text(0.04, 0.90, '', transform=ax1.transAxes, color='#fbbf24', 
                            fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='#0f172a', alpha=0.8, edgecolor='#fbbf24'))
    ax1.legend(loc='upper right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9.5)
    ax1.set_title("Solution Field Evolution  u(x, t)", color='#f8fafc', fontsize=12, fontweight='bold')

    # Styling right panel (Pointwise Absolute Error)
    ax2.set_xlim(config['domain_x'])
    max_err_limit = max(np.max(np.abs(np.array(u_cpinn_frames) - np.array(u_exact_frames))), 0.3)
    ax2.set_ylim(0, max_err_limit * 1.1)
    ax2.set_xlabel('Spatial Coordinate (x)', color='#e2e8f0', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Absolute Error |u_pred - u_exact|', color='#e2e8f0', fontsize=11, fontweight='bold')
    ax2.tick_params(colors='#94a3b8')
    ax2.grid(True, linestyle='--', alpha=0.3, color='#475569')

    fill_qapinn = [ax2.fill_between([], [], color='#38bdf8', alpha=0.5)]
    fill_cpinn  = [ax2.fill_between([], [], color='#f43f5e', alpha=0.35)]
    
    line_err_q, = ax2.plot([], [], '#0284c7', lw=1.5, label='QAPINN Error')
    line_err_c, = ax2.plot([], [], '#e11d48', lw=1.5, label='c-PINN Error')

    time_text_2 = ax2.text(0.04, 0.90, '', transform=ax2.transAxes, color='#38bdf8', 
                           fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='#0f172a', alpha=0.8, edgecolor='#38bdf8'))
    ax2.legend(loc='upper right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9.5)
    ax2.set_title("Pointwise Error Comparison", color='#f8fafc', fontsize=12, fontweight='bold')

    fig.suptitle(f"QAPINN vs Classical PINN Live Pitch Demo\n{config['name']}", 
                 color='#ffffff', fontsize=14, fontweight='bold', y=0.98)

    def init():
        line_exact.set_data([], [])
        line_qapinn.set_data([], [])
        line_cpinn.set_data([], [])
        line_err_q.set_data([], [])
        line_err_c.set_data([], [])
        time_text_1.set_text('')
        time_text_2.set_text('')
        return line_exact, line_qapinn, line_cpinn, line_err_q, line_err_c, time_text_1, time_text_2

    def animate(i):
        t_val = t_frames[i]
        u_q = u_qapinn_frames[i]
        u_c = u_cpinn_frames[i]
        u_e = u_exact_frames[i]

        err_q = np.abs(u_q - u_e)
        err_c = np.abs(u_c - u_e)

        line_exact.set_data(x_arr, u_e)
        line_qapinn.set_data(x_arr, u_q)
        line_cpinn.set_data(x_arr, u_c)

        line_err_q.set_data(x_arr, err_q)
        line_err_c.set_data(x_arr, err_c)

        # Remove old fill collections
        for coll in fill_qapinn + fill_cpinn:
            try:
                coll.remove()
            except Exception:
                pass

        fill_qapinn[0] = ax2.fill_between(x_arr, err_q, color='#38bdf8', alpha=0.5)
        fill_cpinn[0]  = ax2.fill_between(x_arr, err_c, color='#f43f5e', alpha=0.35)

        time_text_1.set_text(f"Time t = {t_val:.3f} s")
        time_text_2.set_text(f"Rel L₂ Error:  QAPINN: {l2_qapinn_list[i]:.2f}%  |  c-PINN: {l2_cpinn_list[i]:.2f}%")

        return line_exact, line_qapinn, line_cpinn, line_err_q, line_err_c, time_text_1, time_text_2, fill_qapinn[0], fill_cpinn[0]

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, 
                                   interval=1000//fps, blit=False)

    out_gif = f"pitch_demo_animation_{equation}.gif"
    if save_gif:
        print(f"  Saving animation to {out_gif} (this takes ~10 seconds)...")
        writer = animation.PillowWriter(fps=fps)
        anim.save(out_gif, writer=writer)
        print(f"  ✓ Saved high-resolution GIF → {out_gif}")

    if show_live:
        print("  Showing live animation window (close window to exit)...")
        plt.show()

    plt.close()
    return out_gif

if __name__ == '__main__':
    eq_choice = sys.argv[1] if len(sys.argv) > 1 else 'burgers'
    generate_animation(equation=eq_choice, n_frames=50, fps=12, save_gif=True, show_live=False)
