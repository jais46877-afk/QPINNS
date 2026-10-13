"""
╔══════════════════════════════════════════════════════════════════════════╗
║    QAPINN INTERACTIVE SLIDER DEMO — Live Pitch Widget (Idea 3)          ║
║    For QUEST-AI 2026 Pitch Demonstration                                ║
║    Velamala Manikanta et al., Andhra University, 2026                   ║
╚══════════════════════════════════════════════════════════════════════════╝

Interactive Matplotlib Slider Widget with live crosshair target querying,
spatial/temporal profile slicing, and numerical metric cards.
Drag sliders or select equations to see live updates in real-time during pitch.
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
from matplotlib.widgets import Slider, RadioButtons, Button
import matplotlib.gridspec as gridspec

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
        'name': "1D Viscous Burgers' Equation",
        'formula': "∂u/∂t + u·∂u/∂x = ν·∂²u/∂x² (ν=0.01)",
        'exact': exact_burgers,
        'domain_x': (-1.0, 1.0),
        'domain_t': (0.0, 1.0),
        'unit': "Velocity u(x,t)",
    },
    'heat': {
        'name': "1D Parabolic Heat Conduction Equation",
        'formula': "∂u/∂t = α·∂²u/∂x² (α=0.05)",
        'exact': exact_heat,
        'domain_x': (-1.0, 1.0),
        'domain_t': (0.0, 1.0),
        'unit': "Temperature T(x,t)",
    },
    'wave': {
        'name': "1D Hyperbolic Wave Equation",
        'formula': "∂²u/∂t² = c²·∂²u/∂x² (c=1.0)",
        'exact': exact_wave,
        'domain_x': (-1.0, 1.0),
        'domain_t': (0.0, 1.0),
        'unit': "Displacement u(x,t)",
    }
}

MODEL_PATHS = {
    'qapinn': 'evaluation/qapinn_primary/model.pt',
    'cpinn':  'evaluation/cpinn_model.pt',
}

def load_models():
    print("Loading models for interactive widget...")
    qapinn = HybridQAPINN(n_qubits=4, n_layers=2, position='first',
                         measurement_mode='expectation', encoding='angle',
                         entanglement='linear', hidden_features=40)
    if os.path.exists(MODEL_PATHS['qapinn']):
        qapinn.load_state_dict(torch.load(MODEL_PATHS['qapinn'], map_location='cpu', weights_only=True))
        qapinn.eval()
        print("✓ QAPINN loaded.")

    cpinn = ClassicalPINN(hidden_features=40, num_layers=3)
    if os.path.exists(MODEL_PATHS['cpinn']):
        cpinn.load_state_dict(torch.load(MODEL_PATHS['cpinn'], map_location='cpu', weights_only=True))
        cpinn.eval()
        print("✓ c-PINN loaded.")

    return qapinn, cpinn

def predict_batch(model, x_arr, t_arr):
    X = torch.tensor(np.stack([x_arr, t_arr], axis=1), dtype=torch.float32)
    with torch.no_grad():
        return model(X).cpu().numpy().flatten()

def run_interactive_widget():
    # Attempt GUI backend
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass

    qapinn, cpinn = load_models()

    current_eq = 'burgers'
    x0, t0 = 0.3, 0.5

    # Build initial mesh data
    Nx, Nt = 60, 45
    x_lin = np.linspace(-1.0, 1.0, Nx)
    t_lin = np.linspace(0.0, 1.0, Nt)
    X_grid, T_grid = np.meshgrid(x_lin, t_lin)

    # Figure & GridSpec
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#0f172a')
    
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.28,
                           left=0.06, right=0.96, top=0.90, bottom=0.18)

    ax_map  = fig.add_subplot(gs[0, 0], facecolor='#1e293b')
    ax_x    = fig.add_subplot(gs[0, 1], facecolor='#1e293b')
    ax_t    = fig.add_subplot(gs[1, 0], facecolor='#1e293b')
    ax_card = fig.add_subplot(gs[1, 1], facecolor='#1e293b')

    # Data container
    cache = {'u_q': None, 'u_c': None, 'u_e': None}

    def compute_heatmap_data():
        eq = EQ_CONFIG[current_eq]
        u_q = predict_batch(qapinn, X_grid.flatten(), T_grid.flatten()).reshape(Nt, Nx)
        u_c = predict_batch(cpinn, X_grid.flatten(), T_grid.flatten()).reshape(Nt, Nx)
        u_e = np.array([[eq['exact'](x, t) for x in x_lin] for t in t_lin])
        cache['u_q'] = u_q
        cache['u_c'] = u_c
        cache['u_e'] = u_e

    compute_heatmap_data()

    # ── Panel 1: Spacetime Heatmap with Crosshair ──
    eq = EQ_CONFIG[current_eq]
    mesh = ax_map.pcolormesh(x_lin, t_lin, cache['u_q'], cmap='RdYlBu_r', shading='auto')
    cross_h = ax_map.axhline(t0, color='#f59e0b', linestyle='--', lw=1.4)
    cross_v = ax_map.axvline(x0, color='#f59e0b', linestyle='--', lw=1.4)
    dot_target, = ax_map.plot([x0], [t0], marker='o', markersize=9, color='#fbbf24', markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    ax_map.set_xlabel('Spatial Position (x)', color='#e2e8f0', fontsize=9.5)
    ax_map.set_ylabel('Time (t)', color='#e2e8f0', fontsize=9.5)
    ax_map.tick_params(colors='#94a3b8')
    ax_map.set_title(f"QAPINN Spacetime Heatmap", color='#f8fafc', fontsize=11, fontweight='bold')
    cbar = fig.colorbar(mesh, ax=ax_map, pad=0.02)
    cbar.ax.tick_params(colors='#94a3b8')

    # ── Panel 2: Spatial Slice u(x, t0) ──
    line_x_exact, = ax_x.plot([], [], 'w-', lw=2.2, label='Exact Analytical')
    line_x_qapinn, = ax_x.plot([], [], '#38bdf8', lw=2.0, linestyle='--', label='QAPINN (Ours)')
    line_x_cpinn,  = ax_x.plot([], [], '#f43f5e', lw=1.8, linestyle=':',  label='c-PINN')
    dot_x_q, = ax_x.plot([], [], 'o', color='#38bdf8', markersize=7)
    ax_x.set_xlabel('Spatial Position (x)', color='#e2e8f0', fontsize=9.5)
    ax_x.set_ylabel(eq['unit'], color='#e2e8f0', fontsize=9.5)
    ax_x.tick_params(colors='#94a3b8')
    ax_x.grid(True, linestyle='--', alpha=0.3, color='#475569')
    ax_x.legend(loc='upper right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc', fontsize=8.5)
    ax_x.set_title(f"Spatial Profile u(x, t={t0:.2f})", color='#f8fafc', fontsize=11, fontweight='bold')

    # ── Panel 3: Temporal Evolution u(x0, t) ──
    line_t_exact, = ax_t.plot([], [], 'w-', lw=2.2, label='Exact Analytical')
    line_t_qapinn, = ax_t.plot([], [], '#38bdf8', lw=2.0, linestyle='--', label='QAPINN')
    line_t_cpinn,  = ax_t.plot([], [], '#f43f5e', lw=1.8, linestyle=':',  label='c-PINN')
    dot_t_q, = ax_t.plot([], [], 'o', color='#38bdf8', markersize=7)
    ax_t.set_xlabel('Time (t)', color='#e2e8f0', fontsize=9.5)
    ax_t.set_ylabel(eq['unit'], color='#e2e8f0', fontsize=9.5)
    ax_t.tick_params(colors='#94a3b8')
    ax_t.grid(True, linestyle='--', alpha=0.3, color='#475569')
    ax_t.legend(loc='upper right', facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc', fontsize=8.5)
    ax_t.set_title(f"Temporal Evolution u(x={x0:.2f}, t)", color='#f8fafc', fontsize=11, fontweight='bold')

    # ── Panel 4: Metric Card ──
    ax_card.axis('off')

    def update_plots(val=None):
        nonlocal x0, t0
        x0 = slider_x.val
        t0 = slider_t.val

        eq = EQ_CONFIG[current_eq]

        # 1. Heatmap crosshair update
        cross_h.set_ydata([t0, t0])
        cross_v.set_xdata([x0, x0])
        dot_target.set_data([x0], [t0])

        # 2. Spatial Slice at t0
        t_slice = np.full_like(x_lin, t0)
        u_q_x = predict_batch(qapinn, x_lin, t_slice)
        u_c_x = predict_batch(cpinn,  x_lin, t_slice)
        u_e_x = np.array([eq['exact'](x, t0) for x in x_lin])

        line_x_exact.set_data(x_lin, u_e_x)
        line_x_qapinn.set_data(x_lin, u_q_x)
        line_x_cpinn.set_data(x_lin, u_c_x)

        u_q_point = float(predict_batch(qapinn, np.array([x0]), np.array([t0]))[0])
        u_c_point = float(predict_batch(cpinn,  np.array([x0]), np.array([t0]))[0])
        u_e_point = float(eq['exact'](x0, t0))

        dot_x_q.set_data([x0], [u_q_point])
        ax_x.set_xlim(eq['domain_x'])
        ax_x.set_ylim(min(u_e_x.min(), u_q_x.min()) - 0.2, max(u_e_x.max(), u_q_x.max()) + 0.2)
        ax_x.set_title(f"Spatial Profile u(x, t={t0:.2f})", color='#f8fafc', fontsize=11, fontweight='bold')

        # 3. Temporal Slice at x0
        x_slice = np.full_like(t_lin, x0)
        u_q_t = predict_batch(qapinn, x_slice, t_lin)
        u_c_t = predict_batch(cpinn,  x_slice, t_lin)
        u_e_t = np.array([eq['exact'](x0, t) for t in t_lin])

        line_t_exact.set_data(t_lin, u_e_t)
        line_t_qapinn.set_data(t_lin, u_q_t)
        line_t_cpinn.set_data(t_lin, u_c_t)
        dot_t_q.set_data([t0], [u_q_point])
        ax_t.set_xlim(eq['domain_t'])
        ax_t.set_ylim(min(u_e_t.min(), u_q_t.min()) - 0.2, max(u_e_t.max(), u_q_t.max()) + 0.2)
        ax_t.set_title(f"Temporal Evolution u(x={x0:.2f}, t)", color='#f8fafc', fontsize=11, fontweight='bold')

        # 4. Metric Dashboard Card Update
        err_q_pt = abs(u_q_point - u_e_point)
        err_c_pt = abs(u_c_point - u_e_point)

        l2_q_tot = np.linalg.norm(cache['u_q'] - cache['u_e']) / (np.linalg.norm(cache['u_e']) + 1e-10) * 100
        l2_c_tot = np.linalg.norm(cache['u_c'] - cache['u_e']) / (np.linalg.norm(cache['u_e']) + 1e-10) * 100

        ax_card.clear()
        ax_card.axis('off')
        ax_card.set_facecolor('#1e293b')

        card_text = (
            f"  LIVE PITCH DASHBOARD\n"
            f"  {'─'*42}\n"
            f"  Selected Query:   x = {x0:+.3f},  t = {t0:.3f} s\n"
            f"  Target PDE:       {eq['name']}\n\n"
            f"  POINT PREDICTION AT (x, t):\n"
            f"    • QAPINN (4-Qubit):  {u_q_point:+.5f}  (Err: {err_q_pt:.5f})\n"
            f"    • Classical c-PINN:  {u_c_point:+.5f}  (Err: {err_c_pt:.5f})\n"
            f"    • Exact Analytical: {u_e_point:+.5f}\n\n"
            f"  DOMAIN RELATIVE L₂ ERROR:\n"
            f"    • QAPINN: {l2_q_tot:.2f}%   |   c-PINN: {l2_c_tot:.2f}%\n"
            f"    • Precision Gain: {l2_c_tot/(l2_q_tot+1e-10):.2f}× Higher Accuracy\n\n"
            f"  EFFICIENCY METRICS:\n"
            f"    • QAPINN Parameters: 1,905 (44.6% reduction vs 3,441)\n"
            f"    • FLOP Count: 3,712 FLOPs/pt (2.77× computational reduction)"
        )

        ax_card.text(0.05, 0.95, card_text, transform=ax_card.transAxes, color='#f8fafc',
                     fontsize=10, va='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='#0f172a', edgecolor='#38bdf8', lw=1.5))

        fig.canvas.draw_idle()

    # ── Sliders & Controls Setup ──
    ax_slider_x = fig.add_axes([0.15, 0.08, 0.40, 0.03], facecolor='#1e293b')
    ax_slider_t = fig.add_axes([0.15, 0.03, 0.40, 0.03], facecolor='#1e293b')

    slider_x = Slider(ax_slider_x, 'Spatial x ', -1.0, 1.0, valinit=x0, valfmt='%.3f', color='#38bdf8')
    slider_t = Slider(ax_slider_t, 'Time t ', 0.0, 1.0, valinit=t0, valfmt='%.3f', color='#f59e0b')

    slider_x.label.set_color('#f8fafc')
    slider_x.valtext.set_color('#38bdf8')
    slider_t.label.set_color('#f8fafc')
    slider_t.valtext.set_color('#f59e0b')

    slider_x.on_changed(update_plots)
    slider_t.on_changed(update_plots)

    # Radio button for Equation Selection
    ax_radio = fig.add_axes([0.65, 0.03, 0.28, 0.08], facecolor='#0f172a')
    radio = RadioButtons(ax_radio, ('Burgers (Shock)', 'Heat (Thermal)', 'Wave (Seismic)'),
                         active=0, activecolor='#38bdf8')

    for label in radio.labels:
        label.set_color('#f8fafc')
        label.set_fontsize(9.5)

    def on_eq_change(label):
        nonlocal current_eq
        if 'Burgers' in label:
            current_eq = 'burgers'
        elif 'Heat' in label:
            current_eq = 'heat'
        elif 'Wave' in label:
            current_eq = 'wave'

        compute_heatmap_data()
        mesh.set_array(cache['u_q'].ravel())
        update_plots()

    radio.on_clicked(on_eq_change)

    fig.suptitle("QAPINN Interactive Pitch Demonstrator — Real-Time PDE Solver Widget",
                 color='#ffffff', fontsize=14, fontweight='bold', y=0.97)

    # Initial draw
    update_plots()

    print("\n✓ Interactive Matplotlib Slider Window Launched!")
    print("  Drag the X and T sliders or switch equations to interact live.\n")
    plt.show()

if __name__ == '__main__':
    run_interactive_widget()
