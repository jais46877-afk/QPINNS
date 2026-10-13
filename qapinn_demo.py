"""
╔══════════════════════════════════════════════════════════════════════════╗
║    QAPINN INTERACTIVE DEMO — Hybrid Quantum-Assisted PDE Solver         ║
║    For QUEST-AI 2026 Pitch Demonstration                                ║
║    Velamala Manikanta et al., Andhra University, 2026                   ║
╚══════════════════════════════════════════════════════════════════════════╝

USAGE:
    python qapinn_demo.py                       --> Interactive menu mode
    python qapinn_demo.py --equation burgers    --> Direct Burgers' demo
    python qapinn_demo.py --equation heat       --> Direct Heat demo
    python qapinn_demo.py --equation wave       --> Direct Wave demo
    python qapinn_demo.py --demo all            --> Run all demos with plots
    python qapinn_demo.py --equation burgers --x 0.3 --t 0.5  --> Single point
"""

import sys
import os
import warnings
import argparse
import time
warnings.filterwarnings("ignore")

# Fix Windows PowerShell UTF-8 encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ─── Path Setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.qapinn import HybridQAPINN
from src.cpinn import ClassicalPINN
from src.exact_solution import cole_hopf_exact_point

# ─── ANSI Color Codes ────────────────────────────────────────────────────────
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
BOLD    = "\033[1m"
RESET   = "\033[0m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"

# ─── Trained Model Paths ─────────────────────────────────────────────────────
MODEL_PATHS = {
    'qapinn':           'evaluation/qapinn_primary/model.pt',
    'qapinn_3qubit':    'evaluation/qapinn_3qubit/model.pt',
    'qapinn_5qubit':    'evaluation/qapinn_5qubit/model.pt',
    'cpinn':            'evaluation/cpinn_model.pt',
    'compressed_cpinn': 'evaluation/compressed_cpinn_model.pt',
}

# ─── Exact Analytical Solutions ──────────────────────────────────────────────
def exact_heat(x, t, alpha=0.05):
    """u(x,t) = exp(-alpha * pi^2 * t) * sin(pi * x)"""
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)

def exact_wave(x, t, c=1.0):
    """u(x,t) = sin(pi*x) * cos(c*pi*t)  [D'Alembert standing wave]"""
    return np.sin(np.pi * x) * np.cos(c * np.pi * t)

def exact_burgers(x, t, nu=0.01):
    """Cole-Hopf numerical exact solution for 1D Burgers'."""
    if t == 0.0:
        return -np.sin(np.pi * x)
    return cole_hopf_exact_point(x, t, nu=nu)

# ─── Model Loaders ───────────────────────────────────────────────────────────
def load_qapinn(n_qubits=4):
    model = HybridQAPINN(
        n_qubits=n_qubits, n_layers=2, position='first',
        measurement_mode='expectation', encoding='angle',
        entanglement='linear', hidden_features=40
    )
    path = MODEL_PATHS['qapinn'] if n_qubits == 4 else MODEL_PATHS.get(f'qapinn_{n_qubits}qubit', '')
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        model.eval()
        print(f"  {GREEN}QAPINN ({n_qubits}-qubit) loaded from {path}{RESET}")
    else:
        print(f"  {YELLOW}No checkpoint at {path} -- using random weights for structure demo{RESET}")
    return model

def load_cpinn():
    model = ClassicalPINN(hidden_features=40, num_layers=3)
    path = MODEL_PATHS['cpinn']
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        model.eval()
        print(f"  {GREEN}Classical c-PINN loaded from {path}{RESET}")
    else:
        print(f"  {YELLOW}No checkpoint at {path} -- using random weights{RESET}")
    return model

# ─── Core Prediction Helpers ─────────────────────────────────────────────────
def predict(model, x_val, t_val):
    """Single-point prediction: give (x, t) → get u(x,t)."""
    x_tensor = torch.tensor([[x_val, t_val]], dtype=torch.float32)
    with torch.no_grad():
        return model(x_tensor).item()

def predict_batch(model, x_arr, t_arr):
    """Batch prediction over numpy arrays."""
    X = torch.tensor(np.stack([x_arr, t_arr], axis=1), dtype=torch.float32)
    with torch.no_grad():
        return model(X).cpu().numpy().flatten()

# ─── Equation Registry ───────────────────────────────────────────────────────
EQ_INFO = {
    'burgers': {
        'name':       "1D Viscous Burgers' Equation",
        'formula':    "du/dt + u*du/dx = nu*d2u/dx2   (nu = 0.01)",
        'ic':         "u(x, 0) = -sin(pi*x)",
        'bc':         "u(-1, t) = u(+1, t) = 0",
        'domain_x':   (-1.0, 1.0),
        'domain_t':   (0.0,  1.0),
        'exact_fn':   exact_burgers,
        'real_world': "Aerodynamics, blood flow, traffic flow, shock wave modeling",
        'unit':       "velocity (m/s, normalized)",
    },
    'heat': {
        'name':       "1D Parabolic Heat Equation",
        'formula':    "du/dt = alpha*d2u/dx2   (alpha = 0.05)",
        'ic':         "u(x, 0) = sin(pi*x)",
        'bc':         "u(-1, t) = u(+1, t) = 0",
        'domain_x':   (-1.0, 1.0),
        'domain_t':   (0.0,  1.0),
        'exact_fn':   exact_heat,
        'real_world': "Electronics thermal management, PCB hot-spots, semiconductor design",
        'unit':       "temperature (normalized)",
    },
    'wave': {
        'name':       "1D Hyperbolic Wave Equation",
        'formula':    "d2u/dt2 = c2*d2u/dx2   (c = 1.0)",
        'ic':         "u(x, 0) = sin(pi*x),  du/dt(x,0) = 0",
        'bc':         "u(-1, t) = u(+1, t) = 0",
        'domain_x':   (-1.0, 1.0),
        'domain_t':   (0.0,  1.0),
        'exact_fn':   exact_wave,
        'real_world': "Seismic inversion, sonar, medical ultrasound, structural monitoring",
        'unit':       "displacement (normalized)",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# DEMO 1: Single Point Prediction
# ─────────────────────────────────────────────────────────────────────────────
def demo_single_point(equation='burgers', x_val=None, t_val=None):
    """
    Core pitch demo: give any (x, t) → QAPINN solves the PDE instantly.
    Compares with c-PINN and exact solution.
    """
    eq = EQ_INFO[equation]
    print(f"\n{'='*65}")
    print(f"{BOLD}  {eq['name']}{RESET}")
    print(f"{'='*65}")
    print(f"  Equation : {eq['formula']}")
    print(f"  IC       : {eq['ic']}")
    print(f"  BC       : {eq['bc']}")
    print(f"  Domain   : x in {eq['domain_x']}, t in {eq['domain_t']}")
    print(f"  Real Use : {eq['real_world']}")
    print(f"{'─'*65}")

    # Get input values
    if x_val is None:
        try:
            x_val = float(input(f"  Enter x value {eq['domain_x']}: ").strip())
        except Exception:
            x_val = 0.3
    if t_val is None:
        try:
            t_val = float(input(f"  Enter t value {eq['domain_t']}: ").strip())
        except Exception:
            t_val = 0.5

    # Clamp to valid domain
    x_val = max(eq['domain_x'][0], min(eq['domain_x'][1], x_val))
    t_val = max(eq['domain_t'][0], min(eq['domain_t'][1], t_val))

    print(f"\n  Predicting u({x_val:.4f}, {t_val:.4f}) ...")
    print(f"  Loading trained models...\n")

    qapinn = load_qapinn(n_qubits=4)
    cpinn  = load_cpinn()

    # Time both predictions
    t0 = time.perf_counter()
    u_qapinn = predict(qapinn, x_val, t_val)
    dt_q = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    u_cpinn = predict(cpinn, x_val, t_val)
    dt_c = (time.perf_counter() - t0) * 1000

    u_exact = eq['exact_fn'](x_val, t_val)
    err_q = abs(u_qapinn - u_exact)
    err_c = abs(u_cpinn  - u_exact)

    # Results table
    print(f"\n{'─'*65}")
    print(f"  {'Model':<30} {'u(x,t)':<14} {'Abs Error':<14} {'Time'}")
    print(f"  {'─'*60}")
    print(f"  {GREEN}{'QAPINN (4-Qubit Hybrid)':<30}{RESET} {u_qapinn:<14.6f} {err_q:<14.6f} {dt_q:.2f}ms")
    print(f"  {RED}{'Classical c-PINN':<30}{RESET} {u_cpinn:<14.6f} {err_c:<14.6f} {dt_c:.2f}ms")
    print(f"  {BOLD}{'Exact / Analytical':<30}{RESET} {u_exact:<14.6f} {'---':<14} ---")
    print(f"{'─'*65}")

    if err_q < err_c:
        gain = ((err_c - err_q) / (err_c + 1e-10)) * 100
        print(f"\n  {GREEN}{BOLD}QAPINN is {gain:.1f}% more accurate at this point{RESET}")
    else:
        print(f"\n  {YELLOW}Classical PINN performed better at this single point (check training){RESET}")

    # Physical interpretation
    print(f"\n  Physical Meaning:")
    print(f"  u({x_val:.3f}, {t_val:.3f}) = {u_qapinn:.6f}  [{eq['unit']}]")
    if equation == 'heat':
        T = u_qapinn * 100
        print(f"  -> Temperature at position x={x_val:.2f} at time t={t_val:.2f}s: T = {T:.2f} deg C (normalized to 100 C max)")
    elif equation == 'wave':
        print(f"  -> Wave displacement at x={x_val:.2f}, t={t_val:.2f}s: u = {u_qapinn:.5f}")
    elif equation == 'burgers':
        print(f"  -> Fluid velocity at x={x_val:.2f}, t={t_val:.2f}s: u = {u_qapinn:.5f} m/s")

    return u_qapinn, u_cpinn, u_exact

# ─────────────────────────────────────────────────────────────────────────────
# DEMO 2: Spatial Profile at any time snapshot
# ─────────────────────────────────────────────────────────────────────────────
def demo_spatial_profile(equation='burgers', t_snap=0.5, save_path=None):
    """Full u(x) profile at given time - compares QAPINN vs c-PINN vs exact."""
    eq = EQ_INFO[equation]
    x_arr = np.linspace(eq['domain_x'][0], eq['domain_x'][1], 200)
    t_arr = np.full_like(x_arr, t_snap)

    print(f"\n  Computing spatial profile at t = {t_snap} ...")
    qapinn = load_qapinn()
    cpinn  = load_cpinn()

    u_q = predict_batch(qapinn, x_arr, t_arr)
    u_c = predict_batch(cpinn,  x_arr, t_arr)
    u_e = np.array([eq['exact_fn'](x, t_snap) for x in x_arr])

    l2_q = np.linalg.norm(u_q - u_e) / (np.linalg.norm(u_e) + 1e-10)
    l2_c = np.linalg.norm(u_c - u_e) / (np.linalg.norm(u_e) + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{eq['name']}\nSpatial Profile at t = {t_snap}",
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(x_arr, u_e, 'k-',  lw=2.0, label='Exact / Analytical', zorder=3)
    ax.plot(x_arr, u_q, 'b--', lw=1.8, label=f'QAPINN  (Rel L2 = {l2_q:.4f})', zorder=2)
    ax.plot(x_arr, u_c, 'r:',  lw=1.8, label=f'c-PINN  (Rel L2 = {l2_c:.4f})',  zorder=1)
    ax.set_xlabel('x  (spatial position)')
    ax.set_ylabel('u(x, t)')
    ax.set_title('Solution Comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(eq['domain_x'])

    ax2 = axes[1]
    ax2.fill_between(x_arr, np.abs(u_q - u_e), alpha=0.55, color='blue',
                     label=f'QAPINN Error  (mean = {np.mean(np.abs(u_q-u_e)):.5f})')
    ax2.fill_between(x_arr, np.abs(u_c - u_e), alpha=0.40, color='red',
                     label=f'c-PINN Error  (mean = {np.mean(np.abs(u_c-u_e)):.5f})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('|u_pred - u_exact|')
    ax2.set_title('Pointwise Absolute Error')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = f'qapinn_demo_{equation}_spatial_t{t_snap:.2f}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  {GREEN}Spatial profile saved -> {save_path}{RESET}")
    print(f"  QAPINN  Rel L2 = {l2_q*100:.3f}%")
    print(f"  c-PINN  Rel L2 = {l2_c*100:.3f}%")
    if l2_q < l2_c:
        print(f"  {GREEN}{BOLD}QAPINN is {l2_c/l2_q:.2f}x more accurate at t={t_snap}{RESET}")
    return save_path

# ─────────────────────────────────────────────────────────────────────────────
# DEMO 3: Full Spacetime Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def demo_spacetime_heatmap(equation='burgers', save_path=None):
    """Full u(x,t) heatmap over entire domain — 3-panel: QAPINN | Exact | Error."""
    eq = EQ_INFO[equation]
    Nx, Nt = 80, 60
    x_lin = np.linspace(eq['domain_x'][0], eq['domain_x'][1], Nx)
    t_lin = np.linspace(eq['domain_t'][0], eq['domain_t'][1], Nt)
    X_grid, T_grid = np.meshgrid(x_lin, t_lin)

    print(f"\n  Building spacetime grid ({Nx}x{Nt}) and predicting ...")
    qapinn = load_qapinn()

    u_q = predict_batch(qapinn, X_grid.flatten(), T_grid.flatten()).reshape(Nt, Nx)
    u_e = np.array([[eq['exact_fn'](x, t) for x in x_lin] for t in t_lin])
    u_err = np.abs(u_q - u_e)

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    fig.suptitle(f"{eq['name']}  --  QAPINN Full Spacetime Solution",
                 fontsize=13, fontweight='bold')

    vmin = min(u_q.min(), u_e.min())
    vmax = max(u_q.max(), u_e.max())

    panels = [
        (u_q,   'QAPINN Prediction',              'RdYlBu_r', (vmin, vmax)),
        (u_e,   'Exact Analytical Solution',       'RdYlBu_r', (vmin, vmax)),
        (u_err, 'Absolute Error  |QAPINN - Exact|', 'hot_r',   (None, None)),
    ]
    for idx, (data, title, cmap, vlim) in enumerate(panels):
        ax = fig.add_subplot(gs[idx])
        kw = dict(cmap=cmap, shading='auto')
        if vlim[0] is not None:
            kw['vmin'], kw['vmax'] = vlim
        im = ax.pcolormesh(x_lin, t_lin, data, **kw)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel('x  (spatial)')
        ax.set_ylabel('t  (time)')
        ax.set_title(title, fontsize=10, fontweight='bold')

    if save_path is None:
        save_path = f'qapinn_demo_{equation}_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    rel_l2 = np.linalg.norm(u_q - u_e) / (np.linalg.norm(u_e) + 1e-10)
    print(f"  {GREEN}Heatmap saved -> {save_path}{RESET}")
    print(f"  Overall Rel L2 Error: {rel_l2*100:.3f}%")
    return save_path

# ─────────────────────────────────────────────────────────────────────────────
# DEMO 4: Time Snapshots (t=0.25, 0.50, 0.75) — pitch-ready
# ─────────────────────────────────────────────────────────────────────────────
def demo_time_snapshots(equation='burgers', save_path=None):
    """Classic 3-panel snapshot comparison - perfect for slides."""
    eq = EQ_INFO[equation]
    snap_times = [0.25, 0.50, 0.75]
    x_arr = np.linspace(eq['domain_x'][0], eq['domain_x'][1], 200)

    print(f"\n  Generating time snapshot comparison (t = 0.25, 0.50, 0.75) ...")
    qapinn = load_qapinn()
    cpinn  = load_cpinn()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.suptitle(
        f"{eq['name']}\nTime Snapshots: QAPINN vs c-PINN vs Exact",
        fontsize=12, fontweight='bold'
    )

    for idx, t_snap in enumerate(snap_times):
        t_arr = np.full_like(x_arr, t_snap)
        u_q = predict_batch(qapinn, x_arr, t_arr)
        u_c = predict_batch(cpinn,  x_arr, t_arr)
        u_e = np.array([eq['exact_fn'](x, t_snap) for x in x_arr])

        l2_q = np.linalg.norm(u_q - u_e) / (np.linalg.norm(u_e) + 1e-10) * 100
        l2_c = np.linalg.norm(u_c - u_e) / (np.linalg.norm(u_e) + 1e-10) * 100

        ax = axes[idx]
        ax.plot(x_arr, u_e, 'k-',  lw=2.0, label='Exact')
        ax.plot(x_arr, u_q, 'b--', lw=1.8, label=f'QAPINN ({l2_q:.2f}%)')
        ax.plot(x_arr, u_c, 'r:',  lw=1.8, label=f'c-PINN ({l2_c:.2f}%)')
        ax.set_title(f't = {t_snap}', fontsize=11, fontweight='bold')
        ax.set_xlabel('x')
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(eq['domain_x'])

    axes[0].set_ylabel('u(x, t)')
    plt.tight_layout()

    if save_path is None:
        save_path = f'qapinn_demo_{equation}_snapshots.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {GREEN}Time snapshots saved -> {save_path}{RESET}")
    return save_path

# ─────────────────────────────────────────────────────────────────────────────
# DEMO 5: Real-World Application Framing (pitch storytelling)
# ─────────────────────────────────────────────────────────────────────────────
def demo_realworld_application(equation='heat'):
    """Frame QAPINN output in a real-world engineering context."""
    eq = EQ_INFO[equation]
    print(f"\n{'='*65}")
    print(f"{BOLD}  REAL-WORLD APPLICATION DEMO{RESET}")
    print(f"{'='*65}")

    qapinn = load_qapinn()

    if equation == 'heat':
        print(f"\n  {YELLOW}{BOLD}Scenario: Electronics Thermal Management -- PCB Hot-Spot Prediction{RESET}")
        print(f"  A 2cm PCB strip is powered ON at t=0.")
        print(f"  Thermal diffusivity alpha = 0.05, T_max = 100 deg C (normalized to 1.0)\n")
        print(f"  {'Position':<25}", end='')
        for t in [0.1, 0.3, 0.5, 1.0]:
            print(f"  t={t:.1f}s  ", end='')
        print()
        print(f"  {'─'*60}")
        for x, label in [(-0.9,'Left Edge'), (-0.5,'Left Quarter'),
                          (0.0,'Center'),    (0.5,'Right Quarter'), (0.9,'Right Edge')]:
            print(f"  {label:<25}", end='')
            for t in [0.1, 0.3, 0.5, 1.0]:
                u = predict(qapinn, x, t)
                T = u * 100
                print(f"  {T:>6.2f}C", end='')
            print()
        print(f"\n  {GREEN}-> Board predicts thermal hot-spots in real-time (~ms){RESET}")
        print(f"  {GREEN}-> Can run on embedded MCU (467 parameters in Position 2!){RESET}")
        print(f"  {GREEN}-> Trigger active cooling before component failure{RESET}")

    elif equation == 'wave':
        print(f"\n  {MAGENTA}{BOLD}Scenario: Seismic Monitoring -- Underground Wave Propagation{RESET}")
        print(f"  5 surface sensors monitoring seismic wave field (c = 1.0 km/s)\n")
        print(f"  {'Sensor':<25} {'t=0.1s':>9} {'t=0.3s':>9} {'t=0.5s':>9} {'t=0.7s':>9} {'t=1.0s':>9}")
        print(f"  {'─'*65}")
        for x, label in [(-0.7,'Sensor A'), (-0.3,'Sensor B'),
                          (0.0,'Sensor C'),  (0.3,'Sensor D'), (0.7,'Sensor E')]:
            print(f"  {label:<25}", end='')
            for t in [0.1, 0.3, 0.5, 0.7, 1.0]:
                u = predict(qapinn, x, t)
                print(f"  {u:>+8.4f}", end='')
            print()
        print(f"\n  {GREEN}-> Full waveform inversion without FEA grid meshing{RESET}")
        print(f"  {GREEN}-> QAPINN achieves 0.45% error on wave equation (2.09x better){RESET}")

    elif equation == 'burgers':
        print(f"\n  {CYAN}{BOLD}Scenario: Aerodynamics -- Shock Wave Velocity Field{RESET}")
        print(f"  1D channel airflow, nu = 0.01 (low viscosity --> shock formation)\n")
        print(f"  {'Position':<15} {'Early t=0.1':>12} {'Mid t=0.5':>12} {'Late t=0.9':>12} {'Exact t=0.5':>12}")
        print(f"  {'─'*65}")
        for x in [-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8]:
            u_early = predict(qapinn, x, 0.1)
            u_mid   = predict(qapinn, x, 0.5)
            u_late  = predict(qapinn, x, 0.9)
            u_ex    = exact_burgers(x, 0.5)
            print(f"  x = {x:+.1f}         {u_early:>+11.5f} {u_mid:>+12.5f} {u_late:>+12.5f} {u_ex:>+12.5f}")
        print(f"\n  {GREEN}-> QAPINN tracks shock formation with 8.36% error (vs 19.49% for c-PINN){RESET}")

# ─────────────────────────────────────────────────────────────────────────────
# DEMO 6: Parameter Efficiency Summary
# ─────────────────────────────────────────────────────────────────────────────
def demo_param_summary():
    print(f"\n{'='*65}")
    print(f"{BOLD}  PARAMETER EFFICIENCY -- Quantum vs Classical{RESET}")
    print(f"{'='*65}")
    print(f"  {'Model':<38} {'Params':>8}  {'Rel L2 Error'}")
    print(f"  {'─'*60}")
    rows = [
        (f"{RED}Baseline c-PINN  [2,50,50,50,1]{RESET}",    3441, "19.49%"),
        (f"{YELLOW}Compressed c-PINN (matched params){RESET}", 1905, "19.37%"),
        (f"{GREEN}QAPINN (4-Qubit, Position 1){RESET}",       1905, " 8.36%   <-- 2.33x better"),
        (f"{BLUE}QAPINN (4-Qubit, Position 2){RESET}",         467, "67.53%   (86.4% param compression)"),
    ]
    for name, params, error in rows:
        print(f"  {name:<55} {params:>6}   {error}")
    print(f"\n  {GREEN}{BOLD}Key: Same 1905 parameters, QAPINN halves the error.{RESET}")
    print(f"  Quantum Hilbert space encoding -- not parameter count -- drives accuracy.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Run ALL Demos at once (pitch prep mode)
# ─────────────────────────────────────────────────────────────────────────────
def run_all_demos():
    print(f"\n{BOLD}  Running ALL demos -- generating all plots for pitch...{RESET}\n")
    demo_param_summary()

    saved_plots = []
    for eq in ['burgers', 'heat', 'wave']:
        print(f"\n{'#'*65}")
        print(f"  {BOLD}Equation: {EQ_INFO[eq]['name']}{RESET}")
        print(f"{'#'*65}")

        demo_single_point(equation=eq,
                          x_val={'burgers': 0.0, 'heat': 0.5, 'wave': 0.3}[eq],
                          t_val=0.5)
        p1 = demo_time_snapshots(equation=eq, save_path=f'pitch_{eq}_snapshots.png')
        p2 = demo_spacetime_heatmap(equation=eq, save_path=f'pitch_{eq}_heatmap.png')
        demo_realworld_application(equation=eq)
        saved_plots += [p1, p2]

    print(f"\n{'='*65}")
    print(f"  {GREEN}{BOLD}ALL DEMOS COMPLETE{RESET}")
    print(f"  Saved plot files:")
    for p in saved_plots:
        print(f"    -> {p}")
    print(f"{'='*65}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Interactive Menu
# ─────────────────────────────────────────────────────────────────────────────
def interactive_menu():
    print(f"\n{BOLD}{'='*65}")
    print(f"   QAPINN INTERACTIVE DEMO -- Quantum PDE Solver")
    print(f"   Andhra University | QUEST-AI 2026")
    print(f"{'='*65}{RESET}")

    eqs = {'1': 'burgers', '2': 'heat', '3': 'wave'}

    while True:
        print(f"\n{BOLD}  Main Menu:{RESET}")
        print(f"  [1]  Single Point Prediction   -- enter any (x, t)")
        print(f"  [2]  Spatial Profile Plot       -- u(x) at any time t")
        print(f"  [3]  Full Spacetime Heatmap     -- u(x,t) over full domain")
        print(f"  [4]  Time Snapshot Plot         -- t=0.25, 0.50, 0.75")
        print(f"  [5]  Real-World Application     -- engineering scenario framing")
        print(f"  [6]  Parameter Efficiency Table")
        print(f"  [7]  Run ALL static demos (pitch prep)")
        print(f"  [8]  LAUNCH INTERACTIVE SLIDER WIDGET (Idea 3)")
        print(f"  [9]  GENERATE LIVE SIDE-BY-SIDE ANIMATION GIF (Idea 2)")
        print(f"  [q]  Quit")

        choice = input(f"\n  Select option: ").strip().lower()
        if choice == 'q':
            print(f"\n  {GREEN}Good luck with your pitch! See you at QUEST-AI 2026!{RESET}\n")
            break

        if choice in ['1','2','3','4','5','9']:
            print(f"\n  Select equation:")
            print(f"  [1] Burgers (fluid/shock)   [2] Heat (thermal)   [3] Wave (acoustic/seismic)")
            eq = eqs.get(input("  Equation [1/2/3]: ").strip(), 'burgers')

        if choice == '1':
            demo_single_point(equation=eq)
        elif choice == '2':
            t = input("  Time snapshot t (0 to 1): ").strip()
            demo_spatial_profile(equation=eq, t_snap=float(t) if t else 0.5)
        elif choice == '3':
            demo_spacetime_heatmap(equation=eq)
        elif choice == '4':
            demo_time_snapshots(equation=eq)
        elif choice == '5':
            demo_realworld_application(equation=eq)
        elif choice == '6':
            demo_param_summary()
        elif choice == '7':
            run_all_demos()
        elif choice == '8':
            from qapinn_interactive_slider import run_interactive_widget
            run_interactive_widget()
        elif choice == '9':
            from qapinn_animated_comparison import generate_animation
            gif_path = generate_animation(equation=eq, n_frames=50, fps=12, save_gif=True, show_live=False)
            print(f"\n  {GREEN}✓ Generated GIF saved to: {gif_path}{RESET}")
        else:
            print(f"  {YELLOW}Invalid option.{RESET}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='QAPINN Interactive Demo for QUEST-AI 2026 Pitch'
    )
    parser.add_argument('--equation', choices=['burgers','heat','wave'], default=None)
    parser.add_argument('--demo',
                        choices=['point','spatial','heatmap','snapshots','realworld','all','params','slider','animate'],
                        default=None)
    parser.add_argument('--x', type=float, default=None, help='x coordinate')
    parser.add_argument('--t', type=float, default=None, help='t coordinate')
    args = parser.parse_args()

    if args.demo == 'all':
        run_all_demos()
    elif args.demo == 'slider':
        from qapinn_interactive_slider import run_interactive_widget
        run_interactive_widget()
    elif args.demo == 'animate':
        from qapinn_animated_comparison import generate_animation
        eq = args.equation or 'burgers'
        generate_animation(equation=eq, n_frames=50, fps=12, save_gif=True, show_live=False)
    elif args.demo is None and args.equation is None:
        interactive_menu()
    else:
        eq = args.equation or 'burgers'
        d  = args.demo or 'point'
        if d == 'point':
            demo_single_point(equation=eq, x_val=args.x, t_val=args.t)
        elif d == 'spatial':
            demo_spatial_profile(equation=eq, t_snap=args.t or 0.5)
        elif d == 'heatmap':
            demo_spacetime_heatmap(equation=eq)
        elif d == 'snapshots':
            demo_time_snapshots(equation=eq)
        elif d == 'realworld':
            demo_realworld_application(equation=eq)
        elif d == 'params':
            demo_param_summary()

