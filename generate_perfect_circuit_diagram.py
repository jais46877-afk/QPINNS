import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def draw_vqc_circuit(output_paths, style='dark'):
    """
    Renders a publication-grade, mathematically exact Qiskit-style circuit diagram 
    for the 4-Qubit 2-Layer VQC with true Circular CNOT Ring Entanglement.
    """
    fig, ax = plt.subplots(figsize=(19, 7.5), dpi=300)
    
    # Palette definition
    if style == 'dark':
        bg_color = '#0b111a'
        text_color = '#ffffff'
        wire_color = '#cbd5e1'
        subtext_color = '#94a3b8'
        enc_box = '#2563eb'      # Blue
        rot_box = '#10b981'      # Emerald Green
        cnot_color = '#f43f5e'   # Coral Red
        meas_box = '#475569'     # Slate Gray
        section_bg = '#131e2e'
        annot_color = '#f59e0b'
    else:
        bg_color = '#ffffff'
        text_color = '#0f172a'
        wire_color = '#334155'
        subtext_color = '#64748b'
        enc_box = '#1d4ed8'
        rot_box = '#059669'
        cnot_color = '#e11d48'
        meas_box = '#64748b'
        section_bg = '#f8fafc'
        annot_color = '#b45309'

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    n_qubits = 4
    y_positions = [3, 2, 1, 0] # q0 at top (3), q3 at bottom (0)
    
    # Draw Qubit wires & labels
    x_start, x_end = 0.5, 18.0
    for i, y in enumerate(y_positions):
        ax.plot([x_start, x_end], [y, y], color=wire_color, lw=1.6, zorder=1)
        ax.text(x_start - 0.2, y, f'$q_{i}$', color=text_color, fontsize=15, 
                ha='right', va='center', fontweight='bold', fontfamily='sans-serif')
        ax.text(x_start + 0.1, y + 0.18, r'$|0\rangle$', color=subtext_color, fontsize=11, ha='left')

    def draw_gate(x, y, w, h, label, sublabel=None, color='#2563eb'):
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                      boxstyle="round,pad=0.03,rounding_size=0.08",
                                      facecolor=color, edgecolor='white', lw=1.2, zorder=3)
        ax.add_patch(rect)
        if sublabel:
            ax.text(x, y + 0.08, label, color='white', fontsize=11, fontweight='bold', ha='center', va='center')
            ax.text(x, y - 0.14, sublabel, color='white', fontsize=8.5, ha='center', va='center')
        else:
            ax.text(x, y, label, color='white', fontsize=11, fontweight='bold', ha='center', va='center')

    def draw_cnot(x, ctrl_y, tgt_y):
        # Line between control and target
        ax.plot([x, x], [ctrl_y, tgt_y], color=cnot_color, lw=2.2, zorder=2)
        # Control dot
        ax.plot(x, ctrl_y, marker='o', markersize=9, color=cnot_color, zorder=4)
        # Target circle with plus
        target_circle = patches.Circle((x, tgt_y), radius=0.18, facecolor=bg_color, 
                                       edgecolor=cnot_color, lw=2.2, zorder=3)
        ax.add_patch(target_circle)
        ax.plot([x - 0.18, x + 0.18], [tgt_y, tgt_y], color=cnot_color, lw=2.0, zorder=4)
        ax.plot([x, x], [tgt_y - 0.18, tgt_y + 0.18], color=cnot_color, lw=2.0, zorder=4)

    # ── STAGE 1: ENCODING LAYER ──
    x_enc = 1.6
    draw_gate(x_enc, y_positions[0], 0.9, 0.45, r"$R_x$", r"$x \cdot \pi$", enc_box)
    draw_gate(x_enc, y_positions[1], 0.9, 0.45, r"$R_y$", r"$t \cdot \pi$", enc_box)
    draw_gate(x_enc, y_positions[2], 0.9, 0.45, r"$R_x$", r"$x \cdot \pi$", enc_box)
    draw_gate(x_enc, y_positions[3], 0.9, 0.45, r"$R_y$", r"$t \cdot \pi$", enc_box)
    
    # ── STAGE 2: VARIATIONAL LAYER 1 ──
    # Rotations (Rx, Ry, Rz)
    x_rot1 = 3.0
    for q_idx, y in enumerate(y_positions):
        draw_gate(x_rot1,       y, 0.72, 0.42, r"$R_x$", r"$\theta_{1,%d,1}$" % q_idx, rot_box)
        draw_gate(x_rot1 + 0.8, y, 0.72, 0.42, r"$R_y$", r"$\theta_{1,%d,2}$" % q_idx, rot_box)
        draw_gate(x_rot1 + 1.6, y, 0.72, 0.42, r"$R_z$", r"$\theta_{1,%d,3}$" % q_idx, rot_box)
        
    # Circular CNOT Ring for Layer 1
    # Step 1: q0 -> q1
    draw_cnot(5.9, y_positions[0], y_positions[1])
    # Step 2: q1 -> q2
    draw_cnot(6.5, y_positions[1], y_positions[2])
    # Step 3: q2 -> q3
    draw_cnot(7.1, y_positions[2], y_positions[3])
    # Step 4: q3 -> q0 (Circular wrap-around connection)
    draw_cnot(7.7, y_positions[3], y_positions[0])

    # ── STAGE 3: VARIATIONAL LAYER 2 ──
    x_rot2 = 9.2
    for q_idx, y in enumerate(y_positions):
        draw_gate(x_rot2,       y, 0.72, 0.42, r"$R_x$", r"$\theta_{2,%d,1}$" % q_idx, rot_box)
        draw_gate(x_rot2 + 0.8, y, 0.72, 0.42, r"$R_y$", r"$\theta_{2,%d,2}$" % q_idx, rot_box)
        draw_gate(x_rot2 + 1.6, y, 0.72, 0.42, r"$R_z$", r"$\theta_{2,%d,3}$" % q_idx, rot_box)
        
    # Circular CNOT Ring for Layer 2
    draw_cnot(12.1, y_positions[0], y_positions[1])
    draw_cnot(12.7, y_positions[1], y_positions[2])
    draw_cnot(13.3, y_positions[2], y_positions[3])
    draw_cnot(13.9, y_positions[3], y_positions[0])

    # ── STAGE 4: MEASUREMENT ──
    x_meas = 15.4
    for q_idx, y in enumerate(y_positions):
        rect = patches.FancyBboxPatch((x_meas - 0.4, y - 0.25), 0.8, 0.5,
                                      boxstyle="round,pad=0.03,rounding_size=0.08",
                                      facecolor=meas_box, edgecolor='white', lw=1.2, zorder=3)
        ax.add_patch(rect)
        # Meter icon
        arc = patches.Arc((x_meas, y - 0.05), 0.4, 0.3, angle=0, theta1=0, theta2=180, color='white', lw=1.4, zorder=4)
        ax.add_patch(arc)
        ax.plot([x_meas, x_meas + 0.12], [y - 0.05, y + 0.14], color='white', lw=1.4, zorder=4)
        ax.text(x_meas, y - 0.16, r"$\langle Z \rangle$", color='white', fontsize=8.5, ha='center', va='center')
        
        # Classical output line & labels
        ax.plot([x_meas + 0.4, x_end], [y, y], color=wire_color, lw=1.2, linestyle='--', zorder=1)
        ax.text(x_end + 0.2, y, f"$y_{q_idx} \in [-1, 1]$", color=text_color, fontsize=12, 
                ha='left', va='center', fontweight='bold')

    # ── Section Division Headers ──
    divisions = [
        (1.6, "INPUT ANGLE ENCODING", "$U_{enc}(x, t)$", enc_box),
        (5.5, "VARIATIONAL LAYER 1", "$U_{var}(\\theta_1) + U_{ent}^{circular}$", rot_box),
        (11.7, "VARIATIONAL LAYER 2", "$U_{var}(\\theta_2) + U_{ent}^{circular}$", rot_box),
        (15.4, "MEASUREMENT", r"Pauli-$Z$ Readout $\langle \sigma_z^{(k)} \rangle$", meas_box)
    ]
    
    for x_center, title, sub, col in divisions:
        ax.text(x_center, 3.8, title, color=col, fontsize=11, fontweight='bold', ha='center', va='center')
        ax.text(x_center, 3.55, sub, color=subtext_color, fontsize=9.5, ha='center', va='center')
        
    # Vertical stage separators (dashed)
    for x_sep in [2.35, 8.45, 14.65]:
        ax.plot([x_sep, x_sep], [-0.5, 3.7], color=wire_color, linestyle=':', alpha=0.4, lw=1.2)

    # Title & Legend
    ax.text(9.2, 4.35, "VQC Architecture for Hybrid QAPINN — 4 Qubits, 2 Layers, Circular CNOT Entanglement",
            color=text_color, fontsize=16, fontweight='bold', ha='center', va='center')
    
    # Legend at bottom
    ax.text(9.2, -0.85, 
            r"$\bullet$ Angle Encoding: $R_x(x\pi), R_y(t\pi)$    " +
            r"$\bullet$ Trainable Variational Weights: 24 params ($R_x, R_y, R_z$)    " +
            r"$\bullet$ Circular CNOT Ring: $(0\to 1), (1\to 2), (2\to 3), (3\to 0)$    " +
            r"$\bullet$ Readout: 4 Expectation Values",
            color=annot_color, fontsize=10.5, ha='center', va='center', fontweight='bold')

    ax.set_xlim(x_start - 1.2, x_end + 2.5)
    ax.set_ylim(-1.2, 4.8)
    ax.axis('off')
    plt.tight_layout()
    
    for p in output_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, facecolor=bg_color, edgecolor='none', bbox_inches='tight', dpi=300)
        print(f"Saved: {p}")
        
    plt.close()

if __name__ == '__main__':
    targets = [
        r'C:\Users\HP\Desktop\Wiser_Projects\evaluation\plots\vqc_circuit_4qubit.png',
        r'C:\Users\HP\Desktop\Wiser_Projects\evaluation\plots\vqc_circuit_qiskit_style.png',
        r'C:\Users\HP\.gemini\antigravity\brain\336e8951-5212-4999-a11a-96c1d45c5569\vqc_circuit_4qubit.png',
        r'C:\Users\HP\.gemini\antigravity\brain\336e8951-5212-4999-a11a-96c1d45c5569\vqc_circuit_qiskit_style.png'
    ]
    draw_vqc_circuit(targets, style='dark')
