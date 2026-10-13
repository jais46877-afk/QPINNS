# 📚 Primary Sources, References & Materials

> **Project:** Quantum-Assisted Physics-Informed Neural Networks (QAPINNs) for Computational Fluid Dynamics  
> **Target Challenge:** WISER Global Quantum+AI Program 2026 (in collaboration with **BosonQ Psi — BQP**)  
> **GitHub Repository Link:** [https://github.com/Mani-kanta-0539/CPINNvsQAPINN/blob/main/REFERENCES.md](https://github.com/Mani-kanta-0539/CPINNvsQAPINN/blob/main/REFERENCES.md)  
> 🔗 **Project Resources (Presentation PPT & Deliverable PDF):** [Google Drive Folder](https://drive.google.com/drive/folders/1oWGC-IF-ohguYIJDN4WMlQSO-oLeg-1t?usp=sharing)

---

## 📄 1. Primary Academic Research Papers

1. **[Core QAPINN Benchmark Paper]**  
   J. Shah, R. Lineswala, and A. Chopra, *"Benchmarking quantum-assisted PINN (QA-PINN) for computational fluid dynamics,"* in *2024 IEEE International Conference on Quantum Computing and Engineering (QCE)*, vol. 01, pp. 1707–1712, 2024.  
   *Relevance:* Introduced the hybrid QA-PINN architecture replacing the first classical hidden layer with a Variational Quantum Circuit (VQC).

2. **[Foundational Physics-Informed Neural Networks]**  
   M. Raissi, P. Perdikaris, and G. E. Karniadakis, *"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,"* *Journal of Computational Physics*, vol. 378, pp. 686–707, 2019.  
   *Relevance:* Establishes the classical c-PINN baseline, spatial-temporal automatic differentiation, and composite PDE residual loss formulation ($L_f + L_i + L_b$).

3. **[Expressivity & Quantum Data Encoding]**  
   S. Maria et al., *"The effect of data encoding on the expressive power of variational quantum machine learning models,"* *Physical Review A*, vol. 103, no. 3, p. 032430, 2021.  
   *Relevance:* Theoretical basis for Fourier feature expansion via rotational gate projections in quantum Hilbert space ($\mathcal{H} \cong \mathbb{C}^{16}$).

4. **[Barren Plateaus in Variational Quantum Circuits]**  
   J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven, *"Barren plateaus in quantum neural network training landscapes,"* *Nature Communications*, vol. 9, no. 1, p. 4812, 2018.  
   *Relevance:* Explains the exponential gradient variance decay ($\text{Var}(\partial_\theta L) < 10^{-6}$) observed at qubit counts $N \ge 5$.

5. **[Domain Decomposition & Generalization]**  
   Z. Hu, A. D. Jagtap, G. E. Karniadakis, and K. Kawaguchi, *"When do extended physics-informed neural networks (XPINNs) improve generalization?"* *SIAM Journal on Scientific Computing*, vol. 44, no. 5, pp. A3158–A3182, 2022.  
   *Relevance:* Generalization bounds and sub-domain decomposition comparators for SciML solvers.

6. **[Differentiable Quantum Circuits for Differential Equations]**  
   O. Kyriienko, A. E. Paine, and V. E. Elfving, *"Solving nonlinear differential equations with differentiable quantum circuits,"* *Physical Review A*, vol. 103, no. 5, p. 052416, 2021.  
   *Relevance:* Formulates quantum automatic differentiation and parameter-shift rule gradients for PDE loss functions.

7. **[Variational Quantum Algorithms Survey]**  
   M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, ... & P. J. Coles, *"Variational quantum algorithms,"* *Nature Reviews Physics*, vol. 3, no. 9, pp. 625–644, 2021.  
   *Relevance:* Survey of NISQ-era optimization landscapes, ansatz design, and entanglement topologies.

---

## 🏢 2. Industry Documentation & Simulation Suites

1. **[Industrial & CFD Integration]**  
   BosonQ Psi (BQP), *"BQPhy® Quantum-Assisted Engineering Solvers & Simulation Suite,"* Technical Whitepaper & BQPhy Engine Documentation, 2025–2026.  
   *Relevance:* Industrial application context for parameter-efficient solvers in aerospace fluid dynamics and structural optimization.

---

## 🧮 3. Analytical Ground Truth Solvers & Datasets

1. **[1D Viscous Burgers' Cole-Hopf Analytical Ground Truth]**  
   Ground truth solution computed via exact Cole-Hopf transformation numerical integration:  
   $$u(x,t) = -2\nu \frac{\phi_x}{\phi}$$  
   Evaluating non-linear advection-diffusion shock wave formation at viscosity scales $\nu \in \{0.1, 0.01, 0.001\}$.

2. **[1D Heat Equation Parabolic Analytical Solution]**  
   Exact analytical solution:  
   $$u_{\text{exact}}(x,t) = \sin(\pi x) \exp(-\alpha \pi^2 t), \quad \alpha = 0.05$$

3. **[1D Wave Equation Hyperbolic Analytical Solution]**  
   Exact 2nd-order hyperbolic wave solution:  
   $$u_{\text{exact}}(x,t) = \sin(\pi x) \cos(\pi c t), \quad c = 1.0$$

---

## 🛠️ 4. Software Frameworks & Technical Libraries

* **PennyLane (Xanadu)**: Differentiable Quantum Machine Learning Platform (`pennylane.ai`).
* **PyTorch (Meta AI)**: Scientific Machine Learning Engine & Automatic Differentiation (`pytorch.org`).
* **Qiskit (IBM Quantum)**: Quantum Circuit Construction & Visualization (`qiskit.org`).
* **SciPy & NumPy**: Numerical Integration & Linear Algebra.
* **Matplotlib**: High-Resolution XAI Feature Map & Solution Contour Visualization.
