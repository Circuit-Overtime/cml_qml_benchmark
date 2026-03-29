# Research Paper Fixes

- [x] 1. Section 3.1 — Rename to "Dataset Description"
- [x] 2. Attach dataset link to the references
- [x] 3. Describe the usage of each data attribute
- [x] 4. Parameters should be inline (not itemized)
- [x] 5. Add theoretical justification for HGBC's performance
- [x] 6. Proper dataset description + feature flow + train-test split explanation
- [x] 7. Move EDA before modelling; state what it is used for and where
- [x] 8. Replace VQLR with VQC in abstract; fix model naming consistency
- [x] 9. Add references in comparison table; normalize dataset names (remove author name from last row)
- [x] 10. Clarify which part of the dataset is used for ML vs EDA

## Quantum Section Fixes

### 11. VQC — Full circuit description + corrected figure
- [x] 11a. Write detailed text: 4 PCA features → angle encoding via RY(π·xᵢ) on each qubit → Rot(θ₁,θ₂,θ₃) variational layer → CNOT ladder (0→1→2→3) for entanglement → measurement of PauliZ on qubit 0 → output mapped to [0,1] via (⟨Z⟩+1)/2
- [x] 11b. Regenerate VQC circuit diagram from PennyLane code (current figure has incorrect black-box U(φ) and mystery R1/R2/R4 gates that don't match the code)
- [x] 11c. Specify: optimizer = Adam(lr=0.1), loss = MSE, epochs = 10, framework = PennyLane

### 12. QSVM — Full circuit description + corrected figure
- [x] 12a. Write detailed text: 4 PCA features → ZZFeatureMap(dim=4, reps=2) encodes data into quantum state — first layer applies H and Rz(xᵢ) on each qubit, second layer applies CNOT + Rz(π·(xᵢ−xⱼ)(xᵢ+xⱼ)) for pairwise entanglement → statevector extracted → fidelity kernel K(i,j)=|⟨ψ(xᵢ)|ψ(xⱼ)⟩|² computed → fed to classical SVC(kernel="precomputed")
- [x] 12b. Regenerate QSVM circuit diagram from Qiskit ZZFeatureMap (current figure shows a variational V(θ) block which is wrong — QSVM has NO trainable parameters, only a data-encoding feature map)
- [x] 12c. Explain the kernel trick: quantum kernel maps to Hilbert space, fidelity measures overlap

### 13. QNB — Full circuit description + corrected figure
- [x] 13a. Recreated QNB implementation (scripts/qnb_quantum.py): RY angle encoding → CNOT ring → RZ phase encoding → PauliZ on all 4 qubits → GaussianNB
- [x] 13b. Wrote full description: 4 stages (angle encoding, CNOT ring entanglement, phase encoding, all-qubit measurement) + classical GNB classification
- [x] 13c. Created draw_qnb.py to regenerate QNB circuit diagram from PennyLane code

### 14. Theoretical descriptions (applies across all 3)
- [x] 14a. Angle encoding theory — covered in VQC §(i): RY(π·xᵢ)|0⟩ equation + Bloch sphere explanation
- [x] 14b. ZZFeatureMap theory — covered in QSVM §(i): H+Rz layer, CNOT-Rz-CNOT entangling, reps=2 expressibility
- [x] 14c. Measurement theory — covered in VQC §(iii): ⟨Z₀⟩ ∈ [−1,1], rescaled (⟨Z⟩+1)/2 ∈ [0,1]

### 15. Regenerate all 3 circuit figures
- [x] 15a. Use PennyLane qml.draw() for VQC circuit → save as figure (scripts/draw_vqc.py)
- [x] 15b. Use Qiskit ZZFeatureMap.draw('mpl') for QSVM circuit → save as figure (scripts/draw_qsvm.py)
- [x] 15c. Created draw_qnb.py for QNB circuit (scripts/draw_qnb.py) — run to generate figure

### 16. Visualizations (also high priority)
- [x] 16. Visualize: raw dataset (§3.1), PCA-reduced (§3.2), quantum angle encoding (§3.4) — all 3 figures added to paper
