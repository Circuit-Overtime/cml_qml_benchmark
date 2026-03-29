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

## Quantum Section Overhaul

### 11. VQC — Full circuit description + corrected figure
- [x] 11a. Write detailed text: 4 PCA features → angle encoding via RY(π·xᵢ) on each qubit → Rot(θ₁,θ₂,θ₃) variational layer → CNOT ladder (0→1→2→3) for entanglement → measurement of PauliZ on qubit 0 → output mapped to [0,1] via (⟨Z⟩+1)/2
- [x] 11b. Regenerate VQC circuit diagram from PennyLane code (current figure has incorrect black-box U(φ) and mystery R1/R2/R4 gates that don't match the code)
- [x] 11c. Specify: optimizer = Adam(lr=0.1), loss = MSE, epochs = 10, framework = PennyLane

### 12. QSVM — Full circuit description + corrected figure
- [x] 12a. Write detailed text: 4 PCA features → ZZFeatureMap(dim=4, reps=2) encodes data into quantum state — first layer applies H and Rz(xᵢ) on each qubit, second layer applies CNOT + Rz(π·(xᵢ−xⱼ)(xᵢ+xⱼ)) for pairwise entanglement → statevector extracted → fidelity kernel K(i,j)=|⟨ψ(xᵢ)|ψ(xⱼ)⟩|² computed → fed to classical SVC(kernel="precomputed")
- [ ] 12b. Regenerate QSVM circuit diagram from Qiskit ZZFeatureMap (current figure shows a variational V(θ) block which is wrong — QSVM has NO trainable parameters, only a data-encoding feature map)
- [x] 12c. Explain the kernel trick: quantum kernel maps to Hilbert space, fidelity measures overlap

### 13. QNB — Full circuit description + corrected figure
- [ ] 13a. **CODE MISSING** — need QNB implementation to write accurate description
- [ ] 13b. Once code is found: describe encoding method, what the Q block does, measurement strategy
- [ ] 13c. Regenerate QNB circuit diagram from actual code

### 14. Theoretical descriptions (applies across all 3)
- [ ] 14a. Angle encoding theory: explain RY(π·xᵢ)|0⟩ = cos(πxᵢ/2)|0⟩ + sin(πxᵢ/2)|1⟩, why it maps normalized features to qubit amplitudes
- [ ] 14b. ZZFeatureMap theory: explain the two-local encoding U_Φ(x) = exp(i·Σφ({i,j})·ZᵢZⱼ)·exp(i·Σφ(i)·Zᵢ)·H⊗n, how reps=2 increases expressibility
- [ ] 14c. Measurement theory: explain PauliZ expectation ⟨ψ|Z|ψ⟩ ∈ [−1,1], how it's mapped to class probability

### 15. Regenerate all 3 circuit figures
- [ ] 15a. Use PennyLane qml.draw() for VQC circuit → save as figure
- [ ] 15b. Use Qiskit ZZFeatureMap.draw('mpl') for QSVM circuit → save as figure
- [ ] 15c. Regenerate QNB circuit from its code (once found)

### 16. Visualizations (also high priority)
- [ ] 16. Visualize: raw dataset, PCA-reduced, post-angular-encoding, 4-qubit flow
