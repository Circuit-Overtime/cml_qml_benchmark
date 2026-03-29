from qiskit import QuantumCircuit
import numpy as np
import matplotlib.pyplot as plt

n_qubits = 4
n_layers = 6
dummy_x = [0.3, 0.5, 0.7, 0.9]

qc = QuantumCircuit(n_qubits)

for layer in range(n_layers):
    scale = layer + 1

    # Amplitude Encoding — RY(π·xᵢ·scale)
    for i in range(n_qubits):
        qc.ry(np.pi * dummy_x[i] * scale, i)

    # Entangling Layer — CNOT ring
    for i in range(n_qubits):
        qc.cx(i, (i + 1) % n_qubits)

    # Phase Encoding — RZ(π·xᵢ·scale)
    for i in range(n_qubits):
        qc.rz(np.pi * dummy_x[i] * scale, i)

    # Visual separator between layers
    if layer < n_layers - 1:
        qc.barrier()

qc.measure_all()

fig = qc.draw("mpl", style="iqp", fold=20)
fig.set_size_inches(14, 16)
fig.savefig("paper/figures/QNB_circuit.png", dpi=300, bbox_inches="tight")
print("Saved → paper/figures/QNB_circuit.png")
plt.close()
