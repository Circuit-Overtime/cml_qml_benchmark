import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib.pyplot as plt

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qcircuit(weights, x):
    # Stage 1: Angle Encoding — RY(π·xᵢ)
    for i in range(n_qubits):
        qml.RY(np.pi * x[i], wires=i)

    # Stage 2: Variational Ansatz — Rot(θ₁,θ₂,θ₃) + CNOT ladder
    for i in range(n_qubits):
        qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=i)

    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    # Stage 3: Measurement
    return qml.expval(qml.PauliZ(0))

# Dummy inputs for drawing
dummy_weights = pnp.random.uniform(0, np.pi, (n_qubits, 3))
dummy_x = pnp.array([0.3, 0.5, 0.7, 0.9])

fig, ax = qml.draw_mpl(qcircuit, style="pennylane")(dummy_weights, dummy_x)
fig.set_size_inches(12, 4)
fig.savefig("paper/figures/VQC_circuit.png", dpi=300, bbox_inches="tight")
print("Saved → paper/figures/VQC_circuit.png")
plt.close()
