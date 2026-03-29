import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib.pyplot as plt

n_qubits = 4
n_layers = 3
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def qnb_circuit(x):
    for layer in range(n_layers):
        # Amplitude Encoding — RY(π·xᵢ)
        for i in range(n_qubits):
            qml.RY(np.pi * x[i], wires=i)

        # Entangling Layer — CNOT ring
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # Phase Encoding — RZ(π·xᵢ)
        for i in range(n_qubits):
            qml.RZ(np.pi * x[i], wires=i)

    # Measurement — probability distribution
    return qml.probs(wires=range(n_qubits))


dummy_x = pnp.array([0.3, 0.5, 0.7, 0.9])

fig, ax = qml.draw_mpl(qnb_circuit, style="pennylane")(dummy_x)
fig.set_size_inches(18, 5)
fig.savefig("paper/figures/QNB_circuit.png", dpi=300, bbox_inches="tight")
print("Saved → paper/figures/QNB_circuit.png")
plt.close()
