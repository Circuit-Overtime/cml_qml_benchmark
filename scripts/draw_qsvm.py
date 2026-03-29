from qiskit.circuit.library import ZZFeatureMap
import matplotlib.pyplot as plt

# Exact same feature map as in pca_quantum.py line 171
feature_map = ZZFeatureMap(feature_dimension=4, reps=2)

fig = feature_map.decompose().draw("mpl", style="iqp", fold=20)
fig.set_size_inches(14, 10)
fig.savefig("paper/figures/QSVM_circuit.png", dpi=300, bbox_inches="tight")
print("Saved → paper/figures/QSVM_circuit.png")
plt.close()
