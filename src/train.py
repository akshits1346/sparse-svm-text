import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from data_loader import load_data
from model import SparseSVM

os.makedirs("experiments", exist_ok=True)

X, y = load_data()

lambdas = [0.0, 0.001, 0.01]
results = []

for lmbda in lambdas:
    print(f"\nTraining with L1 lambda = {lmbda}")
    model = SparseSVM(lr=0.01, lmbda=lmbda, epochs=50)
    model.fit(X, y)

    preds = model.predict(X)
    f1 = f1_score(y, preds)
    acc = accuracy_score(y, preds)
    sparsity = model.sparsity()

    print(f"F1-score: {f1:.2f}, Accuracy: {acc:.2f}, Sparsity: {sparsity:.2f}")

    results.append((lmbda, f1, acc, sparsity))

with open("experiments/results.md", "w") as f:
    f.write("| L1 Lambda | F1 Score | Accuracy | Sparsity |\n")
    f.write("|----------|----------|----------|----------|\n")
    for r in results:
        f.write(f"| {r[0]} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:.2f} |\n")

print("\nResults saved to experiments/results.md")

