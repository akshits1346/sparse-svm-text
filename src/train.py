import numpy as np
from data_loader import load_data
from model import SparseSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lambdas = [0.0, 0.001, 0.01]
results = []

for lmbda in lambdas:
    print(f"\nTraining with L1 lambda = {lmbda}")
    model = SparseSVM(lr=0.1, epochs=50, lmbda=lmbda)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"F1-score: {f1:.2f}, Accuracy: {acc:.2f}")
    results.append((lmbda, f1, acc))

with open("../experiments/results.md", "w") as f:
    f.write("## Sparse SVM Results\n\n")
    f.write("| L1 Regularization | F1-score | Accuracy |\n")
    f.write("|-----------------|----------|---------|\n")
    for lmbda, f1, acc in results:
        f.write(f"| {lmbda} | {f1:.2f} | {acc:.2f} |\n")

print("\nAll results saved to experiments/results.md")

