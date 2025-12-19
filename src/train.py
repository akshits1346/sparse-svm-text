from data_loader import load_data
from model import LinearSVM

X, y = load_data()
model = LinearSVM()
model.fit(X, y)

print("Training finished")

