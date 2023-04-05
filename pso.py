import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import pyswarms as ps
import pandas as pd

# Generate a sample dataset
X, y = pd.read_csv('/Users/jaishree/Downloads/Bank_Personal_Loan_Modelling.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train).long()
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test).long()

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define the fitness function to optimize the neural network
def fitness_function(weights, X, y):
    input_size = X.shape[1]
    hidden_size = 20
    output_size = len(torch.unique(y))
    model = NeuralNetwork(input_size, hidden_size, output_size)
    model.load_state_dict(torch.from_numpy(weights.reshape(-1, 1)))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    y_pred = model(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Define the bounds for the PSO optimizer
n_weights = sum(p.numel() for p in NeuralNetwork(10, 20, 2).parameters())
lb = np.zeros(n_weights)
ub = np.ones(n_weights)

# Set up the PSO optimizer
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=n_weights, options=options, bounds=(lb, ub))

# Optimize the neural network weights
cost, pos = optimizer.optimize(fitness_function, iters=50, X=X_train, y=y_train)

# Evaluate the optimized neural network
model = NeuralNetwork(10, 20, 2)
model.load_state_dict(torch.from_numpy(pos.reshape(-1, 1)))
y_pred = model(X_test)
accuracy = (y_pred.argmax(dim=1) == y_test).sum().item() / len(y_test)
print("Optimized neural network accuracy:", accuracy)
