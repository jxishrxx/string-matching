import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import torch.nn as nn
from aco import ACO, Graph

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
    hidden_size = 10
    output_size = 2
    
    # Convert flattened weights into a PyTorch tensor
    weights = torch.Tensor(weights)
    
    # Create a new instance of the neural network
    net = NeuralNetwork(input_size, hidden_size, output_size)
    
    # Set the weights of the neural network
    start = 0
    for param in net.parameters():
        end = start + np.prod(param.shape)
        param.data = weights[start:end].reshape(param.shape)
        start = end
    
    # Train the neural network
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    num_epochs = 10
    for epoch in range(num_epochs):
        outputs = net(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate the neural network
    outputs = net(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = torch.sum(predicted == y_test).item() / y_test.size(0)
    
    return accuracy

# Define the optimization problem
graph = Graph(len(X_train[0]), hidden_size=10, output_size=2)
aco = ACO(graph, fitness_function, colony_size=10, elite_ant_count=2, max_iterations=50)

# Run the optimization algorithm
weights = aco.run()

# Create a new instance of the neural network using the optimized weights
net = NeuralNetwork(len(X_train[0]), 10, 2)
start = 0
for param in net.parameters():
    end = start + np.prod(param.shape)
    param.data = torch.Tensor(weights[start:end]).reshape(param.shape)
    start = end

# Evaluate the neural network using the optimized weights
outputs = net(X_test)
_, predicted = torch.max(outputs, 1)
accuracy = torch.sum(predicted == y_test).item() / y_test.size(0)
print(f"Accuracy: {accuracy}")
