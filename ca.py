import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.dense1(x)
        out = self.relu(out)
        out = self.dense2(out)
        out = self.softmax(out)
        return out

# Define the Cultural Algorithm optimization function
class CulturalAlgorithm:
    def __init__(self, n_populations=20, n_iterations=100, n_parents=10, n_immigrants=2, mutation_rate=0.1, mutation_range=0.1):
        self.n_populations = n_populations
        self.n_iterations = n_iterations
        self.n_parents = n_parents
        self.n_immigrants = n_immigrants
        self.mutation_rate = mutation_rate
        self.mutation_range = mutation_range
        
    def optimize_weights(self, nn, X, y):
        # Standardize input data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        
        # Create initial population
        populations = []
        for i in range(self.n_populations):
            weights = nn.state_dict()
            populations.append(weights)
        
        # Optimization loop
        for i in range(self.n_iterations):
            fitnesses = []
            new_populations = []
            # Evaluate fitness of each population
            for j in range(self.n_populations):
                nn.load_state_dict(populations[j])
                y_pred = nn(X)
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(y_pred, torch.max(y, 1)[1]).item()
                fitnesses.append(loss)
            # Select parents and immigrants based on fitness
            sorted_populations = [pop for _, pop in sorted(zip(fitnesses, populations))]
            parents = sorted_populations[:self.n_parents]
            immigrants = [np.random.uniform(low=-self.mutation_range, high=self.mutation_range, size=pop.shape) for pop in parents[:self.n_immigrants]]
            # Create new populations using mutation and immigrants
            for j in range(self.n_populations):
                child_weights = sorted_populations[j]
                for key, value in child_weights.items():
                    if np.random.uniform() < self.mutation_rate:
                        child_weights[key] += np.random.normal(scale=self.mutation_range)
                # Apply immigrants
                if j < self.n_immigrants:
                    for key, value in immigrants[j].items():
                        child_weights[key] += value
                # Add to new population
                new_populations.append(child_weights)
            
            populations = new_populations
        
        # Select the best weights
        nn.load_state_dict(sorted_populations[0])
        
        return nn

# Generate sample dataset
X, y = pd.read_csv('/Users/jaishree/Downloads/Bank_Personal_Loan_Modelling.csv')

# Create neural network
nn = NeuralNetwork(X.shape[1], 32, 2)

# Create Cultural Algorithm optimizer and run optimization
ca = CulturalAlgorithm(n_populations=20, n_iterations=100, n_parents=10, n_immigrants=2, mutation_rate=0.1, mutation_range=0.1)
nn_optimized = ca.optimize_weights(nn, X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.Tensor(scaler.transform(X_train))
y_train = torch.Tensor(y_train)
y_train = torch.max(y_train, 1)[1]
X_test = torch.Tensor(scaler.transform(X_test))
y_test = torch.Tensor(y_test)
y_test = torch.max(y_test, 1)[1]

y_pred = nn_optimized(X_test)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(y_pred, y_test).item()
accuracy = (y_pred.argmax(dim=1) == y_test).sum().item() / len(y_test)

print("Optimized neural network loss:", loss)
print("Optimized neural network accuracy:", accuracy)
