# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 19:46:24 2025

@author: angad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
import argparse

class Preprocess:
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.normalized_data = None
        self.train_data = None
        self.test_data = None
        
    def z_score_normalization(self):
        # Separate features and target
        features = self.data.iloc[:, :-1]
        target = self.data.iloc[:, -1]
        
        # Normalize each feature
        normalized_features = (features - features.mean()) / features.std()
        self.normalized_data = pd.concat([normalized_features, target], axis=1)
        
    def train_test_split(self, test_size=0.2):
        if self.normalized_data is None:
            self.z_score_normalization()
            
        # Shuffle the data
        shuffled_data = self.normalized_data.sample(frac=1).reset_index(drop=True)
        
        # Split into train and test
        split_idx = int(len(shuffled_data) * (1 - test_size))
        self.train_data = shuffled_data.iloc[:split_idx]
        self.test_data = shuffled_data.iloc[split_idx:]
        
        # Convert to numpy arrays
        X_train = self.train_data.iloc[:, :-1].values
        y_train = self.train_data.iloc[:, -1].values
        X_test = self.test_data.iloc[:, :-1].values
        y_test = self.test_data.iloc[:, -1].values
        
        return X_train, y_train, X_test, y_test

class DataLoader:
    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_samples = X.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        
    def __iter__(self):
        # Shuffle data at the beginning of each epoch
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]
        
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.num_samples)
            yield self.X[start_idx:end_idx], self.y[start_idx:end_idx]
            
    def __len__(self):
        return self.num_batches

class WeightInitializer:
    @staticmethod
    def initialize_weights(input_size, hidden_size, output_size, num_hidden_layers=2):
        weights = []
        biases = []
        
        # First hidden layer
        weights.append(np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size))
        biases.append(np.zeros(hidden_size))
        
        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            weights.append(np.random.randn(hidden_size, hidden_size) * np.sqrt(2. / hidden_size))
            biases.append(np.zeros(hidden_size))
        
        # Output layer
        weights.append(np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size))
        biases.append(np.zeros(output_size))
        
        return weights, biases

class NeuralNetwork:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def categorical_cross_entropy(y_pred, y_true):
        m = y_true.shape[0]
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # Convert y_true to one-hot encoding if needed
        if len(y_true.shape) == 1:
            y_true_onehot = np.zeros_like(y_pred)
            y_true_onehot[np.arange(m), y_true] = 1
        else:
            y_true_onehot = y_true
        loss = -np.sum(y_true_onehot * np.log(y_pred)) / m
        return loss
    
    @staticmethod
    def accuracy(y_pred, y_true):
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        pred_labels = np.argmax(y_pred, axis=1)
        return np.mean(pred_labels == y_true)

class NeuralNetworkModel:
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=2, 
                 learning_rate=0.01, batch_size=32, epochs=200):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Initialize weights
        self.weights, self.biases = WeightInitializer.initialize_weights(
            input_size, hidden_size, output_size, num_hidden_layers)
        
        # Store training history
        self.train_loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        # Hidden layers with ReLU activation
        for i in range(self.num_hidden_layers):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = NeuralNetwork.relu(z)
            self.activations.append(a)
        
        # Output layer with softmax activation
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        a = NeuralNetwork.softmax(z)
        self.activations.append(a)
        
        return a
    
    def backward(self, X, y, y_pred):
        m = X.shape[0]
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        # Convert y to one-hot encoding if needed
        if len(y.shape) == 1:
            y_onehot = np.zeros_like(y_pred)
            y_onehot[np.arange(m), y] = 1
        else:
            y_onehot = y
        
        # Output layer gradient
        dz = (y_pred - y_onehot) / m
        gradients_w[-1] = np.dot(self.activations[-2].T, dz)
        gradients_b[-1] = np.sum(dz, axis=0)
        
        # Backpropagate through hidden layers
        for i in range(self.num_hidden_layers - 1, -1, -1):
            da = np.dot(dz, self.weights[i + 1].T)
            dz = da * NeuralNetwork.relu_derivative(self.z_values[i])
            gradients_w[i] = np.dot(self.activations[i].T, dz)
            gradients_b[i] = np.sum(dz, axis=0)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def train(self, X_train, y_train, X_test, y_test):
        train_loader = DataLoader(X_train, y_train, self.batch_size)
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_acc = 0
            
            for X_batch, y_batch in train_loader:
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss and accuracy
                loss = NeuralNetwork.categorical_cross_entropy(y_pred, y_batch)
                acc = NeuralNetwork.accuracy(y_pred, y_batch)
                
                epoch_loss += loss
                epoch_acc += acc
                
                # Backward pass and update weights
                self.backward(X_batch, y_batch, y_pred)
            
            # Average loss and accuracy for the epoch
            epoch_loss /= len(train_loader)
            epoch_acc /= len(train_loader)
            self.train_loss_history.append(epoch_loss)
            self.train_acc_history.append(epoch_acc)
            
            # Test accuracy
            test_pred = self.forward(X_test)
            test_acc = NeuralNetwork.accuracy(test_pred, y_test)
            self.test_acc_history.append(test_acc)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}, "
                      f"Train Acc: {epoch_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    def evaluate(self, X, y):
        y_pred = self.forward(X)
        return NeuralNetwork.accuracy(y_pred, y)
    
    def plot_training_history(self):
        plt.figure(figsize=(12, 5))
        
        # Plot training and test accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.train_acc_history, label='Train Accuracy')
        plt.plot(self.test_acc_history, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy')
        plt.legend()
        
        # Plot training loss
        plt.subplot(1, 2, 2)
        plt.plot(self.train_loss_history, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Neural Network for Cardiovascular Disease Prediction')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--analyze', action='store_true', help='Run hyperparameter analysis')
    args = parser.parse_args()
    
    # Preprocess data
    preprocessor = Preprocess('../dataset/cardio_dataset.csv')
    X_train, y_train, X_test, y_test = preprocessor.train_test_split(test_size=0.2)
    
    input_size = X_train.shape[1]
    output_size = 2  # Binary classification (presence/absence of disease)
    
    if not args.analyze:
        # Train model with default or specified parameters
        model = NeuralNetworkModel(
            input_size=input_size,
            hidden_size=32,
            output_size=output_size,
            num_hidden_layers=args.hidden_layers,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs
        )
        
        print("Training model...")
        model.train(X_train, y_train, X_test, y_test)
        
        # Evaluate final model
        final_train_acc = model.evaluate(X_train, y_train)
        final_test_acc = model.evaluate(X_test, y_test)
        
        print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Test Accuracy: {final_test_acc:.4f}")
        
        # Plot training history
        model.plot_training_history()
    else:
        # Hyperparameter analysis
        analyze_hyperparameters(X_train, y_train, X_test, y_test, input_size, output_size)

def analyze_hyperparameters(X_train, y_train, X_test, y_test, input_size, output_size):
    # Batch size analysis
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    batch_accuracies = []
    
    print("Analyzing batch sizes...")
    for bs in batch_sizes:
        model = NeuralNetworkModel(
            input_size=input_size,
            hidden_size=32,
            output_size=output_size,
            num_hidden_layers=2,
            learning_rate=0.01,
            batch_size=bs,
            epochs=50  # Shorter epochs for analysis
        )
        model.train(X_train, y_train, X_test, y_test)
        test_acc = model.evaluate(X_test, y_test)
        batch_accuracies.append(test_acc)
        print(f"Batch Size: {bs}, Test Accuracy: {test_acc:.4f}")
    
    plt.figure()
    plt.plot(batch_sizes, batch_accuracies, marker='o')
    plt.xscale('log', base=2)
    plt.xlabel('Batch Size')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs Batch Size')
    plt.grid()
    plt.show()
    
    # Hidden layers analysis
    hidden_layers = [1, 2, 4, 6, 8]
    layer_accuracies = []
    
    print("\nAnalyzing number of hidden layers...")
    for nl in hidden_layers:
        model = NeuralNetworkModel(
            input_size=input_size,
            hidden_size=32,
            output_size=output_size,
            num_hidden_layers=nl,
            learning_rate=0.01,
            batch_size=32,
            epochs=50
        )
        model.train(X_train, y_train, X_test, y_test)
        test_acc = model.evaluate(X_test, y_test)
        layer_accuracies.append(test_acc)
        print(f"Hidden Layers: {nl}, Test Accuracy: {test_acc:.4f}")
    
    plt.figure()
    plt.plot(hidden_layers, layer_accuracies, marker='o')
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs Number of Hidden Layers')
    plt.grid()
    plt.show()
    
    # Learning rate analysis
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]
    lr_accuracies = []
    
    print("\nAnalyzing learning rates...")
    for lr in learning_rates:
        model = NeuralNetworkModel(
            input_size=input_size,
            hidden_size=32,
            output_size=output_size,
            num_hidden_layers=2,
            learning_rate=lr,
            batch_size=32,
            epochs=50
        )
        model.train(X_train, y_train, X_test, y_test)
        test_acc = model.evaluate(X_test, y_test)
        lr_accuracies.append(test_acc)
        print(f"Learning Rate: {lr}, Test Accuracy: {test_acc:.4f}")
    
    plt.figure()
    plt.plot(learning_rates, lr_accuracies, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs Learning Rate')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()