# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 21:51:34 2025

@author: angad

Roll Number: 24BM6JP04
Project Number: 6
Project Title: Detection of Cardiovascular Disease using Neural Network

Utility classes for data preprocessing, loading, and neural network implementation.
"""


import numpy as np
import pandas as pd

class DataPreprocessor:
    def __init__(self, data, num_cols, cat_cols, target_col, train_split=0.8):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        if not all(col in data.columns for col in num_cols + cat_cols + target_col):
            raise ValueError("Some columns not found in data")
        self.data = data
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.target_col = target_col
        self.train_split = train_split
        self.train_data = None
        self.test_data = None
        self.train_mat = None #row by col
        self.test_mat = None #row by col
        self.train_target = None #row by col
        self.test_target = None #row by col

    def split_train_test(self):
        indices = np.random.permutation(self.data.shape[0])
        train_size = int(self.train_split * self.data.shape[0])
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        self.train_data = self.data.iloc[train_indices].reset_index(drop=True)
        self.test_data = self.data.iloc[test_indices].reset_index(drop=True)
        return self

    def create_mat(self):
        if self.train_data is not None and self.test_data is not None:
            self.train_mat = self.train_data.drop(columns=self.target_col).to_numpy()
            self.test_mat = self.test_data.drop(columns=self.target_col).to_numpy()
            self.train_target = self.train_data[self.target_col].to_numpy()
            self.test_target = self.test_data[self.target_col].to_numpy()
        return self

    def create_dummies(self):
        if isinstance(self.data, pd.DataFrame):
            self.data = pd.get_dummies(self.data, columns=self.cat_cols, drop_first=True, dtype=int)
        if isinstance(self.train_data, pd.DataFrame) and isinstance(self.test_data, pd.DataFrame):
            self.train_data = pd.get_dummies(self.train_data, columns=self.cat_cols, drop_first=True, dtype=int)
            self.test_data = pd.get_dummies(self.test_data, columns=self.cat_cols, drop_first=True, dtype=int)
        return self
    
    def normalize(self):
        if isinstance(self.data, pd.DataFrame) and isinstance(self.train_data, pd.DataFrame) and isinstance(self.test_data, pd.DataFrame):
            means = self.train_data[self.num_cols].mean()
            stds = self.train_data[self.num_cols].std()
            self.train_data[self.num_cols] = (self.train_data[self.num_cols] - means) / stds
            self.test_data[self.num_cols] = (self.test_data[self.num_cols] - means) / stds
        return self

class DataLoader:
    def __init__(self, data, target, batch_size=32, binary_class = True):
        self.data = data
        self.target = target.reshape(-1, 1) if binary_class else pd.get_dummies(target).to_numpy()
        self.num_samples = data.shape[0]
        self.batch_size = batch_size

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self

    def get_batches_per_epoch(self):
        indices = np.random.permutation(self.num_samples)
        num_batches = int(np.ceil(self.num_samples / self.batch_size))
        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min((batch_idx + 1) * self.batch_size, self.num_samples)
            batch_indices = indices[start:end]
            yield self.data[batch_indices], self.target[batch_indices].T

class ANNClassifier:
    def __init__(self, data, input_size, output_size, num_cols, cat_cols, target_col,
                 batch_size=32, learning_rate=0.01, hidden_layer_neurons=[32,32]):
        self.data = data #pandas dataframe
        self.input_size = input_size #input neurons
        self.output_size = output_size #output neurons
        self.is_binary = (output_size == 1) #is a binary classifier
        self.batch_size = batch_size #batch size for training
        self.learning_rate = learning_rate #learning rate for training
        self.hidden_layer_neurons = hidden_layer_neurons #hidden layer configuration
        
        self.preprocessor = DataPreprocessor(data, num_cols, cat_cols, target_col)
        self.preprocessor.split_train_test().create_dummies().normalize().create_mat()
        
        if self.preprocessor.train_mat.shape[1] != input_size:
            raise ValueError(f"Preprocessed data has {self.preprocessor.train_mat.shape[1]} features, expected {input_size}")
        
        self.train_loader = DataLoader(self.preprocessor.train_mat, self.preprocessor.train_target, batch_size, self.is_binary)
        self.test_loader = DataLoader(self.preprocessor.test_mat, self.preprocessor.test_target, batch_size, self.is_binary)
        
        self.weights = []
        self.biases = []
        self.activations = []
        self.pre_activations = []
        self.best_weights = None
        self.best_biases = None
        self.best_loss = float('inf')
        self.best_config = None

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.train_loader.set_batch_size(batch_size)
        return self

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        return self

    def set_hidden_layers(self, hidden_layer_neurons):
        self.hidden_layer_neurons = hidden_layer_neurons
        return self
    
    def sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def weight_initializer(self):
        layer_sizes = [self.input_size] + self.hidden_layer_neurons + [self.output_size]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            weight_matrix = np.random.uniform(-limit, limit, (layer_sizes[i + 1], layer_sizes[i]))
            bias_vector = np.zeros((layer_sizes[i + 1], 1))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def forward_pass(self, X):
        self.activations = []
        self.pre_activations = []
        
        A = X.T
        self.activations.append(A)
        
        for i in range(len(self.weights) - 1):
            Z = self.weights[i] @ A + self.biases[i]
            self.pre_activations.append(Z)
            A = self.relu(Z)
            self.activations.append(A)
        
        Z = self.weights[-1] @ A + self.biases[-1]
        self.pre_activations.append(Z)
        output = self.softmax(Z) if not self.is_binary else self.sigmoid(Z)
        self.activations.append(output)
        
        return output

    def backward_pass(self, X, y_true):
        batch_size = X.shape[0]
        y_pred = self.forward_pass(X)
        
        num_layers = len(self.weights)
        dW = [None] * num_layers
        db = [None] * num_layers
        
        delta = (y_pred - y_true) / batch_size
        
        for i in range(num_layers - 1, -1, -1):
            dW[i] = delta @ self.activations[i].T
            db[i] = np.sum(delta, axis=1, keepdims=True)
            
            if i > 0:
                delta_prev = self.weights[i].T @ delta
                delta = delta_prev * self.relu_derivative(self.pre_activations[i-1])
            
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
        
        if self.is_binary:
            loss = -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
            accuracy = np.mean((y_pred > 0.5) == y_true)
        else:
            loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=0))
            accuracy = np.mean(np.argmax(y_pred, axis=0) == np.argmax(y_true, axis=0))
        
        return loss, accuracy

    def train(self, epochs, verbose=False, plot_interval=10):
        self.weight_initializer()
        train_loss_history = []
        train_accuracy_history = []
        test_accuracy_history = []
        
        for epoch in range(epochs):
            batches = self.train_loader.get_batches_per_epoch()
            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = 0
            
            for X_batch, y_batch in batches:
                #X_batch = X_batch.T # col by rows
                #y_batch = y_batch.reshape(1, -1) if self.is_binary else y_batch.T # col by rows\
                loss, accuracy = self.backward_pass(X_batch, y_batch)
                epoch_loss += loss
                epoch_accuracy += accuracy
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            train_loss_history.append(avg_loss)
            train_accuracy_history.append(avg_accuracy)
            
            if (epoch + 1) % plot_interval == 0:
                test_acc = self.test()
                test_accuracy_history.append(test_acc)
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, Train Acc: {avg_accuracy:.6f}, Test Acc: {test_acc:.6f}")
            
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_weights = [w.copy() for w in self.weights]
                self.best_biases = [b.copy() for b in self.biases]
                self.best_config = {
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate,
                    'hidden_layer_neurons': self.hidden_layer_neurons.copy()
                }
        
        return train_loss_history, train_accuracy_history, test_accuracy_history
    
    def train_with_validation(self, X_train, y_train, X_val, y_val, epochs):
        self.weight_initializer()
        train_accuracy_history = []
        val_accuracy_history = []
        
        train_loader = DataLoader(X_train, y_train, self.batch_size, self.is_binary)
        for epoch in range(epochs):
            batches = train_loader.get_batches_per_epoch()
            epoch_accuracy = 0
            num_batches = 0
            
            for X_batch, y_batch in batches:
                _, accuracy = self.backward_pass(X_batch, y_batch)
                epoch_accuracy += accuracy
                num_batches += 1
            
            train_accuracy_history.append(epoch_accuracy / num_batches)
            if (epoch + 1) % 10 == 0:
                y_pred_val = self.forward_pass(X_val)
                val_acc = np.mean((y_pred_val.T > 0.5) == y_val)
                val_accuracy_history.append(val_acc)
        
        return train_accuracy_history, train_accuracy_history, val_accuracy_history
    
    def test(self):
        X_test = self.test_loader.data
        y_test = self.test_loader.target
        y_pred = self.forward_pass(X_test)
        accuracy = np.mean((y_pred > 0.5) == y_test)
        return accuracy

    def load_best_model(self):
        if self.best_weights is not None and self.best_biases is not None:
            self.weights = [w.copy() for w in self.best_weights]
            self.biases = [b.copy() for b in self.best_biases]
        return self
