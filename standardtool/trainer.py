"""
Roll Number: 24BM6JP04
Project Number: DPNN (6)
Project Title: Detection of Cardiovascular Disease using Neural Network
"""
import torch
import torch.nn as nn
import torch.optim as optim
from utils import calculate_accuracy, plot_accuracies

class Trainer:
    def __init__(self, model, train_loader, test_loader, lr=0.01, epochs=200):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.train_acc_list = []
        self.test_acc_list = []

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for batch_X, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            train_acc = calculate_accuracy(self.model, self.train_loader)
            test_acc = calculate_accuracy(self.model, self.test_loader)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

        plot_accuracies(self.train_acc_list, self.test_acc_list, title="Train/Test Accuracy over Epochs")
