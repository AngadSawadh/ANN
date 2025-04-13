"""
Roll Number: 24BM6JP04
Project Number: DPNN (6)
Project Title: Detection of Cardiovascular Disease using Neural Network
"""
import torch
import matplotlib.pyplot as plt
import os

def calculate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total

def plot_accuracies(train_acc, test_acc, title="Accuracy Plot", filename="train_test_accuracy.png"):
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.figure()
    epochs = list(range(1, len(train_acc) + 1))
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, test_acc, label='Test Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.savefig(f"plots/{filename}")
    plt.close()
