"""
Roll Number: 24BM6JP04
Project Number: DPNN (6)
Project Title: Detection of Cardiovascular Disease using Neural Network
"""
import matplotlib.pyplot as plt
import numpy as np
from main import DataPreprocessor, Trainer, ANNModel, get_data_loaders

def tune_hyperparameters():
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    hidden_layers_options = [1, 2, 4, 6, 8]
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]

    data = DataPreprocessor("cardio_dataset.csv")
    X_train, X_test, y_train, y_test = data.load_and_preprocess()

    # Plot Accuracy vs Batch Size
    batch_size_acc = []
    for batch_size in batch_sizes:
        train_loader, test_loader = get_data_loaders(X_train, y_train, X_test, y_test, batch_size)
        model = ANNModel(input_size=X_train.shape[1], hidden_layers=2)
        trainer = Trainer(model, train_loader, test_loader, lr=0.01)
        trainer.train()
        batch_size_acc.append(trainer.test_acc_list[-1])
    
    plt.figure()
    plt.plot(batch_sizes, batch_size_acc, marker='o')
    plt.xlabel("Batch Size")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Batch Size")
    plt.savefig("plots/acc_vs_batch_size.png")
    plt.close()

    # Plot Accuracy vs Hidden Layers
    hidden_layers_acc = []
    for hidden_layers in hidden_layers_options:
        train_loader, test_loader = get_data_loaders(X_train, y_train, X_test, y_test, batch_size=32)
        model = ANNModel(input_size=X_train.shape[1], hidden_layers=hidden_layers)
        trainer = Trainer(model, train_loader, test_loader, lr=0.01)
        trainer.train()
        hidden_layers_acc.append(trainer.test_acc_list[-1])
    
    plt.figure()
    plt.plot(hidden_layers_options, hidden_layers_acc, marker='o')
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Number of Hidden Layers")
    plt.savefig("plots/acc_vs_hidden_layers.png")
    plt.close()

    # Plot Accuracy vs Learning Rate
    lr_acc = []
    for lr in learning_rates:
        train_loader, test_loader = get_data_loaders(X_train, y_train, X_test, y_test, batch_size=32)
        model = ANNModel(input_size=X_train.shape[1], hidden_layers=2)
        trainer = Trainer(model, train_loader, test_loader, lr=lr)
        trainer.train()
        lr_acc.append(trainer.test_acc_list[-1])
    
    plt.figure()
    plt.plot(learning_rates, lr_acc, marker='o')
    plt.xlabel("Learning Rate")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Learning Rate")
    plt.savefig("plots/acc_vs_lr.png")
    plt.close()

if __name__ == "__main__":
    tune_hyperparameters()
