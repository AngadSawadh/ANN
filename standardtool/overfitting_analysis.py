"""
Roll Number: 24BM6JP04
Project Number: DPNN (6)
Project Title: Detection of Cardiovascular Disease using Neural Network
"""
import matplotlib.pyplot as plt
from main import DataPreprocessor, Trainer, ANNModel, get_data_loaders

def overfitting_analysis():
    # Use first 1000 data points for training
    data = DataPreprocessor("cardio_dataset.csv")
    X_train, X_test, y_train, y_test = data.load_and_preprocess(limit_data=True)

    # Create data loaders
    train_loader, test_loader = get_data_loaders(X_train, y_train, X_test, y_test, batch_size=32)

    # Initialize model
    model = ANNModel(input_size=X_train.shape[1], hidden_layers=2)

    # Initialize trainer
    trainer = Trainer(model, train_loader, test_loader, lr=0.01, epochs=200)
    
    # Train the model
    trainer.train()

    # Plot training and test accuracy over epochs
    plt.figure()
    plt.plot(range(1, len(trainer.train_acc_list) + 1), trainer.train_acc_list, label='Train Accuracy')
    plt.plot(range(1, len(trainer.test_acc_list) + 1), trainer.test_acc_list, label='Test Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Overfitting Analysis: Train vs Test Accuracy")
    plt.legend()
    plt.savefig("plots/overfitting_plot.png")
    plt.close()

if __name__ == "__main__":
    overfitting_analysis()
