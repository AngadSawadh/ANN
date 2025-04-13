"""
Roll Number: 24BM6JP04
Project Number: DPNN (6)
Project Title: Detection of Cardiovascular Disease using Neural Network
"""
import argparse
from data_preprocessor import DataPreprocessor
from data_loader import get_data_loaders
from model import ANNModel
from trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    data = DataPreprocessor("cardio_dataset.csv")
    X_train, X_test, y_train, y_test = data.load_and_preprocess()

    train_loader, test_loader = get_data_loaders(X_train, y_train, X_test, y_test, batch_size=args.batch_size)
    model = ANNModel(input_size=X_train.shape[1], hidden_layers=args.hidden_layers)

    trainer = Trainer(model, train_loader, test_loader, lr=args.lr, epochs=200)
    trainer.train()

    print("Final Train Accuracy:", trainer.train_acc_list[-1])
    print("Final Test Accuracy:", trainer.test_acc_list[-1])
