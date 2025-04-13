# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 23:38:04 2025

@author: angad

Roll Number: 24BM6JP04
Project Number: DPNN (6)
Project Title: Detection of Cardiovascular Disease using Neural Network

Main script to train and evaluate a feed-forward neural network for cardiovascular disease prediction.
"""
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
from utilities import ANNClassifier, DataPreprocessor

def parse_args():
    parser = argparse.ArgumentParser(description="To train the model and get test metrics")
    parser.add_argument("--data", type=str, default="../dataset/cardio_dataset.csv", help="path to data")
    parser.add_argument("--output", type=int, default=1, help="output size")
    parser.add_argument("--catcols", type=str, nargs="+", default=["gender", "cholesterol", "gluc", "smoke", "alco", "active"], help="categorical column names")
    parser.add_argument("--numcols", type=str, nargs="+", default=["age", "height", "weight", "ap_hi", "ap_lo"], help="numerical column names")
    parser.add_argument("--tarcols", type=str, nargs="+", default=["cardio"], help="target column names")
    parser.add_argument("--nlayers", type=int, nargs='+', default=[1, 2, 4, 6, 8], help="number of hidden layers")
    parser.add_argument("--nneurons", type=int, nargs='+', default=[32, 32, 32, 32, 32], help="number of neurons in each hidden layer")
    parser.add_argument("--bs", type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128], help="training batch size")
    parser.add_argument("--lrs", type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.2, 0.4, 0.8], help="learning rate")
    
    if len(sys.argv) == 1:
        return parser.parse_args([])
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    data = pd.read_csv(args.data)
    
    # Preprocess to determine input size after dummy encoding
    preprocessor = DataPreprocessor(data.copy(), args.numcols, args.catcols, args.tarcols)
    preprocessor.split_train_test().create_dummies()
    input_size = len(preprocessor.data.columns) - len(args.tarcols)  # Exclude target column
    
    # Part 1: Default Model Training
    classifier = ANNClassifier(data = data, 
                           input_size = input_size , 
                           output_size = args.output, 
                           num_cols = args.numcols, 
                           cat_cols = args.catcols, 
                           target_col = args.tarcols,
                           batch_size=32,
                           learning_rate=0.01,
                           hidden_layer_neurons=[32, 32])

    train_loss, train_acc, test_acc = classifier.train(200, verbose=True)
    classifier.load_best_model()
    final_test_acc = classifier.test()
    print(f"Final Training Accuracy: {train_acc[-1]:.6f}")
    print(f"Final Test Accuracy: {final_test_acc:.6f}")

    # Plot train/test accuracy
    plt.figure()
    plt.plot(range(10, 201, 10), train_acc[9::10], label="Training Accuracy")
    plt.plot(range(10, 201, 10), test_acc, label="Test Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy Over Epochs')
    plt.legend()
    plt.savefig("plots/accuracy_epochs.png")
    plt.show()

    # Part 2: Hyperparameter Tuning
    accuracies_batch = []
    accuracies_hidden_layers = []
    accuracies_lr = [] 
    best_val_acc = 0
    best_test_acc = 0
    best_params = {}
    
    preprocessor = DataPreprocessor(data.copy(), args.numcols, args.catcols, args.tarcols)
    preprocessor.split_train_test().create_dummies().normalize().create_mat()
    val_size = int(0.2 * preprocessor.train_mat.shape[0])
    X_train, y_train = preprocessor.train_mat[val_size:], preprocessor.train_target[val_size:]
    X_val, y_val = preprocessor.train_mat[:val_size], preprocessor.train_target[:val_size]
    X_test, y_test = preprocessor.test_mat, preprocessor.test_target

    for batch_size in args.bs:
        classifier.set_batch_size(batch_size)
        classifier.set_hidden_layers([32] * 2)
        classifier.set_learning_rate(0.01)
        _, train_acc, val_acc = classifier.train_with_validation(X_train, y_train, X_val, y_val, epochs=200)
        test_acc = classifier.test()
        accuracies_batch.append((val_acc[-1], test_acc))
        if val_acc[-1] > best_val_acc:
            best_val_acc = val_acc[-1]
            best_test_acc = test_acc
            best_params = {'batch_size': batch_size, 'hidden_layers': 2, 'learning_rate': 0.01}
    
    plt.figure()
    plt.plot(args.bs, [x[0] for x in accuracies_batch], marker='o', label='Validation Accuracy')
    plt.plot(args.bs, [x[1] for x in accuracies_batch], marker='o', label='Test Accuracy')
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Batch Size')
    plt.legend()
    plt.savefig("plots/accuracy_batch_size.png")
    plt.show()

    for layers in args.nlayers:
        classifier.set_batch_size(32)
        classifier.set_hidden_layers([32] * layers)
        classifier.set_learning_rate(0.01)
        _, train_acc, val_acc = classifier.train_with_validation(X_train, y_train, X_val, y_val, epochs=200)
        test_acc = classifier.test()
        accuracies_hidden_layers.append((val_acc[-1], test_acc))
        if val_acc[-1] > best_val_acc:
            best_val_acc = val_acc[-1]
            best_test_acc = test_acc
            best_params = {'batch_size': 32, 'hidden_layers': layers, 'learning_rate': 0.01}
    
    plt.figure()
    plt.plot(args.nlayers, [x[0] for x in accuracies_hidden_layers], marker='o', label='Validation Accuracy')
    plt.plot(args.nlayers, [x[1] for x in accuracies_hidden_layers], marker='o', label='Test Accuracy')
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Hidden Layers')
    plt.legend()
    plt.savefig("plots/accuracy_hidden_layers.png")
    plt.show()
    
    for lr in args.lrs:
        classifier.set_batch_size(32)
        classifier.set_hidden_layers([32] * 2)
        classifier.set_learning_rate(lr)
        _, train_acc, val_acc = classifier.train_with_validation(X_train, y_train, X_val, y_val, epochs=200)
        test_acc = classifier.test()
        accuracies_lr.append((val_acc[-1], test_acc))
        if val_acc[-1] > best_val_acc:
            best_val_acc = val_acc[-1]
            best_test_acc = test_acc
            best_params = {'batch_size': 32, 'hidden_layers': 2, 'learning_rate': lr}

    plt.figure()
    plt.plot(args.lrs, [x[0] for x in accuracies_lr], marker='o', label='Validation Accuracy')
    plt.plot(args.lrs, [x[1] for x in accuracies_lr], marker='o', label='Test Accuracy')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Learning Rate')
    plt.legend()
    plt.savefig("plots/accuracy_learning_rate.png")
    plt.show()

    print(f"Best Model Parameters: {best_params}")
    print(f"Best Validation Accuracy: {best_val_acc:.6f}")
    print(f"Best Test Accuracy: {best_test_acc:.6f}")
        
    #Part 3: Overfitting Analysis
    subset_data = data.iloc[:1000]
    classifier_subset = ANNClassifier(
    data=subset_data,
    input_size=input_size,
    output_size=1,
    num_cols=args.numcols,
    cat_cols=args.catcols,
    target_col=args.tarcols,
    batch_size=32,
    learning_rate=0.01,
    hidden_layer_neurons=[32, 32])
    train_loss_sub, train_acc_sub, test_acc_sub = classifier_subset.train(epochs=200, verbose=True, plot_interval=25)
    
    plt.figure()
    plt.plot(range(25, 201, 25), train_acc_sub[24::25], label="Training Accuracy")
    plt.plot(range(25, 201, 25), test_acc_sub, label="Test Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Overfitting Analysis (First 1000 Points)')
    plt.legend()
    plt.savefig("plots/overfitting_analysis.png")
    plt.show()


