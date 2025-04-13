# Detection of Cardiovascular Disease using Neural Network

**Roll Number**: 24BM6JP04  
**Project Number**: 6  
**Course**: Fundamentals of Artificial Intelligence and Deep Machine Learning (FADML)  
**Institution**: IIT Kharagpur  
**Semester**: Spring 2025

## Project Overview

This project implements a feed-forward neural network to predict cardiovascular disease using a dataset with numerical and categorical features. The implementation includes:

- **Part 1**: Training a default model with two hidden layers (32 neurons each), batch size 32, and learning rate 0.01, plotting training and test accuracies over epochs.
- **Part 2**: Hyperparameter tuning for batch size, number of hidden layers, and learning rate, with corresponding accuracy plots.
- **Part 3**: Overfitting analysis on a subset of 1000 samples, visualizing training and test accuracies.

The code is implemented in Python using NumPy, pandas, and matplotlib, with no external machine learning libraries (manual neural network implementation).

## Directory Structure

```
24BM6JP04_DCDT/
├── dataset/
│   └── cardio_dataset.csv
├── manualtool/
│   ├── model.py
│   └── utilities.py
├── plots/
│   ├── accuracy_epochs.png
│   ├── accuracy_batch_size.png
│   ├── accuracy_hidden_layers.png
│   ├── accuracy_learning_rate.png
│   └── overfitting_analysis.png
├── README.md
└── requirements.txt
```

- `dataset/cardio_dataset.csv`: Input dataset (assumed to be provided or included if allowed).
- `manualtool/model.py`: Main script for training, tuning, and plotting.
- `manualtool/utilities.py`: Utility classes (`DataPreprocessor`, `DataLoader`, `ANNClassifier`).
- `plots/`: Directory for output plots (generated during execution).
- `README.md`: This file.
- `requirements.txt`: Python dependencies.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - numpy
  - pandas
  - matplotlib

## Installation

1. Clone or extract the project:
   ```bash
   cd 24BM6JP04_DPNN
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure `cardio_dataset.csv` is in the `dataset/` directory.
2. Run the main script:
   ```bash
   cd manualtool
   python model.py
   ```

3. **Outputs**:
   - Console: Training progress, final accuracies, and best hyperparameters.
   - Plots saved in `../plots/`:
     - `accuracy_epochs.png`: Training vs. test accuracy (Part 1).
     - `accuracy_batch_size.png`: Accuracy vs. batch size (Part 2).
     - `accuracy_hidden_layers.png`: Accuracy vs. number of hidden layers (Part 2).
     - `accuracy_learning_rate.png`: Accuracy vs. learning rate (Part 2).
     - `overfitting_analysis.png`: Training vs. test accuracy on subset (Part 3).

## Implementation Details

- **DataPreprocessor**: Handles train-test splitting (80/20), dummy encoding of categorical variables, normalization of numerical features, and one-hot encoding of the target (`cardio` → `[1, 0]` for 0, `[0, 1]` for 1).
- **DataLoader**: Provides batched data for training with shuffled indices.
- **ANNClassifier**: Implements a feed-forward neural network with:
  - ReLU activation for hidden layers.
  - Softmax activation for output (two classes).
  - Categorical cross-entropy loss.
  - Backpropagation with gradient descent.
  - Xavier weight initialization.
- **Hyperparameters**:
  - Batch sizes: [1, 2, 4, 8, 16, 32, 64, 128]
  - Hidden layers: [1, 2, 4, 6, 8]
  - Learning rates: [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]
- **Overfitting Analysis**: Uses first 1000 samples to demonstrate potential overfitting.

