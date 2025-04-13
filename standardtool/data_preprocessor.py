"""
Roll Number: 24BM6JP04
Project Number: DPNN (6)
Project Title: Detection of Cardiovascular Disease using Neural Network
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_and_preprocess(self, limit_data=False):
        df = pd.read_csv(self.filepath, sep=",")
        if limit_data:
            df = df.iloc[:1000]
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train.values, y_test.values
