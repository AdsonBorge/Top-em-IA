import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA


def load_and_split_data(test_size=0.5, random_state=42):
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def train_model(X_train, y_train, hidden_layer_sizes=(10, 10), max_iter=500, random_state=42):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                          max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall


def main():
    # Carregar e dividir os dados
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Padronizar os dados
    X_train, X_test = standardize_data(X_train, X_test)

    # Treinar o modelo
    model = train_model(X_train, y_train)

    # Avaliar o modelo
    accuracy, precision, recall = evaluate_model(model, X_test, y_test)

    print(f'Acurácia: {accuracy:.3f}')
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')


if __name__ == '__main__':
    main()
