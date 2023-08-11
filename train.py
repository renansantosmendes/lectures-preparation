import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, InputLayer, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mlflow


def reset_seeds():
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)


def read_data():
    data = pd.read_csv('https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv')
    return data


def prepare_data(data):
    X = data.drop(["fetal_health"], axis=1)
    y = data["fetal_health"]

    columns_names = list(X.columns)
    scaler = preprocessing.StandardScaler()
    X_df = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=columns_names)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)

    y_train = y_train - 1
    y_test = y_test - 1

    return X_train, X_test, y_train, y_test


def create_model(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    return model


def compile_model(model):
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


def train_model(model, X_train, y_train):
    with mlflow.start_run(run_name='experiment_01') as run:
        model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=3)


def main():
    reset_seeds()
    data = read_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    model = create_model(input_shape=X_train.shape[1])
    compile_model(model)
    train_model(model, X_train, y_train)


if __name__ == "__main__":
    main()