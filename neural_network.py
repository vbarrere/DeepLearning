#!/usr/bin/python3

import os
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

img_path = "../Data/Data_HRTEM/HRTEM_image"
data = pd.read_csv("../Data/Data_processed/data.dat", sep="\t", na_values=["nan"])

data["image_file"] = data["i_sim"] + '.png'
data["eta_parameter"] = 2 * np.abs(data["nat1_out"] / (data["nat1_out"] + data["nat2_out"]) - data["nat1"] / data["n_atoms"]) + 2 * np.abs(data["nat1_in"] / (data["nat1_in"] + data["nat2_in"]) - data["nat1"] / data["n_atoms"]) - data["d_com"] / (2*data["gyration_radius"])
mask = np.isnan(data["eta_parameter"])
data = data[~mask]

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)


datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=img_path,
    x_col="image_file",
    y_col="eta_parameter",
    class_mode="raw",
    target_size=(64, 64),
    batch_size=32,
    color_mode="grayscale"
)

val_gen = datagen.flow_from_dataframe(
    dataframe=val_data,
    directory=img_path,
    x_col="image_file",
    y_col="eta_parameter",
    class_mode="raw",
    target_size=(64, 64),
    batch_size=32,
    color_mode="grayscale"
)

test_gen = datagen.flow_from_dataframe(
    dataframe=test_data,
    directory=img_path,
    x_col="image_file",
    y_col="eta_parameter",
    class_mode="raw",
    target_size=(64, 64),
    batch_size=32,
    color_mode="grayscale"
)



""" Architecture du réseau de neurones """

modele = tf.keras.models.Sequential()
modele.add(tf.keras.layers.Input(shape=(64, 64, 1)))
modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
modele.add(tf.keras.layers.MaxPooling2D(pool_size=2))
modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
modele.add(tf.keras.layers.MaxPooling2D(pool_size=2))
modele.add(tf.keras.layers.Flatten())
modele.add(tf.keras.layers.Dense(1024, activation='relu'))
modele.add(tf.keras.layers.Dense(768, activation='relu'))
modele.add(tf.keras.layers.Dense(1, activation='sigmoid'))

modele.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(modele.summary())
modele.fit(train_gen, epochs=5)
score = modele.evaluate(train_gen, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



y_pred = modele.predict(test_gen)
y_true = test_data["eta_parameter"].values

plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel("Valeurs réelles de eta")
plt.ylabel("Valeurs prédites de eta")
plt.title("Validation des prédictions")
plt.savefig("validation_predictions.png")
plt.show()