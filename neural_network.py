#!/usr/bin/python3

import os
import glob
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

start_time = time.time()


""" Chargement et prétraitement des données """

img_path = "../Data/HRTEM_data/AgCo/hrtem_images/"
data = pd.read_csv("../Data/HRTEM_data/AgCo/data.dat", sep="\t", na_values=["nan"])
data.columns = ["i_sim", "n_atoms", "n_steps", "initial_temperature", "epot", "surface_area", "solid_volume", "cna_others", "cna_fcc", "cna_hcp", "cna_bcc", "cna_ico", "bond_angle_others", "bond_angle_fcc", "bond_angle_hcp", "bond_angle_bcc", "bond_angle_ico", "csp", "gyration_radius", "nat1", "nat2", "nat1_out", "nat2_out", "nat1_in", "nat2_in", "r_cm1_x", "r_cm1_y", "r_cm1_z", "r_cm2_x", "r_cm2_y", "r_cm2_z", "r_cm_x", "r_cm_y", "r_cm_z", "d_com", "counts", "phi", "theta", "image_shift_x", "image_shift_y", "defocus_x", "defocus_y", "astigmatism_x", "astigmatism_y", "coma_x", "coma_y", "three_lobe_aberration_x", "three_lobe_aberration_y", "spherical_aberration_x", "spherical_aberration_y", "star_aberration_x", "star_aberration_y"]
data = data.head(10000)
data["image_file"] = data["i_sim"] + '.png'
data["eta_parameter"] = 2 * np.abs(data["nat1_out"] / (data["nat1_out"] + data["nat2_out"]) - data["nat1"] / data["n_atoms"]) + 2 * np.abs(data["nat1_in"] / (data["nat1_in"] + data["nat2_in"]) - data["nat1"] / data["n_atoms"]) - data["d_com"] / (2*data["gyration_radius"])
mask = np.isnan(data["eta_parameter"])

data["eta_class"] = (data["eta_parameter"] > 0).astype(int)

data = data[~mask]
train_data, test_data = train_test_split(data, train_size=0.6)
val_data, test_data = train_test_split(test_data, test_size=0.5)


""" Création de générateurs d'images pour l'entraînement, la validation et le test """

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=img_path,
    x_col="image_file",
    y_col="eta_class",
    class_mode="raw",
    target_size=(64, 64),
    batch_size=32,
    color_mode="grayscale"
)
val_gen = datagen.flow_from_dataframe(
    dataframe=val_data,
    directory=img_path,
    x_col="image_file",
    y_col="eta_class",
    class_mode="raw",
    target_size=(64, 64),
    batch_size=32,
    color_mode="grayscale"
)
test_gen = datagen.flow_from_dataframe(
    dataframe=test_data,
    directory=img_path,
    x_col="image_file",
    y_col="eta_class",
    class_mode="raw",
    target_size=(64, 64),
    batch_size=32,
    color_mode="grayscale"
)



""" Architecture du réseau de neurones """

modele = tf.keras.models.Sequential()
modele.add(tf.keras.layers.Input(shape=(64, 64, 1)))
modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
modele.add(tf.keras.layers.MaxPooling2D(pool_size=2))
modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
modele.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
modele.add(tf.keras.layers.MaxPooling2D(pool_size=2))
modele.add(tf.keras.layers.Flatten())
modele.add(tf.keras.layers.Dense(1024, activation='relu'))
modele.add(tf.keras.layers.Dense(768, activation='relu'))
modele.add(tf.keras.layers.Dense(1, activation='sigmoid'))
modele.summary()

""" Entraînement du modèle """
n_epochs = 30
modele.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = modele.fit(train_gen, validation_data=val_gen, epochs=n_epochs, batch_size=32)
score = modele.evaluate(train_gen, verbose=0)
score = modele.evaluate(test_gen, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



""" Visualisation des résultats """
train_perf = history.history['accuracy']
val_perf = history.history['val_accuracy']
for i in range(n_epochs):
    train_perf[i] *= 100
    val_perf[i] *= 100

plt.plot(train_perf, 'o-', label="Training")
plt.plot(val_perf, 'o-', label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend(loc="lower right")
plt.grid(alpha=.2)
plt.title("Training and Validation Accuracy")
plt.savefig("training_accuracy.png")
plt.close()

plt.plot(history.history['loss'], 'o-', label="Training Loss")
plt.plot(history.history['val_loss'], 'o-', label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.title("Training Progress")
plt.grid(alpha=.2)
plt.savefig("training_progress.png")
plt.close()


y_pred = modele.predict(test_gen)
y_true = test_data["eta_class"].values
"""
plt.figure(figsize=(10, 6))
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.scatter(y_true, y_pred, alpha=0.5, s=5)
plt.xlabel("Valeurs réelles de eta_class")
plt.ylabel("Valeurs prédites de eta_class")
plt.title("Validation des prédictions")
plt.grid(alpha=.2)
plt.savefig("validation_predictions.png")
plt.close()
"""

end_time = time.time()
print(f"Temps d'exécution : {end_time - start_time} secondes")