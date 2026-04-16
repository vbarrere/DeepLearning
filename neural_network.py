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
data = data.head(100000)
data["image_file"] = data["i_sim"] + '.png'
data["eta_parameter"] = 2 * np.abs(data["nat1_out"] / (data["nat1_out"] + data["nat2_out"]) - data["nat1"] / data["n_atoms"]) + 2 * np.abs(data["nat1_in"] / (data["nat1_in"] + data["nat2_in"]) - data["nat1"] / data["n_atoms"]) - data["d_com"] / (2*data["gyration_radius"])
mask = np.isnan(data["eta_parameter"])
data = data[~mask]
#data["eta_parameter"] = (data["eta_parameter"] - data["eta_parameter"].min()) / (data["eta_parameter"].max() - data["eta_parameter"].min())



""" Division des données en ensembles d'entraînement, de validation et de test """

#train_data, test_data = train_test_split(data, test_size=0.2)
#train_data, val_data = train_test_split(train_data, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(data["image_file"], data["eta_parameter"], train_size=0.6)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
print(f"Nombre d'échantillons d'entraînement : {len(X_train)}")
print(f"Nombre d'échantillons de validation : {len(X_val)}")
print(f"Nombre d'échantillons de test : {len(X_test)}")

print("Nombre total d'échantillons : ", len(data))
print("Nombre d'échantillons utilisés : ", len(X_train) + len(X_val) + len(X_test))
exit("Fin du prétraitement des données")

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
modele.add(tf.keras.layers.Dense(1, activation='linear'))

modele.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
print(modele.summary())
history = modele.fit(train_gen, validation_data=val_gen, epochs=50)
#score = modele.evaluate(train_gen, verbose=0)
score = modele.evaluate(test_gen, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Accuracy/Loss")
plt.legend(loc="upper right")
plt.title("Training Progress")
plt.grid(alpha=.2)
plt.savefig("training_progress.png")
plt.close()

y_pred = modele.predict(test_gen)
y_true = test_data["eta_parameter"].values

plt.figure(figsize=(10, 6))
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.scatter(y_true, y_pred, alpha=0.5, s=5)
plt.xlabel("Valeurs réelles de eta")
plt.ylabel("Valeurs prédites de eta")
plt.title("Validation des prédictions")
plt.grid(alpha=.2)
plt.savefig("validation_predictions.png")
plt.close()

end_time = time.time()
print(f"Temps d'exécution : {end_time - start_time} secondes")