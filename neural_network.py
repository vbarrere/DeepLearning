#!/usr/bin/python3

import os
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

img_path = "../Data/Data_HRTEM/HRTEM_image"
data = pd.read_csv("../Data/Data_processed/data.dat", sep="\t", na_values=["nan"])
