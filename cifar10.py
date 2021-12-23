import os
import cv2
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

label_name = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

import glob

train_list = glob.glob("/Users/liding/Documents/pytorch_py/train/data_batch_*")
print(train_list)

for l in train_list:
    print(l)
    l_dict = unpickle(l)
    print(l_dict)