from HandNet import HandNet, HandNet_2
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical

import numpy as np
import argparse
import pickle
import cv2
import os

#dataset_dir = ""
model_dir = ""

model = HandNet_2.build(width=64, height=64, depth=3, classes= 26 )
model.summary()
model.save('untrained_HandNet_2.h5')