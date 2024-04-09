import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

MODEL_NAME = "driver_model.pt"
GRAPH_LOSS = True
NUM_DATA = 10
EPOCHS = 1000
#         (width x height)
IMAGE_DIMS = (950, 500)
LEARNING_RATE = 1e-5
BATCH_SIZE = 3 # must be <= NUM_DATA and > 1

transformImg=tf.Compose([tf.ToPILImage(),tf.ToTensor(),
                         tf.Normalize(0.485,
                                      0.229)])
# transformControls=tf.Compose([tf.ToTensor()])

curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, "data/01_labels_reduced_classes")
model_path = os.path.join(curr_dir, "models", MODEL_NAME)

# should return an array of tuples [(image, np.array([steering_angle, accel_value]))]
def read_in_data(dir, num_data, image_dims):
    data_list = []
    counter = num_data
    for filename in os.listdir(data_dir):
        image = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
        # TODO: get real values
        controls = np.array([0, 0])
        data_list.append((image, controls))
        counter -= 1
        if counter == 0:
            break
    return data_list

data = read_in_data(data_dir, NUM_DATA, IMAGE_DIMS)

def get_random_datum():
    rand_idx = np.random.randint(0, NUM_DATA)
    image, controls = data[rand_idx]

    image_t = transformImg(image)

    return image_t, controls

image, controls = get_random_datum()
