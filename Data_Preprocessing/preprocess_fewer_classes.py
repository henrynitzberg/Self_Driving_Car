import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from tqdm import tqdm

ALL_LABELS = True
NUM_IMAGES = 2500

def read_images_from_dir(directory, num_images):
    images = []
    counter = num_images
    for filename in tqdm(os.listdir(directory)):
        filename = str(directory + "/" + filename)
        img = cv2.imread(filename)
        images.append(img)
        counter -= 1
        if (counter == 0):
            break

    return images

# reduces number of classes to 6:
# void, vehicles, road, sidewalk, person, traffic_light
def reduce_classes(img):
    # Extract the BGR channels
    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]

    # Create a mask for pixels that meet the condition (B, G, R)
    # cars (142, 0, 0)
    vehicles_mask = np.logical_and(blue_channel == 142, green_channel == 0, red_channel == 0)
    # trucks (70, 0, 0)
    vehicles_mask |= np.logical_and(blue_channel == 70, green_channel == 0, red_channel == 0)
    # bus (100, 60, 0)
    vehicles_mask |= np.logical_and(blue_channel == 100, green_channel == 60, red_channel == 0)
    # motorcycle (230, 0, 0)
    vehicles_mask |= np.logical_and(blue_channel == 230, green_channel == 0, red_channel == 0)
    # road (128, 64, 128)
    road_mask = np.logical_and(blue_channel == 128, green_channel == 64, red_channel == 128)
    # sidewalk (232, 35, 244)
    sidewalk_mask = np.logical_and(blue_channel == 232, green_channel == 35, red_channel == 244)
    # person (60, 20, 220)
    person_mask = np.logical_and(blue_channel == 60, green_channel == 20, red_channel == 220)
    # traffic_light (30, 170, 250)
    traffic_light_mask = np.logical_and(blue_channel == 30, green_channel == 170, red_channel == 250)


    # 6 classes: (0=void,...,5=traffic_light)
    gray_image = np.zeros_like(blue_channel, dtype=np.uint8)
    gray_image[vehicles_mask] = 1
    gray_image[road_mask] = 2
    gray_image[sidewalk_mask] = 3
    gray_image[person_mask] = 4
    gray_image[traffic_light_mask] = 5

    return gray_image

labels_dir_1 = os.path.abspath("../data/01_labels")
if ALL_LABELS:
    labels_dir_2 = os.path.abspath("../data/02_labels/labels")
    labels_dir_3 = os.path.abspath("../data/03_labels/labels")
write_dir = os.path.abspath("../data/01_labels_reduced_classes")
images = []
only_cars_and_trucks = []

images = read_images_from_dir(labels_dir_1, NUM_IMAGES)
if ALL_LABELS:
    images = images + read_images_from_dir(labels_dir_2, NUM_IMAGES) + read_images_from_dir(labels_dir_3, NUM_IMAGES)

counter = 1
for image in tqdm(images):
    image = reduce_classes(image)
    cv2.imwrite(write_dir + "/" + f'{counter:05d}' + ".png", image)
    counter += 1
