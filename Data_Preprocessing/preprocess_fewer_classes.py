import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from tqdm import tqdm

ALL_LABELS = False
NUM_IMAGES = 100000

def read_images_from_dir(directory, num_images):
    images = []
    names = []
    counter = num_images
    for filename in tqdm(os.listdir(directory)):
        absFilePath = str(directory + "/" + filename)
        img = cv2.imread(absFilePath)
        images.append(img)
        names.append(filename)
        counter -= 1
        if (counter == 0):
            break

    return names, images

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
    # person_mask = np.logical_and(blue_channel == 60, green_channel == 20, red_channel == 220)
    # traffic_light (30, 170, 250)
    # traffic_light_mask = np.logical_and(blue_channel == 30, green_channel == 170, red_channel == 250)
    # lane markings (50, 234, 157)
    lane_marking_mask = np.logical_and(blue_channel == 50, green_channel == 234, red_channel == 157)

    # 6 classes: (0=void,...,5=traffic_light)
    gray_image = np.zeros_like(blue_channel, dtype=np.uint8)
    gray_image[vehicles_mask] = 1
    gray_image[road_mask] = 2
    gray_image[sidewalk_mask] = 3
    gray_image[lane_marking_mask] = 4
    # gray_image[person_mask] = 4
    # gray_image[traffic_light_mask] = 5

    return gray_image

labels_dir_1 = os.path.abspath("../data/CARLA_toSeg/sem")
if ALL_LABELS:
    labels_dir_2 = os.path.abspath("../data/02_labels/labels")
    labels_dir_3 = os.path.abspath("../data/03_labels/labels")
write_dir = os.path.abspath("../data/CARLA_labels_4class")
images = []
names = []

names, images = read_images_from_dir(labels_dir_1, NUM_IMAGES)

if ALL_LABELS:
    images = images + read_images_from_dir(labels_dir_2, NUM_IMAGES) + read_images_from_dir(labels_dir_3, NUM_IMAGES)

counter = 1
for i in tqdm(range(len(images))):
    image = reduce_classes(images[i])
    cv2.imwrite(write_dir + "/" + names[i], image)
    counter += 1
