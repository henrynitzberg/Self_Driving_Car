import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from tqdm import tqdm

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


# def cars_trucks_only(img):
#     height, width, _ = img.shape

#     gray_image = np.zeros((height, width), dtype=np.uint8)

#     for i in range(height):
#         for j in range(width):
#             pix = img[i,j]
#             if pix[2] == 0 and pix[1] == 0 and (pix[0] == 142 or pix[0] == 70):
#                 gray_image[i,j] = 1
#     return gray_image

def cars_trucks_only(img):
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
    traffic_light_mask = np.logical_and(blue_channel == 30, green_channel == 150, red_channel == 250)


    # 6 classes: (0=void,...,5=traffic_light)
    gray_image = np.zeros_like(blue_channel, dtype=np.uint8)
    gray_image[vehicles_mask] = 1
    gray_image[road_mask] = 2
    gray_image[sidewalk_mask] = 3
    gray_image[person_mask] = 4
    gray_image[traffic_light_mask] = 5

    return gray_image

# TODO: cars, trucks, street, sidewalk, sky

num_images = 2500
labels_dir_1 = "C:/Users/nitzb/Developer/CS141/final_project/Self_Driving_Car/data/01_labels"
labels_dir_2 = "C:/Users/nitzb/Developer/CS141/final_project/Self_Driving_Car/data/02_labels/labels"
labels_dir_3 = "C:/Users/nitzb/Developer/CS141/final_project/Self_Driving_Car/data/03_labels/labels"
write_dir = "C:/Users/nitzb/Developer/CS141/final_project/Self_Driving_Car/data/01_labels_reduced_classes"
images = []
only_cars_and_trucks = []

images = read_images_from_dir(labels_dir_1, num_images) + read_images_from_dir(labels_dir_2, num_images) + read_images_from_dir(labels_dir_3, num_images)

counter = 1
for image in tqdm(images):
    gray_two_class = cars_trucks_only(image)
    only_cars_and_trucks.append(gray_two_class)

for image in only_cars_and_trucks:
    cv2.imwrite(write_dir + "/" + f'{counter:05d}' + ".png", image)
    counter += 1