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

    # Create a mask for pixels that meet the condition
    condition_mask = np.logical_and(blue_channel == 142, green_channel == 0, red_channel == 0)
    condition_mask |= np.logical_and(blue_channel == 70, green_channel == 0, red_channel == 0)

    # Create the gray image with the mask applied
    gray_image = np.zeros_like(blue_channel, dtype=np.uint8)
    gray_image[condition_mask] = 1

    return gray_image

# TODO: cars, trucks, street, sidewalk, sky

num_images = 2500
labels_dir = "C:/Users/nitzb/Developer/CS141/final_project/Self_Driving_Car/data/01_labels"
write_dir = "C:/Users/nitzb/Developer/CS141/final_project/Self_Driving_Car/data/01_labels_vehicles_only"
images = []
only_cars_and_trucks = []

images = read_images_from_dir(labels_dir, num_images)

counter = 1
for image in tqdm(images):
    gray_two_class = cars_trucks_only(image)
    only_cars_and_trucks.append(gray_two_class)

for image in only_cars_and_trucks:
    cv2.imwrite(write_dir + "/" + f'{counter:05d}' + ".png", image)
    counter += 1