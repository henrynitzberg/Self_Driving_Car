# take labeled image
# convert to gray
# spread gray values out
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


gray_to_enhanced = {}
counter = 0

def enhance(img, counter):
    height, width, _ = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for i in range(height):
        for j in range(width):
            val = img[i,j]
            if val in gray_to_enhanced:
                img[i,j] = gray_to_enhanced[val]
            else:
                gray_to_enhanced[val] = counter
                counter += 1
                img[i,j] = gray_to_enhanced[val]

    return img, counter

num_images = 2500
labels_dir = "C:/Users/nitzb/Developer/CS141/final_project/segmentation_model/data/01_labels"
write_dir = "C:/Users/nitzb/Developer/CS141/final_project/segmentation_model/data/01_labels_gray"
images = []
enhanced_images = []

images = read_images_from_dir(labels_dir, num_images)
for image in tqdm(images):
    newimg, counter = enhance(image, counter)
    enhanced_images.append(newimg)

counter = 1
for image in enhanced_images:
    cv2.imwrite(write_dir + "/" + str(counter) + ".png", image)
    counter += 1
