import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from tqdm import tqdm

NUM_IMAGES = 30000

def read_images_from_dir(directory, num_images):
    images = []
    counter = num_images
    for filename in tqdm(os.listdir(directory)):
        wholeFilename = str(directory + "/" + filename)
        img = cv2.imread(wholeFilename)
        images.append((filename, img))
        counter -= 1
        if (counter == 0):
            break

    return images

def name_as_control(filename):
    name_arr = filename.split("_")
    name_arr[3] = name_arr[3][:-4]
    control = [float(name_arr[1]), float(name_arr[2]), float(name_arr[3])]
    return control

def avg_steering(data):
    avg = 0.0
    if len(data) == 0:
        return .5
    for control, _ in data:
        avg += control[0]
    return avg / len(data)

def adjust_average(images):
    thresh = .01
    lefts = []
    straights = []
    rights = []
    for filename, image in images:
        control = name_as_control(filename)
        if control[0] < .5:
            lefts.append((control, image))
        elif control[0] > .5:
            rights.append((control, image))
        elif control[0] == .5:
            straights.append((control, image))
    
    if ((avg_steering(rights) * len(rights) + avg_steering(lefts) * len(lefts)) / (len(lefts) + len(rights))) < .5 - thresh:
        # remove lefts until within thresh
        while ((avg_steering(rights) * len(rights) + avg_steering(lefts) * len(lefts)) / (len(lefts) + len(rights))) < .5 - thresh:
            randidx = np.random.randint(len(lefts) - 1)
            lefts.pop(randidx)
    elif ((avg_steering(rights) * len(rights) + avg_steering(lefts) * len(lefts)) / (len(lefts) + len(rights))) > .5 + thresh:
        # remove rights until within thresh
        while ((avg_steering(rights) * len(rights) + avg_steering(lefts) * len(lefts)) / (len(lefts) + len(rights))) > .5 + thresh:
            randidx = np.random.randint(len(rights) - 1)
            rights.pop(randidx)
    
    data = []
    for control, image in lefts:
        data.append((control_to_filename(control), image))
    for control, image in rights:
        data.append((control_to_filename(control), image))
    for control, image in straights:
        data.append((control_to_filename(control), image))

    print((avg_steering(rights) * len(rights) + avg_steering(lefts) * len(lefts)) / (len(lefts) + len(rights)))

    return data

    



def control_to_filename(control):
    new_name = str(control[0]) + "_" + str(control[1]) + "_" + str(control[2]) + ".png"
    return new_name

def convert_filename(filename):
    name_arr = filename.split("_")
    new_name = str(name_arr[1]) + "_" + str(name_arr[2]) + "_" + str(name_arr[3])
    return new_name


def convert_to_gray(filename, img, convert_name=False):
    new_name = filename
    if convert_name:
        new_name = convert_filename(filename)
    
    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]

    # Create a mask for pixels that meet the condition (B, G, R)
    # cars (142, 0, 0)
    vehicles_mask = np.logical_and(blue_channel == 142, green_channel == 0, red_channel == 0)
    # road (128, 64, 128)
    road_mask = np.logical_and(blue_channel == 128, green_channel == 64, red_channel == 128)
    road_mask |= np.logical_and(blue_channel == 50, green_channel == 234, red_channel == 157)
    # sidewalk (232, 35, 244)
    sidewalk_mask = np.logical_and(blue_channel == 232, green_channel == 35, red_channel == 244)
    # person (60, 20, 220)
    person_mask = np.logical_and(blue_channel == 60, green_channel == 20, red_channel == 220)
    # traffic_light (30, 170, 250)
    traffic_light_mask = np.logical_and(blue_channel == 30, green_channel == 170, red_channel == 250)
    # lane markings (50, 234, 157)
    lane_marking_mask = np.logical_and(blue_channel == 50, green_channel == 234, red_channel == 157)

    gray_image = np.zeros_like(blue_channel, dtype=np.uint8)
    gray_image[vehicles_mask] = 1
    gray_image[road_mask] = 2
    gray_image[sidewalk_mask] = 3
    gray_image[person_mask] = 4
    gray_image[traffic_light_mask] = 5
    gray_image[lane_marking_mask] = 6

    return new_name, gray_image


labels_dir_1 = os.path.abspath("../data/sem_town4_2")
write_dir = os.path.abspath("../data/03_controls")

images = []
only_cars_and_trucks = []

images = read_images_from_dir(labels_dir_1, NUM_IMAGES)
images = adjust_average(images)
counter = 1
for filename, image in tqdm(images):
    filename, image = convert_to_gray(filename, image)
    cv2.imwrite(write_dir + "/" + str(counter) + "_" + filename, image)
    counter += 1
