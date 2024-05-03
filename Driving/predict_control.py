# Notes when imported:
# you will need driverNetMk1.py in the same directory, on the same level
# you will need the model 'driver_model.pt' in the 'models' directory
# (that is, you will need a ../models/driver_model.pt)

import os
import numpy as np
import cv2
import torch
import torchvision.transforms as tf
from driverNetMk1 import driverNetMk1

IMAGE_DIMS = (475, 250)
transformImg = tf.ToTensor()
MODEL_PATH = os.path.abspath("../models/highway_driver_traffic-2.pt")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = driverNetMk1(numChannels=1, numClasses=3).to(device)

if (device == torch.device('cpu')):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))) # Load trained model
else:
    model.load_state_dict(torch.load(MODEL_PATH)) # Load trained model

model.eval()

def to_gray(img):
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
    # person_mask = np.logical_and(blue_channel == 60, green_channel == 20, red_channel == 220)
    # traffic_light (30, 170, 250)
    # traffic_light_mask = np.logical_and(blue_channel == 30, green_channel == 170, red_channel == 250)
    # lane markings (50, 234, 157)
    lane_marking_mask = np.logical_and(blue_channel == 50, green_channel == 234, red_channel == 157)

    gray_image = np.zeros_like(blue_channel, dtype=np.uint8)
    gray_image[vehicles_mask] = 1
    gray_image[road_mask] = 2
    gray_image[sidewalk_mask] = 3
    # gray_image[person_mask] = 4
    # gray_image[traffic_light_mask] = 5
    gray_image[lane_marking_mask] = 4

    return gray_image

# takes 2D np array representing a segmented image
# returns control as list [steering, acceleration, brake] with each value [0 - 1]
def predict(img, CARLA=False):
    img = cv2.resize(img, IMAGE_DIMS, interpolation=cv2.INTER_NEAREST)
    if CARLA:
        img = to_gray(img)
    img = transformImg(img)
    img = img.to(device)

    control_T = model(img)

    return [round(control_T[0].item(), 3), round(control_T[1].item(), 3), round(control_T[2].item(), 3)]

