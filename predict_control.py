# Notes when imported:
# you will need driverNet.py in the same directory, on the same level
# you will need the model 'driver_model.pt' in the 'models' directory 
# (that is, you will need a ./models/driver_model.pt)

import os
import numpy as np
import cv2
import torch
import torchvision.transforms as tf
from driverNet import driverNet

curr_dir = os.getcwd()
model_path = os.path.join(curr_dir, "models/driver_model-2.pt")
# image_path = os.path.join(curr_dir, f"data/01_controls/0.178_0.690_0.690.png")

IMAGE_DIMS = (475, 250)
transformImg = tf.ToTensor()

# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# to_show = image.copy() * 50

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = driverNet(numChannels=1, numClasses=3).to(device)

if (device == torch.device('cpu')):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Load trained model
else:
    model.load_state_dict(torch.load(model_path)) # Load trained model

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
    person_mask = np.logical_and(blue_channel == 60, green_channel == 20, red_channel == 220)
    # traffic_light (30, 170, 250)
    traffic_light_mask = np.logical_and(blue_channel == 30, green_channel == 170, red_channel == 250)

    gray_image = np.zeros_like(blue_channel, dtype=np.uint8)
    gray_image[vehicles_mask] = 1
    gray_image[road_mask] = 2
    gray_image[sidewalk_mask] = 3
    gray_image[person_mask] = 4
    gray_image[traffic_light_mask] = 5

    return gray_image

def predict(img, CARLA=False):
    img = cv2.resize(img, IMAGE_DIMS, interpolation=cv2.INTER_NEAREST)
    if CARLA:
        img = to_gray(img)
    img = transformImg(img)
    img = img.to(device)

    control_T = model(img)

    return [round(control_T[0].item(), 3), round(control_T[1].item(), 3), round(control_T[2].item(), 3)]

# print(predict(image))

# print(controls)
# print((0.178, 0.690, 0.690))
# cv2.imshow("segd image", to_show)
# cv2.waitKey()
