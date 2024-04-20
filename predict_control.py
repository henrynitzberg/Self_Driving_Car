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
model_path = os.path.join(curr_dir, "models/driver_model.pt")
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

def predict(img):
    img = cv2.resize(image, IMAGE_DIMS, interpolation=cv2.INTER_NEAREST)
    img = transformImg(img)
    img = img.to(device)

    control_T = model(img)

    return [round(control_T[0].item(), 3), round(control_T[1].item(), 3), round(control_T[2].item(), 3)]

# print(predict(image))

# print(controls)
# print((0.178, 0.690, 0.690))
# cv2.imshow("segd image", to_show)
# cv2.waitKey()
