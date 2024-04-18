import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
from driverNet import driverNet

X = 2
randnum = np.random.randint(1, X)

curr_dir = os.getcwd()
model_path = os.path.join(curr_dir, "models/driver_model.pt")
image_path = os.path.join(curr_dir, f"data/01_labels_reduced_classes/00001.png")

IMAGE_DIMS = (950, 500)
transformImg = tf.ToTensor()

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, IMAGE_DIMS, interpolation=cv2.INTER_NEAREST)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = driverNet(numChannels=1, numClasses=3).to(device)

if (device == torch.device('cpu')):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Load trained model
else:
    model.load_state_dict(torch.load(model_path)) # Load trained model

model.eval()

to_eval = transformImg(image)

controls = model(to_eval)

print(controls)
