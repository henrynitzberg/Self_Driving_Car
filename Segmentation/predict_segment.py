import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt

# info relevant to model (actually image dims are not fixed)
IMAGE_DIMS = (475, 250)
NUM_CLASSES = 5
MODEL_PATH = os.path.abspath("../models/segmentation_model_5class_camvid+carla-1.pt")


transformImg = tf.Compose([tf.ToPILImage(),tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
Net = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
Net.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, 1)
Net.aux_classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, 1)
Net = Net.to(device)  # Set net to GPU or CPU
if (device == torch.device('cpu')):
    Net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))) # Load trained model
else:
    Net.load_state_dict(torch.load(MODEL_PATH)) # Load trained model

Net.eval()

# takes 3D np array representing image
# returns 2D np array representing segmented version of image
def predict_segment(image):
    image = cv2.resize(image, IMAGE_DIMS, interpolation=cv2.INTER_NEAREST)
    to_segment = transformImg(image)
    to_segment = torch.autograd.Variable(to_segment, requires_grad=False).to(device).unsqueeze(0)
    with torch.no_grad():
        prediction = Net(to_segment)['out']  # Run net
    prediction = tf.Resize((IMAGE_DIMS[1], IMAGE_DIMS[0]))(prediction[0])
    prediction = torch.argmax(prediction, 0).cpu().detach().numpy().astype(np.uint8)
    return prediction


