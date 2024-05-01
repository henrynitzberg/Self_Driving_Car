import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt

# picks random image in the first X data to segment
X = 2500
# randnum = np.random.randint(1, X)
randnum = 439 

model_path = os.path.abspath("../models/segmentation_model_4class.pt")
image_path = os.path.abspath(f"../data/01_images/{randnum:05d}.png")
# image_path = os.path.abspath(f"../data/CARLA_toSeg/rgb/1714229204409982800_0.450_0.750_0.000.png")

IMAGE_DIMS = (475, 250)
NUM_CLASSES = 5

image = cv2.imread(image_path)

image = cv2.resize(image, IMAGE_DIMS, interpolation=cv2.INTER_NEAREST)
image = 

transformImg = tf.Compose([tf.ToPILImage(),tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
Net = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
Net.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, 1)
Net.aux_classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, 1)
Net = Net.to(device)  # Set net to GPU or CPU
if (device == torch.device('cpu')):
    Net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Load trained model
else:
    Net.load_state_dict(torch.load(model_path)) # Load trained model

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
seg = torch.argmax(prediction, 0).cpu().detach().numpy()
cv2.imshow("original image", image)
cv2.imshow("gray segmentation map", (seg * 30).astype(np.uint8))

