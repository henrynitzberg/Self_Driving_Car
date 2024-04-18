import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

MODEL_NAME = "driver_model.pt"
GRAPH_LOSS = True
NUM_DATA = 10
EPOCHS = 1000
#         (width x height)
IMAGE_DIMS = (950, 500)
LEARNING_RATE = 1e-5
BATCH_SIZE = 3 # must be <= NUM_DATA and > 1

transformImg=tf.Compose([tf.ToTensor])
# transformControls=tf.Compose([tf.ToTensor()])

curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, "data/01_labels_reduced_classes")
model_path = os.path.join(curr_dir, "models", MODEL_NAME)

# should return an array of tuples [(image, torch.tensor([steering, accel_value, brake]))]
def read_in_data(dir, num_data, image_dims):
    data_list = []
    counter = num_data
    for filename in os.listdir(dir):
        image = cv2.imread(os.path.join(dir, filename), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, image_dims, interpolation=cv2.INTER_NEAREST)
        # TODO: get real values
        # TODO: make sure real values are all positive
        controls = torch.tensor([0, 0, 0])
        data_list.append((image, controls))
        counter -= 1
        if counter == 0:
            break
    return data_list


def get_random_datum():
    rand_idx = np.random.randint(0, NUM_DATA)
    image, controls = data[rand_idx]

    image_t = transformImg(image)

    return image_t, controls


# Input -> 
# [conv2d, maxpool, ReLU]
# [conv2d, maxpool, ReLU]
# flatten
# [linear, ReLU]
# [linear, ReLU] 
# -> output
# TODO: write neural network class (custom)
class driverNet(torch.nn.Module):
    def __init__(self, numChannels, numClasses):
        super(driverNet, self)
        self.conv1 = torch.nn.Conv2d(in_channels=numChannels, out_channels=20, 
                                    kernel_size=(5, 5))
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize second set of CONV => RELU => POOL layers
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize first (and only) set of FC => RELU layers
        self.fc1 = torch.nn.Linear(in_features=800, out_features=500)
        self.relu3 = torch.nn.ReLU()
		# initialize our softmax classifier
        self.fc2 = torch.nnLinear(in_features=500, out_features=numClasses)
        self.relu4 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        output = self.relu4(x)
        return output


data = read_in_data(data_dir, NUM_DATA, IMAGE_DIMS)
image, controls = get_random_datum()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = driverNet(numChannels=3, classes=3).to(device)
opt = Adam(mode.parameters(), lr=LEARNING_RATE)
lossFn = nn.NLLLoss()
