import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from driverNet import driverNet

MODEL_NAME = "driver_model.pt"
GRAPH_LOSS = True
NUM_DATA = 10
EPOCHS = 1000
#         (width x height)
IMAGE_DIMS = (950, 500)
LEARNING_RATE = 1e-5
BATCH_SIZE = 3 # must be <= NUM_DATA and > 1

transformImg=tf.ToTensor()

curr_dir = os.getcwd()
data_dir = os.path.abspath("../data/01_labels_reduced_classes")
model_path = os.path.abspath("../models/" +  MODEL_NAME)

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

    # cv2.imshow("a", image * 50)
    # cv2.waitKey()

    image_t = transformImg(image)

    return image_t, controls


def bake_batch():
    images = torch.zeros([BATCH_SIZE,1,IMAGE_DIMS[1],IMAGE_DIMS[0]])
    controls = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in range(BATCH_SIZE):
        images[i], controls[i] = get_random_datum()

    return images, controls


data = read_in_data(data_dir, NUM_DATA, IMAGE_DIMS)
image, controls = get_random_datum()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = driverNet(numChannels=1, numClasses=3)
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
lossFn = torch.nn.L1Loss()
model.train()
pred = 0
control = 0
for ep in tqdm(range(EPOCHS)):
    if ep % 100 == 0:
        print(pred)
        print(control)
        print("saving to " + model_path)
        torch.save(model.state_dict(), model_path)

    images, controls = bake_batch()
    for i in range(BATCH_SIZE):
        im = images[i]
        control = controls[i]
        # im.to(device)
        # control.to(device)
        im2 = torch.autograd.Variable(im, requires_grad=False).to(device)
        control2 = torch.autograd.Variable(control, requires_grad=False).to(device)

        pred = model(im2)
        loss = lossFn(pred, control2)


        opt.zero_grad()
        loss.backward()
        opt.step()
    
    # images.to(device)
    # controls.to(device)
    # pred = model(images)
    # loss = lossFn(pred, controls)

    # print(pred.shape)

    # opt.zero_grad()
    # loss.backward()
    # opt.step()
