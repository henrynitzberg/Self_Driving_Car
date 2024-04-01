import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from tqdm import tqdm

# TODO: delete
def show(img):
    cv2.imshow("img", img)
    cv2.waitKey()

# number of images used for training.
NUM_DATA = 50

# dimensions of images used for training (width x height)
#   original dims are 1914 x 1052
IMAGE_DIMS = (950, 500)

# params for pytorch DCNN
LEARNING_RATE = 1e-5
BATCH_SIZE = 3 # must be < NUM_DATA

# transforms applied to the images and labels before forward pass of DCNN
transformImg=tf.Compose([tf.ToPILImage(),tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transformLab=tf.Compose([tf.ToPILImage(), tf.ToTensor()])

# TODO: don't hardcode paths
labels_dir = "C:/Users/nitzb/Developer/CS141/final_project/segmentation_model/data/labels"
imgs_dir = "C:/Users/nitzb/Developer/CS141/final_project/segmentation_model/data/imgs"
images = []
# the labels are really segmentation maps, but I call them labels to be less verbose
labels = []

# reads in specified number of images from a given directory, and 
# optionally transforms them all to specified dimensions
# If the directory contains < num_images images, all of the images from the directory are read in
# returns a list of np matrices
def read_images_from_dir(directory, num_images, image_dims=None, to_gray=False):
    images = []
    counter = num_images
    for filename in os.listdir(directory):
        filename = str(directory + "/" + filename)
        img = cv2.imread(filename)
        if (image_dims != None):
            img = cv2.resize(img, image_dims, interpolation= cv2.INTER_NEAREST)
        if (to_gray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        images.append(img)
        counter -= 1
        if (counter == 0):
            break

    return images

# TODO: download actual images
# imgs = read_images_from_dir(imgs_dir, NUM_DATA, IMAGE_DIMS)
labels = read_images_from_dir(labels_dir, NUM_DATA, image_dims=IMAGE_DIMS, to_gray=True)

# gets random image from training data, and corresponding annotated image
# TODO: perform transformations
def pick_random_image():
    idx = np.random.randint(0, NUM_DATA)
    # img = transformImg(imgs[idx])
    # labeled = transformLab(labels[idx])

    img = None
    labeled = labels[idx]
    return (img, labeled)

# returns a batch: a list images and a corresponding list of annotated images
def bake_batch():
    # images = torch.zeros([BATCH_SIZE,3,IMAGE_DIMS[1],IMAGE_DIMS[0]])
    images = [None] * BATCH_SIZE
    labels = torch.zeros([BATCH_SIZE,IMAGE_DIMS[1],IMAGE_DIMS[0]])
    for i in range(BATCH_SIZE):
        images[i], labels[i] = pick_random_image()
    return images, labels

# Defining Network
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Net = torchvision.models.segmentation.deeplabv3_resnet50()
Net.classifier[4] = torch.nn.Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 32 classes
Net=Net.to(device)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=LEARNING_RATE) # Create adam optimizer

# training
