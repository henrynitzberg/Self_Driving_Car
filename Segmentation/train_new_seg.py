import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# name of trained model, option to print loss every 50 epochs, and option
# to graph loss at the end of training
MODEL_NAME = "segmentation_model_5class-2.pt"
PRINT_LOSS = False
GRAPH_LOSS = True

# number of images used for training.
NUM_DATA = 30000

# number of batches to train on
EPOCHS = 50000

# dimensions used for training (width x height)
# (I think) smaller dimensions will generally yield a faster model
IMAGE_DIMS = (475, 250)
NUM_CLASSES = 5

# params for pytorch DCNN
LEARNING_RATE = 1e-5
BATCH_SIZE = 3 # must be <= NUM_DATA and > 1

# transforms applied to the images and labels before forward pass of DCNN
transformImg=tf.Compose([tf.ToPILImage(),tf.ToTensor(),
                         tf.Normalize((0.485, 0.456, 0.406), 
                                      (0.229, 0.224, 0.225))])
transformLab=tf.Compose([tf.ToTensor()])

labels_dir = os.path.abspath("../data/CARLA_labels_4class")
imgs_dir = os.path.abspath("../data/CARLA_toSeg/rgb")
models_dir = os.path.abspath("../models")

# reads in specified number of images from a given directory, and 
# optionally transforms them all to specified dimensions
# optionally reads images in grayscale (height, width) rather than (3, height, width)
# If the directory contains < num_images images, all of the images from the directory are read in
# returns a list of openCV images (np arrays)
def read_images_from_dir(directory, num_images, image_dims=None, read_gray=False):
    filelist = []
    images = []
    counter = num_images
    for filename in (os.listdir(directory)):
        filelist.append(filename)

    for filename in tqdm(sorted(filelist)):
        filename = str(directory + "/" + filename)
        if read_gray:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(filename)
        if (image_dims != None):
            img = cv2.resize(img, image_dims, interpolation= cv2.INTER_NEAREST)
        images.append(img)
        counter -= 1
        if (counter == 0):
            break

    return images.copy()

labels = read_images_from_dir(labels_dir, NUM_DATA, image_dims=IMAGE_DIMS, read_gray=True)
images = read_images_from_dir(imgs_dir, NUM_DATA, IMAGE_DIMS)

# gets random image from training data, and corresponding annotated image
def pick_random_image():
    idx = np.random.randint(0, len(images))
    img = images[idx].copy()
    labeled = labels[idx].copy().astype(np.float32)

    unique_labels = np.unique(labeled)

    AnnMap = np.zeros(img.shape[0:2], np.float32)
    for label in unique_labels:
        AnnMap[labeled == label] = label

    img = transformImg(img)
    AnnMap = transformLab(AnnMap)

    return img, AnnMap

# returns a batch: a list images and a corresponding list of annotated images
def bake_batch():
    imgs = torch.zeros([BATCH_SIZE,3,IMAGE_DIMS[1],IMAGE_DIMS[0]])
    # images = [None] * BATCH_SIZE
    lbs = torch.zeros([BATCH_SIZE,IMAGE_DIMS[1],IMAGE_DIMS[0]])
    for i in range(BATCH_SIZE):
        imgs[i], lbs[i] = pick_random_image()
        imgs[i] = imgs[i].unsqueeze(0)
    return imgs, lbs

# Defining Network
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Net = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
Net.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, 1)
Net.aux_classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, 1)
Net=Net.to(device)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=LEARNING_RATE) # Create adam optimizer
criterion = torch.nn.CrossEntropyLoss() # Set loss function

iteration_list = []
loss_list = []

# training
for itr in tqdm(range(EPOCHS)):
    if itr % 1000 == 0:
        print("saving to " + models_dir + "/" + MODEL_NAME)
        torch.save(Net.state_dict(), models_dir + "/" + MODEL_NAME)

    imgs, lbs = bake_batch()
    imgs2 = torch.autograd.Variable(imgs, requires_grad=False).to(device)
    lbs2 = torch.autograd.Variable(lbs, requires_grad=False).to(device)

    Pred = Net(imgs2)['out']

    Net.zero_grad()

    Loss=criterion(Pred, lbs2.squeeze(1).long()) # Calculate cross entropy loss
    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient descent change to weight

    if (itr % 50 == 0) and PRINT_LOSS:
        print("Step: " + str(itr) + " Loss: " + str(Loss.item()))

    iteration_list.append(itr)
    loss_list.append(Loss.item())  # Assuming Loss is a scalar tensor


print("saving to " + models_dir + "/" + MODEL_NAME)
torch.save(Net.state_dict(), models_dir + "/" + MODEL_NAME)

if GRAPH_LOSS:
    plt.plot(iteration_list, loss_list, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.legend()
    plt.show()