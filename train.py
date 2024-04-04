import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# TODO: delete
def show(img):
    cv2.imshow("img", img)
    cv2.waitKey()

# number of images used for training.
NUM_DATA = 1000

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
# Note: if loading images from multiple label directories, can concat lists with '+'
models_dir = "C:/Users/nitzb/Developer/CS141/final_project/Self_Driving_Car/models"
labels_dir = "C:/Users/nitzb/Developer/CS141/final_project/Self_Driving_Car/data/01_labels_reduced_classes"
imgs_dir = "C:/Users/nitzb/Developer/CS141/final_project/Self_Driving_Car/data/01_images"

# for mac
# labels_dir = "/Users/nitz/Developer/CS141/final_project/Self_Driving_Car/data/labels"
# imgs_dir = "/Users/nitz/Developer/CS141/final_project/Self_Driving_Car/data/images"

images = []
# the labels are really segmentation maps, but I call them labels to be less verbose
labels = []

# reads in specified number of images from a given directory, and 
# optionally transforms them all to specified dimensions
# If the directory contains < num_images images, all of the images from the directory are read in
# returns a list of np matrices
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

# TODO: download actual images
images = read_images_from_dir(imgs_dir, NUM_DATA, IMAGE_DIMS)
labels = read_images_from_dir(labels_dir, NUM_DATA, image_dims=IMAGE_DIMS, read_gray=True)

# show(images[9])
# gets random image from training data, and corresponding annotated image
# TODO: perform transformations
def pick_random_image():
    idx = np.random.randint(0, NUM_DATA)
    img = images[idx].copy()
    labeled = labels[idx].copy()

    unique_labels = np.unique(labeled)

    AnnMap = np.zeros_like(labeled, dtype=np.float32)
    for label in unique_labels:
        AnnMap[labeled == label] = label

    # plt.imshow(AnnMap, cmap='viridis')
    # plt.show()
    # show(AnnMap.astype(np.uint8))

    # for testing STILL LABELS ENTIRE IMAGE AS VOID
    # height = IMAGE_DIMS[1]
    # width = IMAGE_DIMS[0]
    # labeled = np.zeros((height, width), dtype=np.uint8)
    # labeled[:,:int(width/2)] = 1

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
    return imgs, lbs

# Defining Network
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Net = torchvision.models.segmentation.deeplabv3_resnet50()
Net.classifier[4] = torch.nn.Conv2d(256, 6, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 6 classes
Net=Net.to(device)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=LEARNING_RATE) # Create adam optimizer
criterion = torch.nn.CrossEntropyLoss() # Set loss function

iteration_list = []
loss_list = []

# training
model_name = "segmentation_model_6class.pt"
for itr in tqdm(range(3000)):
    imgs, lbs = bake_batch()
    imgs = torch.autograd.Variable(imgs, requires_grad=False).to(device)
    lbs = torch.autograd.Variable(lbs, requires_grad=False).to(device)

    optimizer.zero_grad()

    Pred = Net(imgs)['out']

    Loss=criterion(Pred, lbs.long()) # Calculate cross entropy loss
    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient descent change to weight

    iteration_list.append(itr)
    loss_list.append(Loss.item())  # Assuming Loss is a scalar tensor

    if itr % 500 == 0:
        print("saving to " + models_dir + "/" + model_name)
        torch.save(Net.state_dict(), models_dir + "/" + model_name)

print("saving to " + models_dir + "/" + model_name)
torch.save(Net.state_dict(), models_dir + "/" + model_name)

plt.plot(iteration_list, loss_list, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Iterations')
plt.legend()
plt.show()