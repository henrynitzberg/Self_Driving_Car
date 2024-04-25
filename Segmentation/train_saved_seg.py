# Script to continue training an existing segmentation model
# Notes: datasets, class numbers, and epochs must be edited manually
import os
import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import train_new_seg as train
from tqdm import tqdm

CURR_DIR = os.getcwd()
IMGS_DIR = "01_images"
LABELS_DIR = "01_labels_reduced_classes"
EPOCHS = 10
GRAPH_LOSS = True
NUM_DATA = 10000

def get_print_model():
    model_dict = {}
    counter = 1
    print("")
    print("MODELS:")
    for model in os.listdir(os.path.abspath("../models")):
        print("[" + str(counter) + "] "  + model)
        model_dict[counter] = model
        counter += 1
    print("")
    model_num = input("Which model (input digit): ")
    return model_dict[int(model_num)]

model = os.path.abspath("../models/" + get_print_model())
print("using: " + model)

imgs_dir = os.path.abspath("../data/" + IMGS_DIR)
labels_dir = os.path.abspath("../data/" + LABELS_DIR)

images = train.read_images_from_dir(imgs_dir, NUM_DATA, train.IMAGE_DIMS)
labels = train.read_images_from_dir(imgs_dir, NUM_DATA, train.IMAGE_DIMS)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Net = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
Net.classifier[4] = torch.nn.Conv2d(256, 6, 1)
Net.aux_classifier[4] = torch.nn.Conv2d(256, 6, 1)
Net=Net.to(device)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=train.LEARNING_RATE) # Create adam optimizer
criterion = torch.nn.CrossEntropyLoss() # Set loss function
Net.load_state_dict(torch.load(model))

iteration_list = []
loss_list = []

# gets random image from training data, and corresponding annotated image
def pick_random_image():
    idx = np.random.randint(0, NUM_DATA)
    img = images[idx].copy()
    labeled = labels[idx].copy().astype(np.float32)

    unique_labels = np.unique(labeled)

    AnnMap = np.zeros(img.shape[0:2], np.float32)
    for label in unique_labels:
        AnnMap[labeled == label] = label

    img = train.transformImg(img)
    AnnMap = train.transformLab(AnnMap)

    return img, AnnMap

# returns a batch: a list images and a corresponding list of annotated images
def bake_batch():
    imgs = torch.zeros([train.BATCH_SIZE,3,train.IMAGE_DIMS[1],train.IMAGE_DIMS[0]])
    # images = [None] * BATCH_SIZE
    lbs = torch.zeros([train.BATCH_SIZE,train.IMAGE_DIMS[1],train.IMAGE_DIMS[0]])
    for i in range(train.BATCH_SIZE):
        imgs[i], lbs[i] = pick_random_image()
        imgs[i] = imgs[i].unsqueeze(0)
    return imgs, lbs


# training
for itr in tqdm(range(EPOCHS)):
    if itr % 1000 == 0:
        print("saving to " + model)
        torch.save(Net.state_dict(), model)

    imgs, lbs = bake_batch()
    imgs2 = torch.autograd.Variable(imgs, requires_grad=False).to(device)
    lbs2 = torch.autograd.Variable(lbs, requires_grad=False).to(device)

    Pred = Net(imgs2)['out']

    Net.zero_grad()

    Loss=criterion(Pred, lbs2.squeeze(1).long()) # Calculate cross entropy loss
    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient descent change to weight

    iteration_list.append(itr)
    loss_list.append(Loss.item())  # Assuming Loss is a scalar tensor


print("saving to " + model)
torch.save(Net.state_dict(), model)

if GRAPH_LOSS:
    plt.plot(iteration_list, loss_list, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.legend()
    plt.show()
