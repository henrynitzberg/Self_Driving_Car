import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from driverNet import driverNet

MODEL_NAME = "driver_model-3.pt"
GRAPH_LOSS = True
NUM_DATA = 30000
EPOCHS = 300000
#         (width x height)
IMAGE_DIMS = (475, 250)
LEARNING_RATE = 1e-5
BATCH_SIZE = 3 # must be <= NUM_DATA and > 1

transformImg=tf.Compose([tf.ToPILImage(),tf.ToTensor()])

curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, "data/02_controls")
model_path = os.path.join(curr_dir, "models", MODEL_NAME)

# should return an array of tuples [(image, torch.tensor([steering, accel_value, brake]))]
def read_in_data(dir, num_data, image_dims):
    data_list = []
    counter = num_data
    for filename in tqdm(os.listdir(dir)):
        image = cv2.imread(os.path.join(dir, filename), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, image_dims, interpolation=cv2.INTER_NEAREST)
        carr = filename.split('_')
        carr[3] = carr[3][:-4]
        controls = torch.tensor([float(carr[1]), float(carr[2]), float(carr[3])])
        data_list.append((image, controls))
        counter -= 1
        if counter == 0:
            break
    return data_list

data = read_in_data(data_dir, NUM_DATA, IMAGE_DIMS)

def get_random_datum():
    rand_idx = np.random.randint(0, len(data))
    image, controls = data[rand_idx]

    # cv2.imshow("a", image * 50)
    # cv2.waitKey()

    image_t = transformImg(image)

    return image_t, controls


def bake_batch():
    images = torch.zeros([BATCH_SIZE,1,IMAGE_DIMS[1],IMAGE_DIMS[0]])
    controls = torch.zeros([BATCH_SIZE, 3])
    for i in range(BATCH_SIZE):
        images[i], controls[i] = get_random_datum()

    return images, controls



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = driverNet(numChannels=1, numClasses=3)
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
lossFn = torch.nn.L1Loss()
model.train()

iteration_list = []
loss_list = []

for ep in tqdm(range(EPOCHS)):

    images, controls = bake_batch()
    for i in range(BATCH_SIZE):
        im = images[i]
        control = controls[i]

        im2 = torch.autograd.Variable(im, requires_grad=False).to(device)
        control2 = torch.autograd.Variable(control, requires_grad=False).to(device)

        pred = model(im2)
        loss = lossFn(pred, control2)

        opt.zero_grad()
        loss.backward()
        opt.step()
        iteration_list.append(ep)
        loss_list.append(loss.item())  # Assuming Loss is a scalar tensor
    
    if ep % 5000 == 0:
        print(pred)
        print(control)
        print("saving to " + model_path)
        torch.save(model.state_dict(), model_path)

plt.plot(iteration_list, loss_list, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Iterations')
plt.legend()
plt.show()
