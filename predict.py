import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt

image_no = '00038'
model_path = "C:/Users/nitzb/Developer/CS141/final_project/segmentation_model/models/segmentation_model2.pt"
image_path = "C:/Users/nitzb/Developer/CS141/final_project/segmentation_model/data/01_images/images/" + image_no + ".png"

IMAGE_DIMS = (950, 500)

image = cv2.imread(image_path)
image = cv2.resize(image, IMAGE_DIMS, interpolation= cv2.INTER_NEAREST)

transformImg = tf.Compose([tf.ToPILImage(),tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
Net = torchvision.models.segmentation.deeplabv3_resnet50()  
Net.classifier[4] = torch.nn.Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1)) 
Net = Net.to(device)  # Set net to GPU or CPU
Net.load_state_dict(torch.load(model_path)) # Load trained model

Net.eval()

to_segment = transformImg(image)
to_segment = torch.autograd.Variable(to_segment, requires_grad=False).to(device).unsqueeze(0)
with torch.no_grad():
    prediction = Net(to_segment)['out']  # Run net

prediction = tf.Resize((IMAGE_DIMS[1], IMAGE_DIMS[0]))(prediction[0])
seg = torch.argmax(prediction, 0).cpu().detach().numpy()
cv2.imshow("yay", seg.astype(np.uint8))
cv2.imshow("yay2", image)
plt.imshow(seg)  # display image
plt.show()
cv2.waitKey()