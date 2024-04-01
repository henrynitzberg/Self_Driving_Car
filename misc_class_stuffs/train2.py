import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat

# # Define the frame ID
# frame_id = 00000
# frame_id = 02500

mapping_data = loadmat('./read_mapping/mapping.mat')
mapping = {
    'cityscapesMap': mapping_data['cityscapesMap'],
    'camvidMap': mapping_data['camvidMap'],
    'classes': mapping_data['classes']
}

for frame_id in range(1, 2501):

    # Make filenames from frame ID
    # image_filename = f'../data/images/{frame_id:05d}.png'
    label_filename = f'./data/labels/{frame_id:05d}.png'

    # Read image
    # img = np.array(Image.open(image_filename))

    # Read labels and mapping
    labels = np.array(Image.open(label_filename))
    # mapping = np.load('./read_mapping/mapping.mat', allow_pickle=True).item()  # Assuming mapping is saved as a numpy array

    # Plot images
    plt.figure()
    # plt.imshow(img)
    plt.title(f'Image {frame_id}')

    plt.figure()
    plt.imshow(labels, cmap='viridis', vmin=0, vmax=len(mapping['cityscapesMap']) - 1)
    plt.title('Labels (CityScapes colors)')

    plt.figure()
    plt.imshow(labels, cmap='viridis', vmin=0, vmax=len(mapping['camvidMap']) - 1)
    plt.title('Labels (CamVid colors)')

    plt.show()
