import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from scipy.io import loadmat
from matplotlib import colors

# # Define the frame ID
# frame_id = 00000
# frame_id = 02500

def array2cmap(X):
    N = X.shape[0]

    r = np.linspace(0., 1., N+1)
    r = np.sort(np.concatenate((r, r)))[1:-1]

    rd = np.concatenate([[X[i, 0], X[i, 0]] for i in range(N)])
    gr = np.concatenate([[X[i, 1], X[i, 1]] for i in range(N)])
    bl = np.concatenate([[X[i, 2], X[i, 2]] for i in range(N)])

    rd = tuple([(r[i], rd[i], rd[i]) for i in range(2 * N)])
    gr = tuple([(r[i], gr[i], gr[i]) for i in range(2 * N)])
    bl = tuple([(r[i], bl[i], bl[i]) for i in range(2 * N)])


    cdict = {'red': rd, 'green': gr, 'blue': bl}
    return colors.LinearSegmentedColormap('my_colormap', cdict, N)

mapping_data = loadmat('./read_mapping/mapping.mat')
mapping = {
    'cityscapesMap': mapping_data['cityscapesMap'],
    'camvidMap': mapping_data['camvidMap'],
    'classes': mapping_data['classes']
}

cmap = array2cmap(mapping['camvidMap'])

for frame_id in range(1, 2):

    # Make filenames from frame ID
    # image_filename = f'../data/images/{frame_id:05d}.png'
    label_filename = f'/Users/nitz/Developer/CS141/final_project/Self_Driving_Car/data/labels/{frame_id:05d}.png'

    # Read image
    # img = np.array(Image.open(image_filename))

    # Read labels and mapping
    labels = np.array(Image.open(label_filename))
    # labels = cv2.imread(label_filename)

    # cmap=plt.cm.viridis
    # print(mapping['camvidMap'])
    vmax=len(mapping['camvidMap']) - 1
    norm = plt.Normalize(vmin=0, vmax=vmax)

    image = cmap(norm(labels))

    plt.figure()
    plt.imshow(labels, cmap, norm)
    plt.title(f'Image {frame_id}')

    cv2.imshow("?", image)
    cv2.waitKey()

    # Plot images
    # # plt.imshow(img)

    # plt.figure()
    # plt.imshow(labels, cmap='viridis', vmin=0, vmax=len(mapping['cityscapesMap']) - 1)
    # plt.title('Labels (CityScapes colors)')

    # plt.figure()
    # plt.imshow(labels, cmap='viridis', vmin=0, vmax=len(mapping['camvidMap']) - 1)
    # plt.title('Labels (CamVid colors)')

    # plt.show()
