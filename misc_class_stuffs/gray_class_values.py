# the particular model we are using for segmentation uses a gray image as the segmentation map.
# Thus, (eventually) we have to convert class <- camvid color <- PfD: Ground Truth color <- gray color

import cv2
import numpy as np

# takes pix (b, g, r) and returns grayscale value (0-255)
def pix_to_gray(pix):
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    img[0, 0] = pix
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img[0, 0]

def read_camvid_map(filename):
    color_to_class = {}
    with open(filename) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line[:len(line) - 1]
            split = line.split(' ')
            color_to_class[split[3]] = [int(split[2]), int(split[1]), int(split[0])]
    return color_to_class

BGR_to_camvid = read_camvid_map("camvid.txt")
# reads camvid classes to color map
# reads camvid color to PfD: Ground Truth color (?) map

# create PfD to gray map

# determine class to gray map (by doing PdD_to_gray[camvid_color_to_PfD_color[camvid_class_to_color["person"]]])