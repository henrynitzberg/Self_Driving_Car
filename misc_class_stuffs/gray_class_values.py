# the particular model we are using for segmentation uses a gray image as the segmentation map.
# Thus, we have to convert class -> camvid color -> PfD: Ground Truth color -> gray color

import cv2
import numpy as np

# reads camvid classes to color map
# reads camvid color to PfD: Ground Truth color (?) map

# create PfD to gray map

# determine class to gray map (by doing PdD_to_gray[camvid_color_to_PfD_color[camvid_class_to_color["person"]]])