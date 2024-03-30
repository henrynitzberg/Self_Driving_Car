import os
import cv2

NUM_DATA = 50
# TODO: don't hardcode
labels_dir = "C:/Users/nitzb/Developer/CS141/final_project/segmentation_model/data/labels"
imgs_dir = "C:/Users/nitzb/Developer/CS141/final_project/segmentation_model/data/imgs"
images = []
labels = []

def read_images_from_dir(directory, num_images):
    images = []
    counter = num_images
    for filename in os.listdir(directory):
        filename = str(directory + "/" + filename)
        img = cv2.imread(filename)
        images.append(img)
        counter -= 1
        if (counter == 0):
            break

    return images

# imgs = read_images_from_dir(imgs_dir, NUM_DATA)
labels = read_images_from_dir(labels_dir, NUM_DATA)

# cv2.imshow("yay", labels[0])
# cv2.waitKey()