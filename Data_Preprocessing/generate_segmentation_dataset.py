import os
import numpy as np
import cv2
from tqdm import tqdm

def read_images_from_dir(directory, seg=False):
    images = {}
    for filename in tqdm(os.listdir(directory)):
        absFilePath = str(directory + "/" + filename)
        img = cv2.imread(absFilePath)
        name = filename.split("_")[1]
        if seg:
            if name[0] == 'f':
                name = name[1:]
        else:
            if name[0] == 'f':
                name = name[1:-4]
            else:
                name = name[:-4]

        name = (int(name))
        images[name] = img

    return images

def read_images_from_dir_carla(directory):
    images = []
    for filename in tqdm(os.listdir(directory)):
        absFilePath = str(directory + "/" + filename)
        img = cv2.imread(absFilePath)
        images.append(img)

    return images

# reduces number of classes to 4: void/unlabled, vehicles, road, sidewalk
def reduce_classes(img):
    # Extract the BGR channels
    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]

    # # Create a mask for pixels that meet the condition (R, G, B)
    # Car,64, 0, 128
    vehicles_mask = (blue_channel == 128) & (green_channel == 0) & (red_channel == 64)
    # MotorcycleScooter,192, 0, 192
    vehicles_mask |= (blue_channel == 192) & (green_channel == 0) & (red_channel == 192)
    # SUVPickupTruck,64, 128,192
    vehicles_mask |= (blue_channel == 192) & (green_channel == 128) & (red_channel == 64)
    # Truck_Bus,192, 128, 192
    vehicles_mask |= (blue_channel == 192) & (green_channel == 128) & (red_channel == 192)

    # Road,128, 64, 128
    road_mask = (blue_channel == 128) & (green_channel == 64) & (red_channel == 128)
    # LaneMkgsNonDriv,192, 0, 64
    road_mask |= (blue_channel == 64) & (green_channel == 0) & (red_channel == 192)
    # Sidewalk,0, 0, 192
    sidewalk_mask = (blue_channel == 192) & (green_channel == 0) & (red_channel == 0)
    # LaneMkgsDriv,128, 0, 192
    lane_marking_mask = (blue_channel == 192) & (green_channel == 0) & (red_channel == 128)
    

    # 6 classes: (0=void,...,5=traffic_light)
    gray_image = np.zeros_like(blue_channel, dtype=np.uint8)
    gray_image[vehicles_mask] = 1
    gray_image[road_mask] = 2
    gray_image[sidewalk_mask] = 3
    gray_image[lane_marking_mask] = 4

    return gray_image

# reduces number of classes to 5: void/unlabled, vehicles, road, sidewalk, 
# lane markings
def reduce_classes_CARLA(img):
    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]

    # Create a mask for pixels that meet the condition (B, G, R)
    # cars (142, 0, 0)
    vehicles_mask = (blue_channel == 142) & (green_channel == 0) & (red_channel == 0)
    # road (128, 64, 128)
    road_mask = (blue_channel == 128) & (green_channel == 64) & (red_channel == 128)
    road_mask |= (blue_channel == 50) & (green_channel == 234) & (red_channel == 157)
    # sidewalk (232, 35, 244)
    sidewalk_mask = (blue_channel == 232) & (green_channel == 35) & (red_channel == 244)
    # lane markings (50, 234, 157)
    lane_marking_mask = (blue_channel == 50) & (green_channel == 234) & (red_channel == 157)

    gray_image = np.zeros_like(blue_channel, dtype=np.uint8)
    gray_image[vehicles_mask] = 1
    gray_image[road_mask] = 2
    gray_image[sidewalk_mask] = 3
    gray_image[lane_marking_mask] = 4

    return gray_image

test_rgb_dir = os.path.abspath("../data/camVid/unprocessed/test")
test_seg_dir = os.path.abspath("../data/camVid/unprocessed/test_labels")
train_rgb_dir = os.path.abspath("../data/camVid/unprocessed/train")
train_seg_dir = os.path.abspath("../data/camVid/unprocessed/train_labels")
val_rgb_dir = os.path.abspath("../data/camVid/unprocessed/val")
val_seg_dir = os.path.abspath("../data/camVid/unprocessed/val_labels")

CARLA_rgb_dir = os.path.abspath("../data/camVid/unprocessed/CARLA")
CARLA_seg_dir = os.path.abspath("../data/camVid/unprocessed/CARLA_labels")

write_rgb_dir = os.path.abspath("../data/camVid/processed/rgb")
write_seg_dir = os.path.abspath("../data/camVid/processed/seg")

test_rgb = read_images_from_dir(test_rgb_dir)
test_seg = read_images_from_dir(test_seg_dir, seg=True)
train_rgb = read_images_from_dir(train_rgb_dir)
train_seg = read_images_from_dir(train_seg_dir, seg=True)
val_rgb = read_images_from_dir(val_rgb_dir)
val_seg = read_images_from_dir(val_seg_dir, seg=True)

CARLA_rgb = read_images_from_dir_carla(CARLA_rgb_dir)
CARLA_seg = read_images_from_dir_carla(CARLA_seg_dir)

counter = 1
for i in tqdm(range(len(CARLA_rgb))):
    rgb = CARLA_rgb[i]
    seg = CARLA_seg[i]
    seg_r = reduce_classes_CARLA(seg)

    cv2.imwrite(write_rgb_dir + "/" + str(counter) + ".png", rgb)
    cv2.imwrite(write_seg_dir + "/" + str(counter) + ".png", seg_r)
    counter += 1

for key in tqdm(test_rgb):
    rgb = test_rgb[key]
    seg = test_seg[key]
    seg_r = reduce_classes(seg)

    cv2.imwrite(write_rgb_dir + "/" + str(counter) + ".png", rgb)
    cv2.imwrite(write_seg_dir + "/" + str(counter) + ".png", seg_r)
    counter += 1

    rgb_f = cv2.flip(rgb, 1)
    seg_r_f = cv2.flip(seg_r, 1)
    cv2.imwrite(write_rgb_dir + "/" + str(counter) + ".png", rgb_f)
    cv2.imwrite(write_seg_dir + "/" + str(counter) + ".png", seg_r_f)
    counter += 1

for key in tqdm(train_rgb):
    rgb = train_rgb[key]
    seg = train_seg[key]
    seg_r = reduce_classes(seg)

    cv2.imwrite(write_rgb_dir + "/" + str(counter) + ".png", rgb)
    cv2.imwrite(write_seg_dir + "/" + str(counter) + ".png", seg_r)
    counter += 1

    rgb_f = cv2.flip(rgb, 1)
    seg_r_f = cv2.flip(seg_r, 1)
    cv2.imwrite(write_rgb_dir + "/" + str(counter) + ".png", rgb_f)
    cv2.imwrite(write_seg_dir + "/" + str(counter) + ".png", seg_r_f)
    counter += 1

for key in tqdm(val_rgb):
    rgb = val_rgb[key]
    seg = val_seg[key]
    seg_r = reduce_classes(seg)

    cv2.imwrite(write_rgb_dir + "/" + str(counter) + ".png", rgb)
    cv2.imwrite(write_seg_dir + "/" + str(counter) + ".png", seg_r)
    counter += 1

    rgb_f = cv2.flip(rgb, 1)
    seg_r_f = cv2.flip(seg_r, 1)
    cv2.imwrite(write_rgb_dir + "/" + str(counter) + ".png", rgb_f)
    cv2.imwrite(write_seg_dir + "/" + str(counter) + ".png", seg_r_f)
    counter += 1

