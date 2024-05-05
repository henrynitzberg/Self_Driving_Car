import numpy as np
from PIL import ImageGrab
import cv2
import time

def getFrame():
    screen = ImageGrab.grab(bbox=(0, 40, 800, 600))
    screen_numpy = np.array(screen)
    return cv2.cvtColor(screen_numpy, cv2.COLOR_BGR2RGB)