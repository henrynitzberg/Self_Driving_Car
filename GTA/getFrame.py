import numpy as np
from PIL import ImageGrab
import cv2
import time
from testPredict import predict_segment

def getFrame():
    screen = ImageGrab.grab(bbox=(0, 40, 800, 640))
    screen_numpy = np.array(screen.getdata(), dtype='uint8').reshape((screen.size[1],screen.size[0],3))
    return cv2.cvtColor(screen_numpy, cv2.COLOR_BGR2RGB)

# testing TODO:delete
start = time.time()
for i in range(100):
    frame = getFrame()
    segmented = predict_segment(frame)
    cv2.imshow('frame', segmented * 30)
    if (cv2.waitKey(25) & 0xFF == ord('q')) or (i == 999):
        cv2.destroyAllWindows()
        break
end = time.time()
print("fps: " + str(100 / (end - start)))