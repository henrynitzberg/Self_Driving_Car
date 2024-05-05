from testPredictSegment import predict_segment
from testPredictControl import predict_control
from getFrame import getFrame
from sendControl import W, A, S, D, PressKey, ReleaseKey
import cv2
import time

def drive(control):
    driveKey = W
    steerKey = S
    if control[1] > control[2]:
        PressKey(driveKey)
    else:
        ReleaseKey(driveKey)
    if control[0] > .51:
        steerKey = D
    elif control[0] < .49:
        steerKey = A

    PressKey(steerKey)
    time.sleep(.01)
    ReleaseKey(steerKey)

# grace period to enter the game
for i in range(3):
    print("Starting in " + str(3 - i))
    time.sleep(1)
print("GO!")


# PressKey(W)
# time.sleep(1)
# ReleaseKey(W)

# testing TODO:delete
frames = 1000
start = time.time()
for i in range(frames):
    frame = getFrame()
    seg = predict_segment(frame)
    control = predict_control(seg)
    print(control)

    # "driving"
    drive(control)
    if i % 3 == 0:
        ReleaseKey(W)

    cv2.imshow('frame', frame)
    cv2.imshow('segmented frame', seg * 40)
    cv2.imshow('colorful segmentation!', cv2.applyColorMap(seg * 50, cv2.COLORMAP_JET))
    if (cv2.waitKey(25) & 0xFF == ord('q')) or (i == 999):
        cv2.destroyAllWindows()
        break
end = time.time()
print("fps: " + str(frames / (end - start)))
