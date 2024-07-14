import djitellopy as tello
from ultralytics import YOLO
import time
import cv2
from face import get_face_box
import numpy as np

model = YOLO("yolo-Weights/yolov8n.pt")

drone = tello.Tello()

drone.connect()
print(drone.get_battery())

# # take off and move to ceiling
drone.takeoff()
drone.move_up(100)

tick = 0

drone.streamon()
frame = drone.get_frame_read(with_queue=True)

# display frames from drone
while True:
    img = frame.frame

    if img is None:
        continue
    # img = np.zeros((480, 640, 3), np.uint8)

    results = list(model(img, stream=True))
    if not results or len(results[0].boxes) == 0:
        continue
    box = results[0].boxes[0]

    # if box.cls[0] != 0:
    #     continue

    # bounding box
    # boxes = get_face_box(img)
    # if not boxes:
    #     continue

    # x1, y1, w, h = boxes[0]
    # x2, y2 = x1 + w, y1 + h

    x1, y1, x2, y2  = box.xyxy[0]

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

    # # put box in cam
    cv2.imshow('Webcam', img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    if (x2 + x1) / 2 < img.shape[1] / 2 - 170:
        drone.send_rc_control(-50, 0, 0, 0)

    elif (x2 + x1) / 2 > img.shape[1] / 2 + 170:
        drone.send_rc_control(50, 0, 0, 0)

    elif x2 - x1 < 20:
        drone.send_rc_control(0, 50, 0, 0)
    
    elif x2 - x1 > 50:
        drone.send_rc_control(0, -50, 0, 0)

    else:
        drone.send_rc_control(0, 0, 0, 0)

    if cv2.waitKey(10) == ord('q'):
        break

drone.land()

cv2.destroyAllWindows()

drone.land()

drone.end()