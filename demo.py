import djitellopy as tello
from ultralytics import YOLO
import time
import cv2

model = YOLO("yolo-Weights/yolov8n.pt")

drone = tello.Tello()

drone.connect()
print(drone.get_battery())

# take off and move to ceiling
drone.takeoff()
drone.streamon()
frame = drone.get_frame_read()

tick = 0

# display frames from drone
while True:
    img = frame.frame

    result = model(img, stream=True)[0]
    box = result.boxes[0]

    if box.cls[0] != 0:
        continue

    # bounding box
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

    # put box in cam
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    if (x2 + x1) / 2 < img.shape[1] / 2 - 50:
        drone.move_left(40)

    elif (x2 + x1) / 2 > img.shape[1] / 2 + 50:
        drone.move_right(40)

    elif x2 - x1 < 350:
        drone.move_forward(40)
    
    elif x2 - x1 > 400:
        drone.move_back(40)

    # object details
    org = [x1, y1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    cv2.putText(img, 'target', org, font, fontScale, color, thickness)

    cv2.imshow('win', img)

    if cv2.waitKey(10) == ord('q'):
        break

drone.land()

cv2.destroyAllWindows()

drone.land()

drone.end()