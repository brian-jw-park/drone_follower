import djitellopy as tello
import time
import cv2

node = tello.Tello()

node.connect()
print(node.get_battery())

# take off and move to ceiling
node.takeoff()
# node.move_up(100)

node.streamon()
node.send_command_with_return("downvision 1")
frame = node.get_frame_read()

# display frames from drone
while True:
    img = frame.frame

    cv2.imshow('win', img)

    if cv2.waitKey(10) == ord('q'):
        break


  # 1. detect person bounding boxes and select the one closest to the center


  # 2. move to center bounding box

node.land()

cv2.destroyAllWindows()

# node.land()

node.end()