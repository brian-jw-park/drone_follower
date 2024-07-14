import djitellopy as tello

node = tello.Tello()

node.connect()

# take off and move to ceiling
node.takeoff()
node.move_up(100)

node.set_video_direction(tello.Tello.CAMERA_DOWNWARD)

frames = node.get_frame_read(with_queue=True)

# display frames from drone
for img in frames:
  pass


node.land()

node.end()