# from helpers.sorter import main
# main()

from ultralytics import YOLO


model = YOLO("runs/detect/robot_detector/weights/best.pt")

results = model("dataset/raw/tests/tests_0003.png")
results[0].show()