# from helpers.sorter import main
# main()

from ultralytics import YOLO

model = YOLO("runs/detect/robot_detection/weights/best.pt")

result = model("dataset/raw/tests/tests_0003.png")
result[0].show()

