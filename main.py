from ultralytics import YOLO

model = YOLO("hawk_robot_model.pt")

results = model("dataset/raw/tests/tests_0009.png")
results[0].show()