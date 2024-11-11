from ultralytics import YOLO

yolo = YOLO("runs/detect/train76/weights/best.pt", task="detect")

results = yolo(source="D:/DL_Project/data/test/images", save=True, save_txt=True)