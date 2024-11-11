from ultralytics import YOLO
import torch
from torch.utils.data import DataLoader
 
if __name__ == '__main__':
    model = YOLO("ultralytics/cfg/models/v8/models/yolov8-RFAConv.yaml")  # 从头开始构建新模型
    model.load("runs/detect/train11/weights/best.pt")  # 加载预训练模型（建议用于训练）
    #model.load("yolov8n.pt")


    model.train(data="RiceSeedData1.yaml", epochs=300,workers = 4,batch=2, optimizer='auto')  # 训练模型
    #metrics = model.val()  # 在测试集上评估模型性能


    #results = model.predict(source=r"D:\DL_Project\data\test",save=True,save_conf=True,save_txt=True,name='output')

