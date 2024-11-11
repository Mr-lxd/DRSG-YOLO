from ultralytics import YOLO

if __name__ == '__main__':
    # # 模型验证
    model = YOLO(' runs/detect/train76/weights/best.pt')
    model.val(**{'data': 'RiceSeedData1.yaml'},batch=1)

    # # 模型推理
    #model = YOLO('weight/8nseg-best.pt')
    #model.predict(source=r'F:\Desktop\other-method\miou\photo-1131\stem', **{'save': True},name="improve-8nseg")
