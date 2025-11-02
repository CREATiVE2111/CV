from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/segment/train/weights/best.pt')
    metrics = model.val(data='data.yaml')

    print("Box mAP50-95:", metrics.box.map)
    print("Mask mAP50-95:", metrics.seg.map)
    print("Box mAP50:", metrics.box.map50)
    print("Mask mAP50:", metrics.seg.map50)
