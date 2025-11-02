from ultralytics import YOLO
import torch

model = YOLO("yolo11l-seg.pt")

def train_yolo_segmentation():
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("No GPU found, using CPU.")

    model.to(device)

    try:
        results = model.train(
            data='data.yaml',
            epochs=100,
            imgsz=640,
            workers=4,
            device=device,
            batch=4,
            val=True,
            optimizer='auto',
            seed=42,
            verbose=True,
            degrees=15.0,
            translate=0.15,
            scale=0.2,
            fliplr=0.5,
            mosaic=1.0,
            erasing=0.4
        )
        print("Training complete. Model saved in the 'runs/segment/' directory.")
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    train_yolo_segmentation()

    model = YOLO('runs/segment/train/weights/best.pt')

    metrics = model.val(data='data.yaml')

    print("Box mAP50-95:", metrics.box.map)
    print("Mask mAP50-95:", metrics.seg.map)
    print("Box mAP50:", metrics.box.map50)
    print("Mask mAP50:", metrics.seg.map50)
