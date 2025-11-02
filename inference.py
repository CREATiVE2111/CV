import gradio as gr
from ultralytics import YOLO
import cv2

print("Loading YOLO model...")
model = YOLO('runs/segment/train/weights/best.pt')
print("Model loaded.")


def yolo_inference(image):
    """
    Takes an image in NumPy format, runs YOLO inference, and returns the annotated image.
    """
    results = model(image)

    annotated_image = results[0].plot()

    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    return annotated_image_rgb


iface = gr.Interface(
    fn=yolo_inference,
    inputs=gr.Image(type="numpy", label="Upload your image"),
    outputs=gr.Image(type="numpy", label="Result"),
    title="YOLOv11-seg Inference",
    description="Upload an image and the model will perform segmentation."
)

if __name__ == "__main__":
    iface.launch()
