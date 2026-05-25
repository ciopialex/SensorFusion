import cv2
from ultralytics import YOLO
import pandas as pd
import requests
import os

def download_sample_image():
    # Download a sample traffic image if it doesn't exist
    url = "https://ultralytics.com/images/zidane.jpg"  # Better: A traffic image
    # Let's use a real traffic image URL commonly used for object detection testing
    url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
    filename = "sample_traffic.jpg"
    
    if not os.path.exists(filename):
        print(f"Downloading sample image from {url}...")
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    return filename

def run_inference():
    print("Loading YOLOv11 Nano model...")
    # Load the computationally lightweight YOLO configuration suitable for 4GB VRAM
    model = YOLO("yolo11n.pt")  

    image_path = download_sample_image()

    print(f"Running inference on {image_path}...")
    # Run the model
    results = model(image_path)

    # There's only one image, so we take the first result
    result = results[0]

    # Save the visual result
    output_image = "result_traffic.jpg"
    result.save(filename=output_image)
    print(f"Saved visual predictions to {output_image}")

    # Extract bounding boxes and convert to Pandas DataFrame
    print("Extracting bounding box data...")
    boxes = result.boxes
    
    data = []
    if boxes is not None:
        for box in boxes:
            # Get coordinates in [x_center, y_center, width, height] format or [x1, y1, x2, y2]
            # Since standard formats usually prefer x1, y1, x2, y2 (top-left, bottom-right)
            coords = box.xyxy[0].tolist() 
            x1, y1, x2, y2 = coords
            
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            data.append({
                "class_name": class_name,
                "confidence": round(conf, 4),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            })

    if data:
        df = pd.DataFrame(data)
        csv_filename = "sample_detections.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Saved detections to {csv_filename}")
        print("\nDetection Summary:")
        print(df)
    else:
        print("No objects detected.")

if __name__ == "__main__":
    run_inference()
