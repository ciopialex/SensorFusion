import cv2
import numpy as np
import os
import pandas as pd
import re
from radar_processor import RadarProcessor
from projection_math import CameraProjector
from ultralytics import YOLO

def main():
    video_path = "Dataset/rectified_videos/20260421_164627-TM02HRO-StabilityTests_CAM-Doc-F_rectified.mp4"
    excel_path = "Dataset/CameraParameters_Front.xlsx"
    json_path = "Dataset/fisheye_calibration.json"
    radar_dir = "Dataset/Generated_CSV"
    model_path = "/home/shennyonthebeat/ESP23DEVKIT/ESP32_DEV_KIT_Project/yolov8n.pt"

    model = YOLO(model_path)
    projector = CameraProjector(excel_path, json_path)

    video_basename = os.path.basename(video_path)
    match = re.search(r'(.*?_.*?-TM02HRO-StabilityTests)', video_basename)
    base_prefix = match.group(1)
    radar_prefix = f"{base_prefix}_CAM-Doc-B"

    radar = RadarProcessor(radar_dir)
    radar.load_data(radar_prefix)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_ts = radar.ego_data.iloc[0]['Timestamp']

    # Let's inspect Frame 650 where we had 51 False Negatives
    frame_idx = 650
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        return

    curr_ts = int(start_ts + (frame_idx / fps) * 1e9)
    
    # YOLO
    results = model(frame, verbose=False)[0]
    print("=== YOLO BOXES ===")
    yolo_boxes = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in [0, 1, 2, 3, 5, 7]:
            xyxy = box.xyxy[0].cpu().numpy()
            yolo_boxes.append(xyxy)
            print(f"Class: {model.names[cls]}, Box: {xyxy}")

    # Radar
    dets = radar.get_detections_at_timestamp(curr_ts)
    print(f"\n=== RADAR DETECTIONS (Total {len(dets)}) ===")
    for d in dets[:10]:
        print(f"ID: {d['id']}, X: {d['x']}, Y: {d['y']}")

    # Projected Points
    img_pts, valid_mask = projector.project_points(dets)
    print("\n=== PROJECTED RADAR POINTS ===")
    for i, (pt, val) in enumerate(zip(img_pts, valid_mask)):
        print(f"ID: {dets[i]['id']} -> Pixel: ({pt[0]:.1f}, {pt[1]:.1f}), Valid: {val}, Radar X: {dets[i]['x']:.2f}, Y: {dets[i]['y']:.2f}")

    cap.release()

if __name__ == "__main__":
    main()
