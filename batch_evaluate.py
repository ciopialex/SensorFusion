import os
import cv2
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from radar_processor import RadarProcessor
from projection_math import CameraProjector

def get_mappings():
    # Maps video suffix to (Excel file name, active sensors)
    return {
        'CAM-Doc-L': ('Dataset/CameraParameters_Front.xlsx', ['FC']),
        'CAM-Doc-F': ('Dataset/CameraParameters_Left.xlsx', ['FL', 'RL']),
        'CAM-Doc-R': ('Dataset/CameraParameters_Right.xlsx', ['FR', 'RR']),
        'CAM-Doc-B': ('Dataset/CameraParameters_Rear.xlsx', ['RL', 'RR'])
    }

def process_video(video_path, model, radar_dir, excel_path, json_path, sensors_filter):
    # Load projector
    projector = CameraProjector(excel_path, json_path)
    
    # Load radar
    video_basename = os.path.basename(video_path)
    match = re.search(r'(.*?_.*?-TM02HRO-StabilityTests)', video_basename)
    if match:
        base_prefix = match.group(1)
        radar_prefix = f"{base_prefix}_CAM-Doc-B"
    else:
        radar_prefix = video_basename.replace("_rectified.mp4", "").replace(".mp4", "")
        
    radar = RadarProcessor(radar_dir)
    radar.load_data(radar_prefix)
    
    if radar.ego_data is None:
        return 0, 0, 0
        
    # Open Video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sync timestamp
    start_ts = radar.ego_data.iloc[0]['Timestamp']
    
    tp = 0
    fp = 0
    fn = 0
    
    frame_idx = 0
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        curr_ts = int(start_ts + (frame_idx / fps) * 1e9)
        
        # 1. YOLO with Persistent BoT-SORT Tracking
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
        yolo_boxes = []
        target_classes = [0, 1, 2, 3, 5, 7] # person, bicycle, car, motorcycle, bus, truck
        
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in target_classes:
                xyxy = box.xyxy[0].cpu().numpy()
                w_box = xyxy[2] - xyxy[0]
                h_box = xyxy[3] - xyxy[1]
                if w_box < 40 or h_box < 40:
                    continue
                yolo_boxes.append({'bbox': xyxy})
                
        # 2. Radar Detection & Projection
        raw_dets = radar.get_detections_at_timestamp(curr_ts)
        dets = [d for d in raw_dets if d['sensor'] in sensors_filter]
        
        img_pts, valid_mask = projector.project_points(dets)
        
        frame_tp = 0
        frame_fp = 0
        matched_yolo_boxes = set()

        for pt, is_valid in zip(img_pts, valid_mask):
            if not is_valid:
                continue
                
            px, py = int(pt[0]), int(pt[1])
            radar_matched = False
            
            for j, ybox in enumerate(yolo_boxes):
                xb = ybox['bbox']
                w_box = xb[2] - xb[0]
                h_box = xb[3] - xb[1]
                
                tol_w = 0.15 * w_box
                tol_h = 0.20 * h_box
                
                if (xb[0] - tol_w) <= px <= (xb[2] + tol_w) and (xb[1] - tol_h) <= py <= (xb[3] + tol_h):
                    radar_matched = True
                    matched_yolo_boxes.add(j)
                    break
            
            if radar_matched:
                frame_tp += 1
            else:
                frame_fp += 1

        frame_fn = len(yolo_boxes) - len(matched_yolo_boxes)
        
        tp += frame_tp
        fp += frame_fp
        fn += frame_fn
        frame_idx += 1
        
    cap.release()
    return tp, fp, fn

def main():
    video_dir = "Dataset/rectified_videos"
    radar_dir = "Dataset/Generated_CSV"
    json_path = "Dataset/fisheye_calibration.json"
    model_path = "/home/shennyonthebeat/ESP23DEVKIT/ESP32_DEV_KIT_Project/yolov8n.pt"
    
    # Load YOLO
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    # Find all rectified videos
    videos = [f for f in os.listdir(video_dir) if f.endswith("_rectified.mp4")]
    videos.sort()
    
    mappings = get_mappings()
    results = []
    
    agg_tp = 0
    agg_fp = 0
    agg_fn = 0
    
    print(f"Found {len(videos)} videos to process.")
    
    for v_file in videos:
        # Determine camera model mapping
        mapped_key = None
        for key in mappings.keys():
            if key in v_file:
                mapped_key = key
                break
                
        if mapped_key is None:
            print(f"Skipping {v_file}: could not map camera type.")
            continue
            
        excel_path, sensors = mappings[mapped_key]
        video_path = os.path.join(video_dir, v_file)
        
        print(f"\nProcessing {v_file}...")
        print(f"  Camera Model: {mapped_key} ({os.path.basename(excel_path)})")
        print(f"  Radar Sensors: {sensors}")
        
        tp, fp, fn = process_video(video_path, model, radar_dir, excel_path, json_path, sensors)
        
        print(f"  Result -> TP: {tp} | FP (Ghosts): {fp} | FN (Misses): {fn}")
        
        results.append({
            'Video': v_file,
            'Camera': mapped_key,
            'TP': tp,
            'FP': fp,
            'FN': fn
        })
        
        agg_tp += tp
        agg_fp += fp
        agg_fn += fn
        
    # Print final summary table
    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT")
    print("="*80)
    
    df = pd.DataFrame(results)
    print("| Video | Camera | TP | FP (Ghosts) | FN (Misses) |")
    print("| :--- | :--- | :---: | :---: | :---: |")
    for r in results:
        print(f"| {r['Video']} | {r['Camera']} | {r['TP']} | {r['FP']} | {r['FN']} |")
    
    print("\n" + "="*80)
    print("AGGREGATE CONFUSION MATRIX:")
    print(f"Total True Positives (TP): {agg_tp}")
    print(f"Total False Positives (FP): {agg_fp}")
    print(f"Total False Negatives (FN): {agg_fn}")
    print("="*80)
    
    # Save results to CSV
    df.to_csv("evaluation_summary.csv", index=False)
    print("Summary saved to evaluation_summary.csv")

if __name__ == "__main__":
    main()
