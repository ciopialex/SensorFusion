import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
import argparse
from radar_processor import RadarProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to rectified video")
    parser.add_argument("--radar_dir", type=str, default="extracted_csvs/radar_csv", help="Dir with radar CSVs")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="YOLO model")
    parser.add_argument("--output", type=str, default="output_sync.mp4", help="Output video path")
    args = parser.parse_args()

    # Load YOLO
    model = YOLO(args.model)
    
    # Load Radar
    # Extract prefix from video filename
    # e.g. 20260309_164510-TM02HRO-StabilityTest_CAM-Doc-B_rectified.mp4
    video_basename = os.path.basename(args.video)
    prefix = video_basename.replace("_rectified.mp4", "")
    
    radar = RadarProcessor(args.radar_dir)
    radar.load_data(prefix)
    
    if radar.ego_data is None:
        print(f"Warning: No ego data found for prefix {prefix}")
        # We'll just use objlist if available
    
    # Open Video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup Output
    bev_w = 400
    combined_w = width + bev_w
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (combined_w, height))
    
    # Timestamp Synchronization
    # We'll assume the first frame matches the first timestamp in ego_data or objlist
    start_ts = 0
    if radar.ego_data is not None:
        start_ts = radar.ego_data.iloc[0]['Timestamp']
    elif len(radar.obj_data) > 0:
        # Get min timestamp from any sensor
        start_ts = min([df['Timestamp'].min() for df in radar.obj_data.values()])
    
    print(f"Starting synchronization at {start_ts}")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate Current Timestamp
        # 20 fps means 50ms (50,000,000 ns) per frame
        curr_ts = int(start_ts + (frame_idx / fps) * 1e9)
        
        # 1. YOLO Detection
        results = model(frame, verbose=False)[0]
        
        # Draw YOLO detections
        # Selective classes: person (0), bicycle (1), car (2), motorcycle (3), bus (5), truck (7)
        target_classes = [0, 1, 2, 3, 5, 7]
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in target_classes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                label = f"{model.names[cls]} {conf:.2f}"
                
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 2. Radar Detection
        dets = radar.get_detections_at_timestamp(curr_ts)
        bev = radar.render_bev(dets, width=bev_w, height=height)
        
        # 3. Combine Views
        canvas = np.zeros((height, combined_w, 3), dtype=np.uint8)
        canvas[:, :width] = frame
        canvas[:, width:] = bev
        
        # Add Progress Info
        cv2.putText(canvas, f"Frame: {frame_idx}/{total_frames} | {prefix}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(canvas)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"Done! Output saved to {args.output}")

if __name__ == "__main__":
    main()
