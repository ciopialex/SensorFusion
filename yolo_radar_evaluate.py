import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
import argparse
import pandas as pd
import re
from tqdm import tqdm
from radar_processor import RadarProcessor
from projection_math import CameraProjector


def draw_radar_bbox(frame, corners_2d, color=(255, 255, 0), thickness=2):
    """
    Draws a projected 3D bounding box wireframe on the frame.
    corners_2d: 8x2 array of projected corner points.
    Order: bottom [0-3], top [4-7], each face is (FL, FR, BR, BL).
    """
    pts = corners_2d.astype(np.int32)
    
    # Draw bottom face
    for i in range(4):
        cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color, thickness)
    # Draw top face
    for i in range(4):
        cv2.line(frame, tuple(pts[4 + i]), tuple(pts[4 + (i + 1) % 4]), color, thickness)
    # Draw vertical pillars
    for i in range(4):
        cv2.line(frame, tuple(pts[i]), tuple(pts[i + 4]), color, thickness)


def compute_overlap_ratio(yolo_bbox, radar_bbox_2d):
    """
    Computes the IoU between a YOLO axis-aligned bbox and the axis-aligned
    bounding rect of a projected radar 3D box.
    """
    # YOLO bbox
    x1_y, y1_y, x2_y, y2_y = yolo_bbox
    
    # Radar projected: take the axis-aligned bounding rect of all 8 projected corners
    x1_r = np.min(radar_bbox_2d[:, 0])
    y1_r = np.min(radar_bbox_2d[:, 1])
    x2_r = np.max(radar_bbox_2d[:, 0])
    y2_r = np.max(radar_bbox_2d[:, 1])
    
    # Intersection
    x1_i = max(x1_y, x1_r)
    y1_i = max(y1_y, y1_r)
    x2_i = min(x2_y, x2_r)
    y2_i = min(y2_y, y2_r)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    area_y = (x2_y - x1_y) * (y2_y - y1_y)
    area_r = (x2_r - x1_r) * (y2_r - y1_r)
    union = area_y + area_r - inter
    
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to rectified video")
    parser.add_argument("--radar_dir", type=str, default="Dataset/Generated_CSV", help="Dir with radar CSVs")
    parser.add_argument("--model", type=str, default="/home/shennyonthebeat/ESP23DEVKIT/ESP32_DEV_KIT_Project/yolov8n.pt", help="YOLO model")
    parser.add_argument("--output", type=str, default="output_evaluate.mp4", help="Output video path")
    parser.add_argument("--excel", type=str, default="Dataset/CameraParameters_Front.xlsx", help="Camera Extrinsics")
    parser.add_argument("--json", type=str, default="Dataset/fisheye_calibration.json", help="Camera Intrinsics")
    parser.add_argument("--sensors", type=str, default="all", help="Comma-separated radar sensors to use (e.g. FC,FL). Default: all")
    parser.add_argument("--show", action="store_true", help="Show real-time visualization window")
    args = parser.parse_args()

    # Load YOLO
    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    
    # Load Camera Projector
    print("Loading Camera Projector Math Engine...")
    projector = CameraProjector(args.excel, args.json)

    # Load Radar
    video_basename = os.path.basename(args.video)
    
    # Hack for Hella dataset: video might be -F, but CSVs are all exported with -B
    match = re.search(r'(.*?_.*?-TM02HRO-StabilityTests)', video_basename)
    if match:
        base_prefix = match.group(1)
        radar_prefix = f"{base_prefix}_CAM-Doc-B"
    else:
        radar_prefix = video_basename.replace("_rectified.mp4", "").replace(".mp4", "")
        
    print(f"Loading Radar data for prefix: {radar_prefix}")
    radar = RadarProcessor(args.radar_dir)
    radar.load_data(radar_prefix)
    
    if radar.ego_data is None:
        print(f"Warning: No ego data found for prefix {radar_prefix}")
        
    # Filter active sensors
    active_sensors = None
    if args.sensors.lower() != "all":
        active_sensors = [s.strip().upper() for s in args.sensors.split(",")]
        print(f"Filtering radar detections to sensors: {active_sensors}")
    
    # Open Video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup Output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Timestamp Synchronization
    # Use the LATEST start among all active data sources so frame 0 aligns
    # to a point where every sensor has live data. Ego data can start up to
    # 10s before the objlists, which would lock early frames to stale radar rows.
    start_ts = 0
    if radar.ego_data is not None:
        start_ts = radar.ego_data.iloc[0]['Timestamp']

    if len(radar.obj_data) > 0:
        active_obj_data = radar.obj_data
        if active_sensors is not None:
            active_obj_data = {k: v for k, v in radar.obj_data.items() if k in active_sensors}
        if active_obj_data:
            objlist_start = max([df['Timestamp'].min() for df in active_obj_data.values()])
            # Sync to whichever source starts LATER so all are live from frame 0
            start_ts = max(start_ts, objlist_start)
    
    print(f"Starting synchronization at {start_ts} (Video FPS: {fps})")
    
    # Metrics
    tp = 0
    fp = 0
    fn = 0

    frame_idx = 0
    
    if args.show:
        cv2.namedWindow("Radar vs YOLO Evaluation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Radar vs YOLO Evaluation", 1280, 720)
    
    for _ in tqdm(range(total_frames), desc="Processing Video Frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
            
        curr_ts = int(start_ts + (frame_idx / fps) * 1e9)
        
        # Create overlay for semi-transparent fills
        overlay = frame.copy()
        
        # 1. YOLO Detection with Persistent BoT-SORT Tracking
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
        yolo_boxes = []
        target_classes = [0, 1, 2, 3, 5, 7] # person, bicycle, car, motorcycle, bus, truck
        
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in target_classes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                # IMPLEMENTATION B: Filter out small/far-away boxes
                w_box = xyxy[2] - xyxy[0]
                h_box = xyxy[3] - xyxy[1]
                if w_box < 40 or h_box < 40:
                    continue
                
                track_id = int(box.id[0]) if box.id is not None else -1
                yolo_boxes.append({
                    'bbox': xyxy,
                    'cls': model.names[cls],
                    'conf': conf,
                    'track_id': track_id
                })
                # Draw YOLO bounding box (green, semi-transparent fill)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"T{track_id} {model.names[cls]} {conf:.2f}" if track_id != -1 else f"{model.names[cls]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 2. Radar Detection & 3D Bounding Box Projection
        raw_dets = radar.get_detections_at_timestamp(curr_ts)
        
        # Filter detections by sensor if specified
        if active_sensors is not None:
            dets = [d for d in raw_dets if d['sensor'] in active_sensors]
        else:
            dets = raw_dets
        
        # Also get center-point projections for matching logic
        img_pts, valid_mask = projector.project_points(dets)
        
        frame_tp = 0
        frame_fp = 0
        matched_yolo_boxes = set()

        for i, det in enumerate(dets):
            # Project 3D bounding box
            corners_2d, bbox_valid = projector.project_bbox_3d(det)
            
            # Check if center point is valid (from original projection)
            center_valid = i < len(valid_mask) and valid_mask[i]
            
            if not (bbox_valid or center_valid):
                continue
            
            # Match radar to YOLO using center point (existing logic)
            radar_matched = False
            if center_valid:
                px, py = int(img_pts[i][0]), int(img_pts[i][1])
                
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
            
            # Also check overlap via projected bounding box IoU
            if not radar_matched and bbox_valid:
                for j, ybox in enumerate(yolo_boxes):
                    iou = compute_overlap_ratio(ybox['bbox'], corners_2d)
                    if iou > 0.05:
                        radar_matched = True
                        matched_yolo_boxes.add(j)
                        break
            
            if radar_matched:
                frame_tp += 1
                radar_color = (255, 200, 0)  # Cyan-ish for True Positive
            else:
                frame_fp += 1
                radar_color = (0, 0, 255)    # Red for False Positive (Ghost)
            
            # Draw radar 3D bounding box wireframe
            if bbox_valid:
                draw_radar_bbox(frame, corners_2d, color=radar_color, thickness=2)
                
                # Semi-transparent fill on the front face (corners 0,1,5,4)
                front_face = corners_2d[[0, 1, 5, 4]].astype(np.int32)
                cv2.fillConvexPoly(overlay, front_face, radar_color)
            
            # Draw center dot
            if center_valid:
                px, py = int(img_pts[i][0]), int(img_pts[i][1])
                cv2.circle(frame, (px, py), 6, radar_color, -1)
                cv2.circle(frame, (px, py), 9, (255, 255, 255), 1)
                cv2.putText(frame, f"ID:{det['id']}({det['sensor']})", 
                           (px + 10, py + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        frame_fn = len(yolo_boxes) - len(matched_yolo_boxes)
        
        tp += frame_tp
        fp += frame_fp
        fn += frame_fn

        # Blend the semi-transparent overlay
        alpha = 0.15
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Add Metrics Info
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"TP: {tp} | FP (Ghosts): {fp} | FN (Misses): {fn}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Legend
        cv2.rectangle(frame, (10, height - 90), (300, height - 10), (30, 30, 30), -1)
        cv2.rectangle(frame, (15, height - 85), (35, height - 70), (0, 255, 0), -1)
        cv2.putText(frame, "YOLO Detection", (40, height - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(frame, (15, height - 65), (35, height - 50), (255, 200, 0), -1)
        cv2.putText(frame, "Radar BBox (TP Match)", (40, height - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(frame, (15, height - 45), (35, height - 30), (0, 0, 255), -1)
        cv2.putText(frame, "Radar BBox (FP Ghost)", (40, height - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        out.write(frame)
        
        if args.show:
            cv2.imshow("Radar vs YOLO Evaluation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        frame_idx += 1

    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()
    
    print("\n=== CLASSIFICATION MATRIX ===")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"\nEvaluation Output saved to {args.output}")

if __name__ == "__main__":
    main()

