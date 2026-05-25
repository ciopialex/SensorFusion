import pandas as pd
import numpy as np
import cv2
import os

class RadarProcessor:
    def __init__(self, radar_base_dir):
        self.radar_base_dir = radar_base_dir
        self.ego_data = None
        self.obj_data = {} # sensor -> dataframe

    def load_data(self, prefix):
        """
        prefix: e.g. '20260309_164510-TM02HRO-StabilityTest_CAM-Doc-B'
        """
        ego_file = os.path.join(self.radar_base_dir, f"{prefix}_EgoValues.csv")
        if os.path.exists(ego_file):
            self.ego_data = pd.read_csv(ego_file, sep=';')
            # Convert timestamp to numeric (assuming ns)
            self.ego_data['Timestamp'] = pd.to_numeric(self.ego_data['Timestamp'])
        
        sensors = ['FC', 'FL', 'FR', 'RL', 'RR']
        for s in sensors:
            obj_file = os.path.join(self.radar_base_dir, f"{prefix}_{s}_objlist.csv")
            if os.path.exists(obj_file):
                df = pd.read_csv(obj_file)
                df['Timestamp'] = pd.to_numeric(df['Timestamp'])
                self.obj_data[s] = df

    def get_detections_at_timestamp(self, ts):
        """
        Returns radar detections closest to the given timestamp.
        """
        detections = []
        for sensor, df in self.obj_data.items():
            # Find closest row
            idx = (df['Timestamp'] - ts).abs().idxmin()
            row = df.loc[idx]
            
            # Filter detections by the same cycle if possible, or just take the row if it's a single obj list
            # Actually, the CSV might have multiple objects PER timestamp.
            # Let's check if 'Timestamp' is unique in my previous view.
            # Looking back at Step 40: Timestamp 1773074712914982600 is repeated for obj 107 and 108.
            
            target_ts = df.iloc[idx]['Timestamp']
            frame_objs = df[df['Timestamp'] == target_ts]
            
            for _, obj in frame_objs.iterrows():
                vx = obj.get('TrAi_DynamicObject_t\\dynVelocityXInMpS_fl32', 0)
                vy = obj.get('TrAi_DynamicObject_t\\dynVelocityYInMpS_fl32', 0)
                
                # IMPLEMENTATION A: Filter out static clutter
                speed = np.sqrt(vx**2 + vy**2)
                if speed < 1.0: # Ignore objects moving less than 3.6 km/h (static noise like fences/signs)
                    continue
                    
                detections.append({
                    'sensor': sensor,
                    'id': obj.get('TrAi_DynamicObject_t\\statusObjectId_u16', -1),
                    'x': obj.get('TrAi_DynamicObject_t\\posXInM_fl32', 0),
                    'y': obj.get('TrAi_DynamicObject_t\\posYInM_fl32', 0),
                    'z': obj.get('TrAi_DynamicObject_t\\posZInM_fl32', 0),
                    'vx': vx,
                    'vy': vy,
                    'length': obj.get('TrAi_DynamicObject_t\\bboxBoundingBoxExtentLengthInM_fl32', 4.0),
                    'width': obj.get('TrAi_DynamicObject_t\\bboxBoundingBoxExtentWidthInM_fl32', 1.8),
                    'yaw': obj.get('TrAi_DynamicObject_t\\posOrientationYawInRad_fl32', 0),
                })
        return detections

    def render_bev(self, detections, width=400, height=800, scale=10):
        """
        Renders a Bird's-Eye View image.
        scale: pixels per meter
        """
        bev = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Origin (vehicle center) at (width//2, height//2) or bottom center?
        # Typically front-center camera: vehicle is at bottom center.
        cx, cy = width // 2, height - 100
        
        # Draw vehicle
        cv2.rectangle(bev, (cx-10, cy-20), (cx+10, cy+20), (255, 255, 255), -1)
        
        for det in detections:
            # Radar X is forward (up in BEV), Radar Y is left (left in BEV? Need to verify coordinate system)
            # Standard: X forward, Y left.
            # In image: x_img = cx - y_coord * scale, y_img = cy - x_coord * scale
            x_m = det['x']
            y_m = det['y']
            
            px = int(cx - y_m * scale)
            py = int(cy - x_m * scale)
            
            if 0 <= px < width and 0 <= py < height:
                color = (0, 0, 255) # Red for radar
                cv2.circle(bev, (px, py), 5, color, -1)
                cv2.putText(bev, str(int(det['id'])), (px+5, py+5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw distance rings
        for r in range(10, 100, 10):
            cv2.circle(bev, (cx, cy), r * scale, (50, 50, 50), 1)
            
        return bev

if __name__ == "__main__":
    # Test loading
    processor = RadarProcessor("/home/shennyonthebeat/YOLO/extracted_csvs/radar_csv")
    processor.load_data("20260309_164510-TM02HRO-StabilityTest_CAM-Doc-B")
    
    # Get a sample timestamp from ego data
    if processor.ego_data is not None:
        sample_ts = processor.ego_data.iloc[100]['Timestamp']
        dets = processor.get_detections_at_timestamp(sample_ts)
        print(f"Found {len(dets)} radar detections at {sample_ts}")
        bev = processor.render_bev(dets)
        cv2.imwrite("test_bev.jpg", bev)
