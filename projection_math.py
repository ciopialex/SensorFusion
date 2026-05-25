import numpy as np
import cv2
import pandas as pd
import json
import os

class CameraProjector:
    def __init__(self, excel_path, json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON not found: {json_path}")
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel not found: {excel_path}")

        # Load Intrinsics
        with open(json_path, 'r') as f:
            calib = json.load(f)
            self.K = np.array(calib['K'], dtype=np.float64)
            # The JSON D vector has 4 values, which maps to k1, k2, k3, k4 in fisheye model
            self.D = np.array(calib['D'], dtype=np.float64).flatten()
            self.image_size = tuple(calib['image_size'])

        # Load Extrinsics from Excel
        df = pd.read_excel(excel_path, sheet_name='Measurements')
        
        # Extract Symbol and Value columns based on their indices
        # Column 2 is Symbol, Column 3 is Value based on the dump
        symbols = df.iloc[:, 2].dropna().astype(str).str.strip().values
        values = df.iloc[:, 3].dropna().values
        
        # Filter matching pairs where symbol looks valid
        self.params = {}
        for s, v in zip(symbols, values):
            if isinstance(s, str) and (s.startswith('cam_') or s in ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4']):
                try:
                    self.params[s] = float(v)
                except ValueError:
                    pass

        # Extract Extrinsics (default fallbacks provided)
        self.cam_x = self.params.get('cam_x', 0.17)
        self.cam_y = self.params.get('cam_y', 2.0)
        self.cam_z = self.params.get('cam_z', 1.26)
        
        # Invert yaw and pitch because Excel uses Left-Hand rules / custom conventions
        # (+ = right yaw, + = nose up pitch), but mathematical rotation matrices
        # require standard Right-Hand conventions (+ = left yaw, + = nose down pitch).
        raw_yaw = self.params.get('cam_yaw', 0)
        if "rear" in excel_path.lower():
            raw_yaw += 180
            
        self.yaw = np.radians(-raw_yaw)
        self.pitch = np.radians(-self.params.get('cam_pitch', 0))
        self.roll = np.radians(self.params.get('cam_roll', 0))
        
        self._compute_extrinsics()

    def _compute_extrinsics(self):
        """
        Computes Rotation and Translation matrices from Vehicle Coordinate System to Camera Coordinate System.
        Vehicle (ISO 8855): X forward, Y left, Z up.
        Camera (OpenCV): Z forward, X right, Y down.
        """
        # Translate from vehicle origin to camera position
        # X_veh = cam_y (longitudinal)
        # Y_veh = -cam_x (lateral, since + is right in Excel, but + is left in ISO 8855)
        # Z_veh = cam_z (height)
        T_veh2cam = np.array([-self.cam_y, self.cam_x, -self.cam_z]).reshape(3, 1)

        # Yaw (around Z), Pitch (around Y), Roll (around X)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        R_yaw = np.array([
            [cy, -sy, 0],
            [sy,  cy, 0],
            [ 0,   0, 1]
        ])

        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        R_pitch = np.array([
            [ cp, 0, sp],
            [  0, 1,  0],
            [-sp, 0, cp]
        ])

        cr, sr = np.cos(self.roll), np.sin(self.roll)
        R_roll = np.array([
            [1,  0,   0],
            [0, cr, -sr],
            [0, sr,  cr]
        ])

        R_veh = R_yaw @ R_pitch @ R_roll

        # Axis conversion: Vehicle -> Camera
        # X_cam = -Y_veh
        # Y_cam = -Z_veh
        # Z_cam = X_veh
        R_axes = np.array([
            [0, -1,  0],
            [0,  0, -1],
            [1,  0,  0]
        ])

        # Final extrinsic matrices
        self.R = R_axes @ np.linalg.inv(R_veh)
        self.T = self.R @ T_veh2cam
        
        # Rotation vector for cv2
        self.rvec, _ = cv2.Rodrigues(self.R)

    def project_points(self, points_3d):
        """
        points_3d: list of dicts with 'x', 'y' (assuming Z=0.5 for mid-height of car)
        Returns: list of (u, v) tuples, and boolean validity mask
        """
        if not points_3d:
            return [], []

        pts = []
        for p in points_3d:
            # Assume Z=0.5m if not provided
            pts.append([p['x'], p['y'], p.get('z', 0.5)])

        pts_arr = np.array(pts, dtype=np.float64)
        
        # 1xNx3 for cv2.fisheye.projectPoints
        pts_arr_reshaped = pts_arr.reshape(1, -1, 3)

        img_pts, _ = cv2.fisheye.projectPoints(pts_arr_reshaped, self.rvec, self.T, self.K, self.D)
        img_pts = img_pts.reshape(-1, 2)

        # Filter out points behind the camera (Z_cam <= 0)
        pts_cam = (self.R @ pts_arr.T + self.T).T
        valid_mask = pts_cam[:, 2] > 0

        # Also filter out points far outside the image boundaries (optional but good)
        w, h = self.image_size
        in_frame = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < w) & (img_pts[:, 1] >= 0) & (img_pts[:, 1] < h)
        valid_mask = valid_mask & in_frame

        return img_pts, valid_mask

    def project_bbox_3d(self, detection, assumed_height=1.5):
        """
        Projects a 3D bounding box from radar detection into the camera image.
        
        detection: dict with keys 'x', 'y', 'z', 'length', 'width', 'vx', 'vy'
        assumed_height: height of the box in meters (radar doesn't always have height)
        
        Returns: list of projected corner points (8x2), validity boolean
        """
        cx = detection['x']
        cy = detection['y']
        cz = detection.get('z', 0)
        length = detection.get('length', 4.0)
        width = detection.get('width', 1.8)

        # Derive heading from velocity vector — reliably in vehicle frame.
        # posOrientationYawInRad is in sensor-absolute frame and must not be used.
        vx = detection.get('vx', 0)
        vy = detection.get('vy', 0)
        speed = np.sqrt(vx**2 + vy**2)
        yaw = np.arctan2(vy, vx) if speed > 0.5 else 0.0

        hl = length / 2.0
        hw = width / 2.0

        # 8 corners relative to object center (before yaw rotation)
        # In vehicle coordinate system: X forward, Y left, Z up
        local_corners = np.array([
            [ hl,  hw, 0],
            [ hl, -hw, 0],
            [-hl, -hw, 0],
            [-hl,  hw, 0],
            [ hl,  hw, assumed_height],
            [ hl, -hw, assumed_height],
            [-hl, -hw, assumed_height],
            [-hl,  hw, assumed_height],
        ])

        # Apply yaw rotation (around Z axis in vehicle coords)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1],
        ])

        rotated = (R_yaw @ local_corners.T).T

        # Translate to world position
        world_corners = rotated + np.array([cx, cy, cz])

        # Project through the camera model
        pts_arr = world_corners.astype(np.float64).reshape(1, -1, 3)
        img_pts, _ = cv2.fisheye.projectPoints(pts_arr, self.rvec, self.T, self.K, self.D)
        img_pts = img_pts.reshape(-1, 2)

        # Check validity: all corners must be in front of camera and roughly in-frame
        pts_cam = (self.R @ world_corners.T + self.T).T
        all_in_front = np.all(pts_cam[:, 2] > 0)

        w, h = self.image_size
        # At least some corners should be within a generous frame margin
        margin = 200
        any_in_frame = np.any(
            (img_pts[:, 0] >= -margin) & (img_pts[:, 0] < w + margin) &
            (img_pts[:, 1] >= -margin) & (img_pts[:, 1] < h + margin)
        )

        valid = all_in_front and any_in_frame
        return img_pts, valid
