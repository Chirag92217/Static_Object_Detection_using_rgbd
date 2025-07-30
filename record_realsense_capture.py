import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime

# Base recordings directory
base_dir = 'recordings'
os.makedirs(base_dir, exist_ok=True)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color,    640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth,    640, 480, rs.format.z16,   30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8,   30)
profile = pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)

recording   = False
session_dir = None
frame_idx   = 0

print("Press 'c' to start recording, 'e' to end recording, 'q' to quit.")

try:
    while True:
        # Get aligned frames
        frames      = pipeline.wait_for_frames()
        aligned     = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        ir_frame    = aligned.get_infrared_frame(1)

        if not color_frame or not depth_frame or not ir_frame:
            continue

        # Raw data arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        ir_image    = np.asanyarray(ir_frame.get_data())

        # For display only
        depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        ir_vis    = cv2.applyColorMap(cv2.convertScaleAbs(ir_image, alpha=0.03), cv2.COLORMAP_BONE)

        cv2.imshow('RGB Stream',   color_image)
        cv2.imshow('Depth Stream', depth_vis)
        cv2.imshow('IR Stream',    ir_vis)

        key = cv2.waitKey(1) & 0xFF

        # Start recording
        if key == ord('c') and not recording:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(base_dir, f"session_{ts}")
            rgb_dir     = os.path.join(session_dir, 'rgb')
            depth_dir   = os.path.join(session_dir, 'depth')
            ir_dir      = os.path.join(session_dir, 'ir')
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(ir_dir, exist_ok=True)
            recording = True
            frame_idx = 0
            print(f"Recording started. Saving to {session_dir}")

        # Stop recording
        elif key == ord('e') and recording:
            recording = False
            print(f"Recording stopped. {frame_idx} frames saved in each folder.")

        # Quit
        elif key == ord('q'):
            break

        # Save raw frames if recording
        if recording:
            rgb_path   = os.path.join(rgb_dir,   f"rgb_{frame_idx:06d}.png")
            depth_path = os.path.join(depth_dir, f"depth_{frame_idx:06d}.png")
            ir_path    = os.path.join(ir_dir,    f"ir_{frame_idx:06d}.png")

            cv2.imwrite(rgb_path,   color_image)  # 8-bit BGR
            cv2.imwrite(depth_path, depth_image)  # 16-bit PNG with original depth
            cv2.imwrite(ir_path,    ir_image)     # 8-bit PNG with original IR

            frame_idx += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
