import cv2
import numpy as np
from tkinter import filedialog
import json
import os

def get_meters_per_pixel(video_path, real_length_m=0.381,gui_enabled=True,default_mpp = 0.0004083): #0.381 meters for 15 inches (15 squares)
    if gui_enabled:
        print("Select 15 inches (15 squares)")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Unable to read video")

        pts = []

        def click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
                pts.append((x, y))
                cv2.circle(display, (x, y), 5, (0,255,0), -1)
                cv2.imshow("Select endpoints", display)

        display = frame.copy()
        cv2.imshow("Select endpoints", display)
        cv2.setMouseCallback("Select endpoints", click)
        print("Click two endpoints of a known-length segment.")

        while len(pts) < 2:
            cv2.waitKey(1)
        cv2.destroyWindow("Select endpoints")

        p1, p2 = pts
        pixel_dist = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
        result = real_length_m / pixel_dist

    else:
        result = default_mpp

    result_name = 'Bottom meters per pixel'

    with open(os.path.join(os.path.dirname(video_path),os.path.basename(video_path).split('.')[0]+'.json'), 'w') as f:
        json.dump({result_name:result}, f,indent=4)

    return result

if __name__ == '__main__':
    video_path = filedialog.askopenfilename()
    m_per_px = get_meters_per_pixel(video_path)
    print(f"Meters per pixel: {m_per_px:.6f}")


