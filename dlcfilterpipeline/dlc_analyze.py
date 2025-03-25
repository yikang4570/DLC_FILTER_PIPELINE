import deeplabcut
import os

config_path = "/Volumes/lake.s/Active/Shawn P/AAA. DEEPLABCUT/Cropped_HiRes-Shawn-2025-03-14/config.yaml"
video_path = "/Volumes/lake.s/Active/Shawn P/AAA. DEEPLABCUT/TEST20250303/20250303_4421_01.avi"
os.chdir("/Volumes/lake.s/Active/Shawn P/AAA. DEEPLABCUT/Cropped_HiRes-Shawn-2025-03-14/")
deeplabcut.analyze_videos(config_path, [video_path], videotype='avi',shuffle=0, save_as_csv=True)