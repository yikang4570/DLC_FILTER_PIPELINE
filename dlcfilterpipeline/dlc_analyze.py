import deeplabcut
import os
import sys
import json

config_path = sys.argv[1]
video_path = sys.argv[2]
shuffle = int(sys.argv[3])

video_dir, video_filename = os.path.split(video_path)
video_name, _ = os.path.splitext(video_filename)
files_before = set(os.listdir(video_dir))

deeplabcut.analyze_videos(config_path, [video_path], videotype='avi',shuffle=0, save_as_csv=True)

files_after = set(os.listdir(video_dir))
new_files = files_after - files_before

h5_file = None
for file in new_files:
    if file.endswith('.h5') and video_name in file:
        h5_file = file
        break
if h5_file is None:
    raise FileNotFoundError("No .h5 file was generated. Please check the analysis process.")

scorer_suffix = h5_file.replace(video_name, '').replace('.h5', '')
json_object = json.dumps({'model_name':scorer_suffix})
with open("dlcfilterpipeline" + os.sep + "model_name.json", "w") as outfile:
    outfile.write(json_object)
