import deeplabcut
import sys

config_path = sys.argv[1]
video_path = sys.argv[2]
shuffle = int(sys.argv[3])
bodyparts = sys.argv[4].split(",")

deeplabcut.create_labeled_video(config_path, [video_path], shuffle=shuffle, filtered=False, displayedbodyparts=bodyparts)