import dlcfilterpipeline as dlp
import argparse

parser = argparse.ArgumentParser(description="Process a file by filename.")
parser.add_argument("filename", help="Name of the file to process")
args = parser.parse_args()
video = args.filename

using_docker = True
if using_docker:
    import yaml
    yaml_file_path = "dlcfilterpipeline/config.yaml"
    with open(yaml_file_path, 'r') as file: data = yaml.safe_load(file)
    data['project_path'] = '/opt/app/dlcfilterpipeline'
    with open(yaml_file_path, 'w') as file: yaml.dump(data, file, default_flow_style=False)
    dlc_venv_path = "/opt/conda/bin/python"
else:
    dlc_venv_path = "/opt/anaconda3/envs/DEEPLABCUT/bin/python"

inputs = {'video':video,#"/Users/paveyboys/Desktop/EXAMPLE/SPENCE_EXAMPLE/AVI/20250303_4421_01.avi",
          'kwargs':{
            'defaultdir':"/Users/paveyboys/Desktop/EXAMPLE/",
            'save_files':True,
            'model_name':'DLC_Resnet50_Cropped_HiResMar14shuffle0_snapshot_190',
            'analyze_videos':True,
            'create_videos':True,
            'animate':False,
            'graph_video_frame_step': 10,
            'dlc_venv_path':dlc_venv_path,
            'footstrike_algorithm':'position',
            'bodyparts': ['FL_gen', 'FR_gen', 'BL_gen', 'BR_gen']}}

pipe = dlp.manager(inputs['video'],**inputs['kwargs'])
pipe.batch_process()