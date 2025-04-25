# DLC_FILTER_PIPELINE
A pipeline for processing gait videos with a deep lab cut model, with integrated data filtering before creating the final output videos

Start building multi platform version of image:
docker buildx create --use
docker build --platform linux/amd64,linux/arm64 -t shawnpavey/dlc_filter_pipeline:1.0 .
                   
Get your screen IP and allow xhost acces of your IP
ipconfig getifaddr en0
xhost + 192.168.1.144

Run Docker and forward port :0 of your IP to the display of the virtual machine
docker run -p 8888:8888/tcp -it -e DISPLAY=192.168.1.144:0 -v /Users/paveyboys/Desktop/EXAMPLE:/opt/app/host shawnpavey/dlc_filter_pipeline:1.0

Display is from xQuartz on mac, and you need to figure out what your number to make your display port available