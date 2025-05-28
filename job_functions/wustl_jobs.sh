#!/bin/bash

# Define your compute group and Docker image
COMPUTE_GROUP="compute-lake.s"
COMPUTE_USERNAME="shawn.pavey"
GROUP_NAME="dlc"
DOCKER_IMAGE="shawnpavey/dlc_filter_pipeline:2.2"

# Export the volume mount
export LSF_DOCKER_VOLUMES="/storage1/fs1/lake.s/Active:/data"
export LSF_DOCKER_WORKDIR=/opt/app

# Read the file list and submit a job for each file
while IFS= read -r FILENAME; do
 [ -z "$FILENAME" ] && continue

  bsub \
    -g /${COMPUTE_USERNAME}/${GROUP_NAME}\
    -q general \
    -R "rusage[mem=4000]" \
    -Ne \
    -M 4000 \
    -a "docker($DOCKER_IMAGE)" \
    /opt/conda/bin/python /opt/app/run_container.py "/data/$FILENAME"
done < /storage1/fs1/lake.s/Active/Shawn\ P/AAA.\ DEEPLABCUT/TEST20250303/wustl_cluster/file_list.txt

#-G $COMPUTE_GROUP \