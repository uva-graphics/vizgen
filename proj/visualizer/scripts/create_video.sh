#! /bin/bash

# This combines the frames outputed by render_video.py into an x264 MP4 video
# Run as:
#   $ sh create_video.sh <path to folder containing frames>

# This will output the video in the folder you passed in

ffmpeg -i $1/image-%6d.png -vf fps=30 -pix_fmt yuv420p -c:v libx264 $1/output.mp4