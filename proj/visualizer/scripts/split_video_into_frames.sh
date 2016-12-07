#! /bin/bash

# this splits an input video into individual frames
# run as:
#   $ sh split_video_into_frames.sh <path to input video>

# it will output the frames in the input video's directory

vid_dir=$( dirname $1)/frames
mkdir $vid_dir
mkdir $vid_dir/rgb
mkdir $vid_dir/rgba
mkdir $vid_dir/gray
ffmpeg -i $1 -t 30 -ss 00:00:02 -f image2 -pix_fmt rgba $vid_dir/rgba/image-%06d.png
ffmpeg -i $1 -t 30 -ss 00:00:02 -f image2 $vid_dir/rgb/image-%06d.png
ffmpeg -i $1 -t 30 -ss 00:00:02 -f image2 -pix_fmt gray $vid_dir/gray/image-%06d.png
