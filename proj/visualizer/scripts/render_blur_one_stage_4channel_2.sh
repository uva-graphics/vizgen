python render_video2.py \
    --python_app ../../apps/blur_one_stage/blur_one_stage_4channel.py \
    --c_version ../../apps/blur_one_stage/c/ \
    --input_frame_dir ../input_vids/the_nature_of_montenegro/frames/ \
    --output_dir ../input_vids/the_nature_of_montenegro/output_blur_one_stage_4channel/ \
    --output_duration $1 \
    --output_fps 30 \
    --app_versions_to_use ours,numba,unpython,c \
    --no_input_img False \
    --app_title_str "One-Stage Blur" \
#    --use_3_channel_img_for_python True