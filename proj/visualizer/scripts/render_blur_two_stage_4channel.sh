python render_video2.py \
    --python_app ../../apps/blur_two_stage/blur_two_stage_4channel.py \
    --c_version ../../apps/blur_two_stage/c/ \
    --input_frame_dir ../input_vids/fireworks_hd/frames/ \
    --output_dir ../input_vids/fireworks_hd/output_blur_two_stage/ \
    --output_duration $1 \
    --output_fps 30 \
    --app_versions_to_use ours,numba,unpython,c \
    --no_input_img False \
    --app_title_str "Two-Stage Blur" \
    # --use_3_channel_img_for_python True
    # --use_grayscale_img_for_python True \
    # --use_grayscale_img_for_c True