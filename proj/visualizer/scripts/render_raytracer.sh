python render_video2.py \
    --python_app ../../apps/raytracer/raytracer_short_simplified_animate.py \
    --c_version ../../apps/raytracer/c/ \
    --input_frame_dir ../input_vids/fireworks_hd/frames/ \
    --output_dir ../input_vids/fireworks_hd/output_raytracer/ \
    --output_duration $1 \
    --output_fps 30 \
    --app_versions_to_use ours,numba,unpython,c \
    --no_input_img True \
    --app_title_str "Raytracer"
