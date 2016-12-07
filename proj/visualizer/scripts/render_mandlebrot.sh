python render_video2.py \
    --python_app ../../apps/mandelbrot/mandelbrot_animate.py \
    --c_version ../../apps/mandelbrot/c/ \
    --input_frame_dir ../input_vids/fireworks_hd/frames/ \
    --output_dir ../input_vids/fireworks_hd/output_mandelbrot/ \
    --output_duration $1 \
    --output_fps 30 \
    --app_versions_to_use ours,numba,unpython,c \
    --no_input_img True \
    --app_title_str "Mandlebrot"
