import collections
import csv
import json
import math
import numpy as np
import os
import PIL.ImageDraw
import PIL.ImageFont
import skimage.io
import skimage.transform
import subprocess
import sys

def print_usage_help(args, opt_args):
    """Prints how to run this program
    """

    print("-----------------------------------------------")
    print("This script renders a 2x2 visualization. Usage:\n")
    print("python3 render_video.py")

    for arg in args:
        print("\t %s [%s]" % (arg, args[arg]))

    print("\nOptional Arguments:")

    for opt_arg in opt_args:
        print("\t %s [%s]" % (opt_arg, opt_args[opt_arg]))

    print("-----------------------------------------------")

def parse_cmd_args(args, args_to_check_for, optional_args):
    """Parses the command line arguments for this program and returns a 
    dictionary mapping the argument name to its value.
    """

    parsed_args = {}
    i = 0

    while i < len(args):
        if args[i] in args_to_check_for or args[i] in optional_args:
            name_without_double_dash = args[i][2:]
            parsed_args[name_without_double_dash] = args[i + 1]
            i += 1
        i += 1

    for arg in args_to_check_for:
        name_without_double_dash = arg[2:]
        if name_without_double_dash not in parsed_args:
            print("ERROR: Missing:")
            print("\t %s [%s]" % (arg, args_to_check_for[arg]))
            print_usage_help(args_to_check_for)
            exit()
        elif name_without_double_dash == "app_versions_to_use" and \
             "ours" not in parsed_args[name_without_double_dash]:
            print("ERROR: must include \"ours\" as an app version with \"--app_versions_to_use\" flag")
            print_usage_help(args_to_check_for)
            exit()

    return parsed_args

def compile_python_app(python_app, output_dir):
    """This runs our compiler on the python app supplied in the command line
    parameters. If a compiled version of the app already exists, we use it and
    skip compilation. If you want to force recompilation of the app, delete its
    compiler output directory.
    """

    starting_dir = os.getcwd()
    python_filename = python_app.split(os.sep)[-1]
    
    full_python_app_path = os.path.abspath(python_app)
    full_compiler_output_path = os.path.abspath(os.path.join(
        output_dir, python_filename[:len(python_filename) - 3]))

    print() # blank line for visual separation in terminal output

    if os.path.isdir(os.path.join(full_compiler_output_path, "final")) and \
       len(os.listdir(os.path.join(full_compiler_output_path, "final"))) > 0:
        print("Found previously compiled python; skipping compilation")
    else:
        print("Couldn't find a previous compiled version of the app")
        print("Commencing compilation... (warning: this may take 30+ minutes)")
        os.chdir("../../compiler")

        compile_cmd = "python3 compiler.py \"%s\" --out-dir \"%s\"" % (
            full_python_app_path, full_compiler_output_path)
        # compile_cmd += " --max-iters 40 "

        print("\nRunning command:\n%s\n" % compile_cmd)
        subprocess.Popen(compile_cmd, shell=True).wait()

        os.chdir(starting_dir)
        print("Done!")

    print() # blank line for visual separation in terminal output

def compile_c_implementation(path_to_c_implementation):
    """Runs make to compile the c implementation

    If there's an error when compiling, this will return False, otherwise, it
    will return True.
    """

    print("\nCompiling C implementation of the app...")

    starting_dir = os.getcwd()
    os.chdir(path_to_c_implementation)

    proc = subprocess.Popen(
        ["make"],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE)
    output, error = proc.communicate()

    os.chdir(starting_dir)

    if "error" in str(error):
        print("Error when compiling %s:" % path_to_c_implementation)
        print("--> Output from make:\n%s" % bytes.decode(output))
        print("--> Error information:\n%s" % bytes.decode(error))
        exit()

    print("Done!\n")

def run_c_implementation(path_to_c_implementation, 
                         output_img,
                         input_img=None,
                         time_arg=None,
                         prev_input_img=None):
    """Runs the c implementation; we assume that it has already been compiled
    correctly.

    Returns the amount of time the program took to run (outputted by the
    program as a float to stdout); if there is a problem with this, returns -1.

    Args:
        path_to_c_implementation, string, path to the c implementation folder
        input_img, string, path to the input image
        output_img, string, path to where to save the output image
    """

    starting_dir = os.getcwd()
    os.chdir(path_to_c_implementation)

    proc = None

    print("input_img=%s" % input_img)
    print("prev_input_img=%s" % prev_input_img)

    cmd = ["./a.out"]

    if prev_input_img and input_img:
        cmd += [prev_input_img, input_img, output_img]
    elif input_img:
        cmd += [input_img, output_img]
    else:
        cmd += [output_img, str(time_arg)]

    print("Running command:\n", " ".join(cmd), "\n")
    # subprocess.Popen(cmd, shell=True).wait()

    # if input_img and not prev_input_img:
    #     proc = subprocess.Popen(
    #         ["./a.out", "\"" + input_img + "\"", "\"" + output_img + "\""],
    #         stdout=subprocess.PIPE, 
    #         stderr=subprocess.PIPE)
    # elif input_img and prev_input_img:

    #     print("using this one!")
    #     print(["./a.out", "\"" + prev_input_img + "\"", "\"" + input_img + "\"",
    #             "\"" + output_img + "\""])
    #     proc = subprocess.Popen(
    #         ["./a.out", "\"" + prev_input_img + "\"", "\"" + input_img + "\"",
    #             "\"" + output_img + "\""],
    #         stdout=subprocess.PIPE, 
    #         stderr=subprocess.PIPE)
    # else:
    proc = subprocess.Popen(
        # ["./a.out", "\"" + output_img + "\"", str(time_arg)],
        cmd,
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE)

    output, error = proc.communicate()

    os.chdir(starting_dir)

    try:
        result = float(bytes.decode(output))
    except ValueError:
        print("Error when attempting to capture timing output from %s" % 
            path_to_c_implementation)
        print("--> Output:\n%s" % bytes.decode(output))
        print("--> Error information:\n%s" % bytes.decode(error))
        result = -1

    return result

def get_frame_filenames_in_order(path_to_frame_folder, ext="png"):
    """Given a path to the folder of frames for the video, this returns a list
    of paths to each filename, sorted so frame 1 is first, frame 2 is second,
    and so on...
    """

    img_paths = []

    for f in os.listdir(path_to_frame_folder):
        if f.split(".")[-1] == ext:
            img_paths.append(os.path.join(path_to_frame_folder, f))

    return sorted(img_paths)

def log_json_time(parsed_args, img_file, json_time_file):
    """This takes the json-formated time outputted by the compiler and saves it
    to a csv file for each app version. 
    """

    with open(json_time_file, "r") as json_file:
        json_data = json.load(json_file)

    for key in json_data.keys():
        framerate_stat_key = "framerate_stats_" + key
        output_img_file = img_file.split(os.sep)[-1]
        filename, ext = output_img_file.split(".")

        with open(parsed_args[framerate_stat_key], "a") as f:
            if key == "ours":
                av_output_img_filename = filename + "." + ext
            else:
                av_output_img_filename = filename + "-" + key + "." + ext

            f.write("%s,%f,%f\n" % (
                os.path.abspath(os.path.join(parsed_args["separate_frames_dir"], 
                    av_output_img_filename)), 
                json_data[key],
                1. / json_data[key]
            ))

def render_separate_frames_no_input(parsed_args):
    """This runs each version of the app (compiled, pypy, numba, c, etc.), times
    how long it takes, and saves the output frames--but doesn't require an input
    image.

    We assume that the app versions have all been compiled/prepared beforehand.
    """

    output_dir = parsed_args["output_dir"]
    frame_output_dir = parsed_args["separate_frames_dir"]
    python_filename = parsed_args["python_app"].split(os.sep)[-1]

    full_c_impl_path = os.path.abspath(parsed_args["c_version"])
    full_python_app_path = os.path.abspath(parsed_args["python_app"])
    full_compiler_output_path = os.path.abspath(os.path.join(
        output_dir, python_filename[:len(python_filename) - 3]))
    time_output_filepath = os.path.abspath(
        os.path.join(frame_output_dir, "time.txt"))

    i = 0
    stop_index = int(float(parsed_args["output_duration"]) * \
        float(parsed_args["output_fps"]))

    while i < stop_index:
        print("\n-------------------------------------------")
        print("Rendering frame %d of %d (%f%% complete)\n" % (i + 1, stop_index,
            float(i) * 100. / stop_index))

        c_output_img_filename = os.path.abspath(
            os.path.join(frame_output_dir, "image-%06d-c.png" % i))
        compiler_output_img_filename = os.path.abspath(
            os.path.join(frame_output_dir, "image-%06d.png" % i))

        cmd = \
            ("python compiler.py \"%s\" " % full_python_app_path) + \
            ("--out-dir \"%s\" " % full_compiler_output_path) + \
            ("--no-tune ") + \
            ("--in-image \"%s\" " % "image-%06d.png" % i) + \
            ("--out-image \"%s\" " % compiler_output_img_filename) + \
            ("--out-time \"%s\" " % time_output_filepath) + \
            (parsed_args["app_versions_to_ignore_compiler_flags"]) + \
            ("--args %f ") % float(i) + \
            ("--ntests %d " % 5)

        print("Running command:\n", cmd, "\n")

        starting_dir = os.getcwd()
        os.chdir("../../compiler")
        subprocess.Popen(cmd, shell=True).wait()
        os.chdir(starting_dir)
        log_json_time(parsed_args, "image-%06d.png" % i, time_output_filepath)

        if parsed_args["run_c_implementation"]:
            c_runtime = run_c_implementation(full_c_impl_path,
                c_output_img_filename, input_img=None, time_arg=i)
            with open(parsed_args["framerate_stats_c"], "a") as f:
                f.write("%s,%f,%f\n" % (c_output_img_filename, c_runtime, 
                    1. / c_runtime))

        i += 1

def get_input_img_filename(initial_filepath, parsed_args):
    """
    """

    frame_output_dir = parsed_args["separate_frames_dir"]
    filename, ext = initial_filepath.split(os.sep)[-1].split(".")

    path_to_img_parent_dir = os.sep.join(initial_filepath.split(os.sep)[:-2])
    input_rgb_img_path = os.path.join(path_to_img_parent_dir, "rgb")
    input_gray_img_path = os.path.join(path_to_img_parent_dir, "gray")

    input_rgba_img_filename = os.path.abspath(initial_filepath)
    input_rgb_img_filename = os.path.abspath(os.path.join(
        input_rgb_img_path, filename + "." + ext))
    input_gray_img_filename = os.path.abspath(os.path.join(
        input_gray_img_path, filename + "." + ext))

    c_out_img = os.path.abspath(os.path.join(frame_output_dir, 
        filename + "-c." + ext))
    py_out_img = os.path.abspath(os.path.join(frame_output_dir, 
        filename + "." + ext))

    # python input image:

    py_in_img = input_rgba_img_filename

    if parsed_args["use_3_channel_img_for_python"]:
        py_in_img = input_rgb_img_filename
    elif parsed_args["use_grayscale_img_for_python"]:
        py_in_img = input_gray_img_filename

    # c input image:

    c_in_img = input_rgb_img_filename

    if parsed_args["use_4_channel_img_for_c"]:
        c_in_img = input_rgba_img_filename
    elif parsed_args["use_grayscale_img_for_c"]:
        c_in_img = input_gray_img_filename

    return py_in_img, c_in_img, py_out_img, c_out_img

def render_separate_frames(parsed_args):
    """This runs each version of the app (compiled, pypy, numba, c, etc.), times
    how long it takes, and saves the output frames

    We assume that the app versions have all been compiled/prepared beforehand.
    """

    input_dir = parsed_args["input_frame_dir"]
    output_dir = parsed_args["output_dir"]
    frame_output_dir = parsed_args["separate_frames_dir"]
    python_filename = parsed_args["python_app"].split(os.sep)[-1]

    full_c_impl_path = os.path.abspath(parsed_args["c_version"])
    full_python_app_path = os.path.abspath(parsed_args["python_app"])
    full_compiler_output_path = os.path.abspath(os.path.join(
        output_dir, python_filename[:len(python_filename) - 3]))
    time_output_filepath = os.path.abspath(
        os.path.join(frame_output_dir, "time.txt"))

    frame_filepaths = get_frame_filenames_in_order(
        os.path.join(input_dir, "rgba"))
    i = 0
    stop_index = min(len(frame_filepaths), int(float(
        parsed_args["output_duration"]) * float(parsed_args["output_fps"])))

    while i < stop_index:
        print("\n-------------------------------------------")
        print("Rendering frame %d of %d (%f%% complete)\n" % (i + 1, stop_index,
            float(i) * 100. / stop_index))

        if parsed_args["use_2_input_frames"] and i == 0:
            i += 1
            continue

        # frame_filepath = frame_filepaths[i]
        # filename, ext = frame_filepath.split(os.sep)[-1].split(".")

        # path_to_img_parent_dir = os.sep.join(frame_filepath.split(os.sep)[:-2])
        # input_rgb_img_path = os.path.join(path_to_img_parent_dir, "rgb")
        # input_gray_img_path = os.path.join(path_to_img_parent_dir, "gray")

        # input_rgba_img_filename = os.path.abspath(frame_filepath)
        # input_rgb_img_filename = os.path.abspath(os.path.join(
        #     input_rgb_img_path, filename + "." + ext))
        # input_gray_img_filename = os.path.abspath(os.path.join(
        #     input_gray_img_path, filename + "." + ext))

        # c_output_img_filename = os.path.abspath(
        #     os.path.join(frame_output_dir, filename + "-c." + ext))
        # compiler_output_img_filename = os.path.abspath(
        #     os.path.join(frame_output_dir, filename + "." + ext))

        # in_image = input_rgba_img_filename

        # if parsed_args["use_3_channel_img_for_python"]:
        #     in_image = input_rgb_img_filename
        # elif parsed_args["use_grayscale_img_for_python"]:
        #     in_image = input_gray_img_filename

        compiler_in_img, c_in_img, compiler_out_img, c_out_img = \
            get_input_img_filename(frame_filepaths[i], parsed_args)

        cmd = \
            ("python compiler.py \"%s\" " % full_python_app_path) + \
            ("--out-dir \"%s\" " % full_compiler_output_path) + \
            ("--no-tune ") + \
            ("--in-image \"%s\" " % compiler_in_img) + \
            ("--out-image \"%s\" " % compiler_out_img) + \
            ("--out-time \"%s\" " % time_output_filepath) + \
            ("--ntests %d " % 10) + \
            (parsed_args["app_versions_to_ignore_compiler_flags"])

        if parsed_args["use_2_input_frames"]:
            compiler_in_img_before, _, _, _ = \
                get_input_img_filename(frame_filepaths[i - 1], parsed_args)
            cmd += " --args " + ("\'img1=\"%s\",img2=\"%s\"\'" % (
                compiler_in_img_before, compiler_in_img))

        print("Running command:\n", cmd)

        starting_dir = os.getcwd()
        os.chdir("../../compiler")
        subprocess.Popen(cmd, shell=True).wait()
        os.chdir(starting_dir)
        log_json_time(parsed_args, compiler_in_img, 
            time_output_filepath)

        if parsed_args["run_c_implementation"]:
            # if parsed_args["use_4_channel_img_for_c"]:
            #     c_runtime = run_c_implementation(full_c_impl_path, 
            #         c_output_img_filename, input_img=input_rgba_img_filename)
            # elif parsed_args["use_grayscale_img_for_c"]:
            #     c_runtime = run_c_implementation(full_c_impl_path, 
            #         c_output_img_filename, input_img=input_gray_img_filename)
            # else:
            #     c_runtime = run_c_implementation(full_c_impl_path, 
            #         c_output_img_filename, input_img=input_rgb_img_filename)

            c_runtime = -1

            if parsed_args["use_2_input_frames"]:
                _, c_in_img_before, _, _ = \
                    get_input_img_filename(frame_filepaths[i - 1], parsed_args)
                c_runtime = run_c_implementation(full_c_impl_path, c_out_img,
                    input_img=c_in_img, prev_input_img=c_in_img_before)
            else:
                c_runtime = run_c_implementation(full_c_impl_path, c_out_img,
                    input_img=c_in_img)

            with open(parsed_args["framerate_stats_c"], "a") as f:
                f.write("%s,%f,%f\n" % (c_out_img, c_runtime, 
                    1. / c_runtime))

        i += 1

def get_quad_csv(quad_dir_csv_file):
    """Given the framerate csv file for a 2x2 video quadrant, return a list
    whose rows represent the rows of the CSV file and the columns also match the
    CSV file's columns

    We also append a column to each row, which represents the current time at
    that specific frame. Output is:
    [
        [<path to image frame>, <time spent rendering>, <fps>, <current time>],
        ...,
    ]
    """

    result = []
    current_time = 0.0
    i_am_on_the_first_row_of_the_csv_file = True

    with open(quad_dir_csv_file, newline="") as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True, delimiter=",")

        for row in reader:
            # skip header row in the csv file:
            if not i_am_on_the_first_row_of_the_csv_file:
                result.append(row + [current_time])
                current_time += float(row[1])
            else:
                i_am_on_the_first_row_of_the_csv_file = False

    return result

def resize_img(np_image, max_height, max_width):
    """Given an image (as a numpy array), this will resize it to fit in a 
    bounding box of the supplied dimensions, while still keeping the aspect 
    ratio of the original image.

    Used: http://stackoverflow.com/a/3008966
    """

    if len(np_image.shape) == 2: # check if it's a grayscale image
        np_image = np.dstack((np_image, np_image, np_image))

    img_height = np_image.shape[0]
    img_width = np_image.shape[1]
    resize_height = max_height
    resize_width = max_width

    original_ratio = float(img_width) / img_height
    resize_ratio = float(resize_width) / resize_height

    if original_ratio > resize_ratio:
        resize_height = resize_width / original_ratio
    else:
        resize_width = resize_height * original_ratio

    resize_height = int(resize_height)
    resize_width = int(resize_width)

    return skimage.transform.resize(np_image, (resize_height, resize_width))*255

def draw_quad_text(PIL_img, fname, fps, font, color, pad=15, title=None):
    """This draws text on a quadrant image for the output video

    Args:
        PIL_img, PIL.Image, PIL version of the image to draw the text on (this 
            *must* be a PIL image because PIL allows us to render text)
        fname, string, filename for the frame, this tells us which app variant
            was used to render the frame
        fps, float/string, this is the framerate to draw on the frame
        font, PIL.ImageFont, font to use for text rendering
        color, tuple(int), color for the text
        pad, int, how many pixels to pad from the edge of the frame
    """

    width, height = PIL_img.size
    draw = PIL.ImageDraw.Draw(PIL_img)
    img_filename = fname.split(os.sep)[-1]

    # filename should be "image-######-<app version>.png" or "image-######.png"
    # if it's our compiler; first we assume it's our compiler:

    display_str = "Our Compiler"

    if "-c" in img_filename:
        display_str = "Handwritten C"

    elif "-numba" in img_filename:
        display_str = "Numba"

    elif "-unpython" in img_filename:
        display_str = "unPython"

    elif "-pypy" in img_filename:
        display_str = "PyPy"

    elif "-python" in img_filename:
        display_str = "Raw Python"

    display_str += ": {0:.4g} fps".format(float(fps))
    draw.text((pad, pad), display_str, fill=color, font=font)

def render_output_frames(video_cube, parsed_args):
    """Given a row of the video_cube (see render_video() for a description of
    the video_cube), this will combine the quadrant-frames into a single, 2x2,
    output frame.
    """

    # used http://stackoverflow.com/a/5430111 for help with text rendering

    output_shape = (720, 1280, 3)
    font = PIL.ImageFont.load("pilfonts/helvR24.pil")
    font_color = (255, 255, 255)

    title_gap = 76
    title_str = "Application: " + parsed_args["app_name"]
    title_w, title_h = font.getsize(title_str)
    title_img = PIL.Image.new("RGB", (output_shape[1], title_gap))

    max_quad_height = int((output_shape[0] - title_gap) / 2)
    max_quad_width = int(output_shape[1] / 2)

    PIL.ImageDraw.Draw(title_img).text(
        (
            int((output_shape[1] - title_w) / 2),
            int((title_gap - title_h) / 2)
        ),
        title_str, fill=font_color, font=font)

    for i in range(len(video_cube)):
        print("Rendering frame %d of %d" % (i, len(video_cube)))

        # draw in title (by default, PIL.Image.new() is a black image):

        output_img = np.zeros(output_shape)
        output_img[:title_gap, :, :3] = np.array(title_img)[:, :, :]

        # draw in frames:

        for j in range(4):
            np_img = skimage.io.imread(video_cube[i][j][0])

            if np.max(np_img) > 255:
                np_img /= 255.0
                np_img = np.clip(np_img, 0, 255)

            resized_img = resize_img(np_img.astype(np.uint8), max_quad_height, 
                max_quad_width)

            start_r = title_gap + \
                int((max_quad_height - resized_img.shape[0]) / 2)
            stop_r = title_gap + resized_img.shape[0] + \
                int((max_quad_height - resized_img.shape[0]) / 2)
            start_c = int((max_quad_width - resized_img.shape[1]) / 2)
            stop_c = resized_img.shape[1] + \
                int((max_quad_width - resized_img.shape[1]) / 2)

            if j == 1:
                start_r = title_gap + \
                    int((max_quad_height - resized_img.shape[0]) / 2)
                stop_r = title_gap + resized_img.shape[0] + \
                    int((max_quad_height - resized_img.shape[0]) / 2)
                start_c = max_quad_width + \
                    int((max_quad_width - resized_img.shape[1]) / 2)
                stop_c = max_quad_width + resized_img.shape[1] + \
                    int((max_quad_width - resized_img.shape[1]) / 2)

            if j == 2:
                start_r = title_gap + max_quad_height + \
                    int((max_quad_height - resized_img.shape[0]) / 2)
                stop_r = title_gap + max_quad_height + resized_img.shape[0] + \
                    int((max_quad_height - resized_img.shape[0]) / 2)
                start_c = int((max_quad_width - resized_img.shape[1]) / 2)
                stop_c = resized_img.shape[1] + \
                    int((max_quad_width - resized_img.shape[1]) / 2)

            if j == 3:
                start_r = title_gap + max_quad_height + \
                    int((max_quad_height - resized_img.shape[0]) / 2)
                stop_r = title_gap + max_quad_height + resized_img.shape[0] + \
                    int((max_quad_height - resized_img.shape[0]) / 2)
                start_c = max_quad_width + \
                    int((max_quad_width - resized_img.shape[1]) / 2)
                stop_c = max_quad_width + resized_img.shape[1] + \
                    int((max_quad_width - resized_img.shape[1]) / 2)

            output_img[start_r:stop_r, start_c:stop_c, :3] = resized_img[:,:,:3]

            start_r = title_gap
            stop_r = title_gap + max_quad_height
            start_c = 0
            stop_c = max_quad_width

            if j == 1:
                start_r = title_gap
                stop_r = title_gap + max_quad_height
                start_c = max_quad_width
                stop_c = 2 * max_quad_width

            if j == 2:
                start_r = title_gap + max_quad_height
                stop_r = title_gap + 2 * max_quad_height
                start_c = 0
                stop_c = max_quad_width

            if j == 3:
                start_r = title_gap + max_quad_height
                stop_r = title_gap + 2 * max_quad_height
                start_c = max_quad_width
                stop_c = 2 * max_quad_width

            np_img = output_img[start_r:stop_r, start_c:stop_c, :3]
            text_img = PIL.Image.fromarray(np_img.astype(np.uint8))
            draw_quad_text(text_img, video_cube[i][j][0], video_cube[i][j][1], 
                font, font_color)
            output_img[start_r:stop_r, start_c:stop_c, :3] = \
                np.array(text_img)[:, :, :3]


        skimage.io.imsave(os.path.join(parsed_args["combined_frames_dir"], 
            "image-%06d.png" % i), output_img / 255.0)

def combine_frames(parsed_args):
    """This combines the individual frames rendered by each app variant into a 
    single set of output frames that can be later joined into a movie with 
    ffmpeg. 

    If an app variant renders faster than the output framerate, it's quadrant in
    the output frame will displayed at the output framerate. If an app variant
    renders slower than the output framerate, it renders a frame, then keeps 
    displaying that frame for an amount of time equivalent to its framerate.

    Another way to think about the output of this function, is that it's trying
    to make an output video which looks like each app variant is running in
    parallel on live incoming footage. When a app variant finishes rendering a 
    frame, it tries to render the next newest frame.
    """

    app_versions_list = parsed_args["app_versions_to_use"].split(",")
    csvs = []

    for i in range(4):
        csvs.append(get_quad_csv(parsed_args["framerate_stats_" + \
            app_versions_list[i]]))

    assert len(csvs[0]) == len(csvs[1]), \
        "len(csvs[0]) = %d, len(csvs[1]) = %d" % (len(csvs[0]), len(csvs[1]))
    assert len(csvs[0]) == len(csvs[2]), \
        "len(csvs[0]) = %d, len(csvs[1]) = %d" % (len(csvs[0]), len(csvs[2]))
    assert len(csvs[0]) == len(csvs[3]), \
        "len(csvs[0]) = %d, len(csvs[1]) = %d" % (len(csvs[0]), len(csvs[3]))

    total_frames = len(csvs[0])
    video_cube = []
    virtual_render_time_left = [0, 0, 0, 0]
    prev_img = ["", "", "", ""]
    prev_fps = [0, 0, 0, 0]
    dt = 1 / float(parsed_args["output_fps"])

    # first pass, just put in the frames at their appropriate time in the cube:

    for i in range(total_frames):
        video_cube.append([])
        video_cube[i] = [[] for _ in range(4)]

        for j in range(4):
            if virtual_render_time_left[j] <= 0:
                video_cube[i][j] = [csvs[j][i][0], float(csvs[j][i][2])]
                prev_img[j] = csvs[j][i][0]
                prev_fps[j] = float(csvs[j][i][2])
                virtual_render_time_left[j] = float(csvs[j][i][1])

            video_cube[i][j] = [prev_img[j], prev_fps[j]]
            virtual_render_time_left[j] -= dt

    render_output_frames(video_cube, parsed_args)

def init_output_dirs(parsed_args):
    """Creates the necessary output directories and files
    """

    output_dir = parsed_args["output_dir"]

    if not os.path.isdir(output_dir):
        print("Output directory (\"%s\") doesn't exist; creating it..." % 
            output_dir)
        os.makedirs(output_dir)

    python_filename = parsed_args["python_app"].split(os.sep)[-1]
    compiler_output_dir = os.path.join(
        output_dir, python_filename[:len(python_filename) - 3]) # chop off ".py"

    if not os.path.isdir(compiler_output_dir):
        print("Compiler output directory (\"%s\") doesn't exist; creating it..." % compiler_output_dir)
        os.makedirs(compiler_output_dir)

    if "app_title_str" in parsed_args:
        parsed_args["app_name"] = parsed_args["app_title_str"]
    else:
        parsed_args["app_name"] = " ".join(
            python_filename[:len(python_filename) - 3].split("_")).title()

    separate_frames_dir = os.path.join(output_dir, "separate_frames")
    parsed_args["separate_frames_dir"] = separate_frames_dir

    if not os.path.isdir(separate_frames_dir):
        print("Separate frames directory (\"%s\") doesn't exist; creating it..." % separate_frames_dir)
        os.makedirs(separate_frames_dir)

    combined_frames_dir = os.path.join(output_dir, "combined_frames")
    parsed_args["combined_frames_dir"] = combined_frames_dir

    if not os.path.isdir(combined_frames_dir):
        print("Combined frames directory (\"%s\") doesn't exist; creating it..." %combined_frames_dir)
        os.makedirs(combined_frames_dir)

    if parsed_args["no_input_img"] == "True":
        parsed_args["no_input_img"] = True
    else:
        parsed_args["no_input_img"] = False

    if "use_4_channel_img_for_c" in parsed_args and \
       parsed_args["use_4_channel_img_for_c"] == "True":
        parsed_args["use_4_channel_img_for_c"] = True
    else:
        parsed_args["use_4_channel_img_for_c"] = False

    if "use_3_channel_img_for_python" in parsed_args and \
       parsed_args["use_3_channel_img_for_python"] == "True":
        parsed_args["use_3_channel_img_for_python"] = True
    else:
        parsed_args["use_3_channel_img_for_python"] = False

    if "use_grayscale_img_for_python" in parsed_args and \
       parsed_args["use_grayscale_img_for_python"] == "True":
        parsed_args["use_grayscale_img_for_python"] = True
    else:
        parsed_args["use_grayscale_img_for_python"] = False

    if "use_grayscale_img_for_c" in parsed_args and \
       parsed_args["use_grayscale_img_for_c"] == "True":
        parsed_args["use_grayscale_img_for_c"] = True
    else:
        parsed_args["use_grayscale_img_for_c"] = False

    if "use_2_input_frames" in parsed_args and \
       parsed_args["use_2_input_frames"] == "True":
        parsed_args["use_2_input_frames"] = True
    else:
        parsed_args["use_2_input_frames"] = False

    # parse the "--app_versions_to_use" flag:

    app_versions_list = parsed_args["app_versions_to_use"].split(",")
    ignore_compiler_flags = ""

    if "numba" not in app_versions_list:
        ignore_compiler_flags += "--no-numba "

    if "pypy" not in app_versions_list:
        ignore_compiler_flags += "--no-pypy "

    if "unpython" not in app_versions_list:
        ignore_compiler_flags += "--no-unpython "

    if "python" not in app_versions_list:
        ignore_compiler_flags += "--no-python "

    parsed_args["app_versions_to_ignore_compiler_flags"] = ignore_compiler_flags

    if "c" in app_versions_list:
        parsed_args["run_c_implementation"] = True
    else:
        parsed_args["run_c_implementation"] = False

    # create csv framerate log files:

    parsed_args["framerate_stats_ours"] = os.path.join(
        separate_frames_dir, "framerate_stats_ours.csv")
    with open(parsed_args["framerate_stats_ours"], "w") as f:
        f.write("filename,time spent rendering (sec), fps\n")

    if "--no-python" not in ignore_compiler_flags:
        parsed_args["framerate_stats_python"] = os.path.join(
            separate_frames_dir, "framerate_stats_python.csv")
        with open(parsed_args["framerate_stats_python"], "w") as f:
            f.write("filename,time spent rendering (sec), fps\n")

    if "--no-numba" not in ignore_compiler_flags:
        parsed_args["framerate_stats_numba"] = os.path.join(
            separate_frames_dir, "framerate_stats_numba.csv")
        with open(parsed_args["framerate_stats_numba"], "w") as f:
            f.write("filename,time spent rendering (sec), fps\n")

    if "--no-pypy" not in ignore_compiler_flags:
        parsed_args["framerate_stats_pypy"] = os.path.join(
            separate_frames_dir, "framerate_stats_pypy.csv")
        with open(parsed_args["framerate_stats_pypy"], "w") as f:
            f.write("filename,time spent rendering (sec), fps\n")

    if "--no-unpython" not in ignore_compiler_flags:
        parsed_args["framerate_stats_unpython"] = os.path.join(
            separate_frames_dir, "framerate_stats_unpython.csv")
        with open(parsed_args["framerate_stats_unpython"], "w") as f:
            f.write("filename,time spent rendering (sec), fps\n")

    if parsed_args["run_c_implementation"]:
        parsed_args["framerate_stats_c"] = os.path.join(
            separate_frames_dir, "framerate_stats_c.csv")
        with open(parsed_args["framerate_stats_c"], "w") as f:
            f.write("filename,time spent rendering (sec), fps\n")

if __name__ == "__main__":
    args_to_check_for = collections.OrderedDict()

    args_to_check_for["--python_app"] = "path to python version of app"
    args_to_check_for["--c_version"] = \
        "path to the folder of the c implementation of the app"
    args_to_check_for["--input_frame_dir"] = \
        "path to the input video frame directory"
    args_to_check_for["--output_dir"] = \
        "path to the output dir for frames, compiled code, etc."
    args_to_check_for["--output_duration"] = \
        "how long (in seconds) output video should be"
    args_to_check_for["--output_fps"] = "framerate of output video"
    args_to_check_for["--app_versions_to_use"] = \
        "app versions to include in the output video, as a comma separated list of length 4; possible options are: ours, numba, pypy, unpython, python, c (example: \"ours,numba,unpython,c\")"
    args_to_check_for["--no_input_img"] = "whether or not to use an input image (use \"True\" or \"False\")"

    optional_args = collections.OrderedDict()

    optional_args["--app_title_str"] = "Title string to display for the app in the output video (surround string with quotes)"
    optional_args["--use_4_channel_img_for_c"] = "whether or not to use a 4-channel rgba image for the c program (\"True\" or \"False\")"
    optional_args["--use_3_channel_img_for_python"] = "whether or not to use a 4-channel rgba image for the python program (\"True\" or \"False\")"
    optional_args["--use_grayscale_img_for_python"] = "whether or not to use a grayscale image for the python program (\"True\" or \"False\")"
    optional_args["--use_grayscale_img_for_c"] = "whether or not to use a grayscale image for the c program (\"True\" or \"False\")"
    optional_args["--use_2_input_frames"] = "whether or not to use a 2 input_images for the program (\"True\" or \"False\")"
    

    if len(sys.argv) < 3:
        print_usage_help(args_to_check_for, optional_args)
        exit()

    parsed_args = parse_cmd_args(sys.argv, args_to_check_for, optional_args)
    init_output_dirs(parsed_args)

    compile_python_app(parsed_args["python_app"], parsed_args["output_dir"])

    if parsed_args["run_c_implementation"]:
        compile_c_implementation(parsed_args["c_version"])
    
    if parsed_args["no_input_img"]:
        render_separate_frames_no_input(parsed_args)
    else:
        render_separate_frames(parsed_args)

    combine_frames(parsed_args)