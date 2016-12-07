#transform(ArrayStorage(use_float32=True))
import numpy as np
import skimage
import skimage.io
import skimage.color
import sys; sys.path += ['../../compiler']
import util


Display = False

if Display:
    import cv2

#transform(TypeSpecialize(checks=False))
def draw_line(x0, y0, x1, y1, input_img, color):

    #x0, y0, x1, y1 can be double
    #p0 = (x0, y0), p1 = (x1, y1) are start and ending point of a line

    w = input_img.shape[0]
    h = input_img.shape[1]
    
    if input_img.shape[2] == 1 and len(color) == 3:
        color = 0.299 * color[0] + 0.589 * color[1] + 0.114 * color[2]
    
    isvalid_slope = abs(y1 - y0) < abs(x1 - x0)
    
    if not isvalid_slope:
        
        #switch position of x, y
        temp = x0
        x0 = y0
        y0 = temp
        temp = x1
        x1 = y1
        y1 = temp
    
    isvalid_x = x1 > x0
    
    if not isvalid_x:
        
        #swith position of p1, p2
        temp = x0
        x0 = x1
        x1 = temp
        temp = y0
        y0 = y1
        y1 = temp
        
    dx = x1 - x0
    dy = abs(y1 - y0)
    err = dx / 2.0
    
    if y0 < y1:
        y_sign = 1
    else:
        y_sign = -1
    
    y = int(y0)
   
    #points = np.zeros([int(x1) - int(x0) + 1, 2])
    count = 0
    
    for x in range(int(x0), int(x1) + 1):
    
        #points[count, 0] = x
        #points[count, 1] = y
        if isvalid_slope:
            if x > -1 and y > -1 and x < w and y < h:
                input_img[x, y] = color
        else:
            if x > -1 and y > -1 and x < h and y < w:
                input_img[y, x] = color
        count += 1
        err -= dy
        if err < 0:
            y += y_sign
            err += dx
            
    return input_img
    
    #if not isvalid_slope:
        #temp = points[:, 0]
        #points[:, 0] = points[:, 1]
        #points[:, 1] = temp
        
    #return points
    
#transform(TypeSpecialize(checks=False))
def draw_arrow(x0, y0, x1, y1, input_img, color):

    #x0 = start_point[0]
    #y0 = start_point[1]
    
    #x1 = x0 + velocity[0]
    #y1 = y0 + velocity[1]
    
    input_img = draw_line(x0, y0, x1, y1, input_img, color)
    
    dx = x1 - x0
    dy = y1 - y0
    
    #endpoint for one edge of arrow
    x2 = x0 + 0.75 * dx + 0.25 * (3 ** -0.5) * dy
    y2 = y0 + 0.75 * dy - 0.25 * (3 ** -0.5) * dx
    
    x3 = x0 + 0.75 * dx - 0.25 * (3 ** -0.5) * dy
    y3 = y0 + 0.75 * dy + 0.25 * (3 ** -0.5) * dx
    
    input_img = draw_line(x2, y2, x1, y1, input_img, color)
    input_img = draw_line(x3, y3, x1, y1, input_img, color)
    
    return input_img

#transform(TypeSpecialize(checks=False))
def find_color_value(dist, r):

    #dist is a 2D vector with magnitude smaller than 1, r is its manitude 
    #representing a position inside the color wheel
    
    angle = np.arctan2(dist[0], dist[1])
    color = np.array([0.0, 0.0, 0.0])
    
    if angle >= 0 and angle <= 2.0 * np.pi / 3.0:
        
        scale = angle * 3.0 / (2.0 * np.pi)
        color[0] = 1.0 - scale
        color[1] = scale
        
    if angle > 2.0 * np.pi / 3.0:
    
        scale = (angle - 2.0 * np.pi / 3.0) * 3.0 / (2.0 * np.pi)
        color[1] = 1.0 - scale
        color[2] = scale
    
    if angle < -2.0 * np.pi / 3.0:
    
        real_angle = angle + 2.0 * np.pi
        scale = (real_angle - 2.0 * np.pi / 3.0) * 3.0 / (2.0 * np.pi)
        color[1] = 1.0 - scale
        color[2] = scale
        
    if angle < 0 and angle >= -2.0 * np.pi / 3.0:
    
        real_angle = angle + 2.0 * np.pi
        scale = (real_angle - 4.0 * np.pi / 3.0) * 3.0 / (2.0 * np.pi)
        color[2] = 1.0 - scale
        color[0] = scale
    
    color *= r ** 0.25
        
    return color    

#transform(TypeSpecialize(checks=False))
def optical_flow_ssd(input_img1, input_img2):

    """input_img1 = np.zeros([input_color1.shape[0], input_color1.shape[1]])
    input_img2 = np.zeros([input_color2.shape[0], input_color2.shape[1]])
    
    input_img1 = 0.299 * input_color1[:, :, 0] + 0.587 * input_color1[:, :, 1] + 0.114 * input_color1[:, :, 2]
    input_img2 = 0.299 * input_color2[:, :, 0] + 0.587 * input_color2[:, :, 1] + 0.114 * input_color2[:, :, 2]
    """
    
    h = input_img1.shape[0]
    w = input_img1.shape[1]
    
    output_img = np.zeros(input_img1.shape)
    output_img[:, :, :] = input_img2[:, :, :]
    u = np.zeros([h, w])
    v = np.zeros([h, w])
    
    window_size = 21
    offset = int((window_size - 1) / 2.0)
    
    summed_ssd = np.zeros([input_img1.shape[0], input_img1.shape[1], window_size ** 2])
    
    for r in range(window_size):
        for c in range(window_size):
            offset_y = r - offset
            offset_x = c - offset
            ind = window_size * r + c
            summed_ssd[:, :, ind] = calc_summed_ssd(input_img1, input_img2, offset_y, offset_x)
        
    max_of = 0.0
    
    for r in range(offset, h - offset):
        for c in range(offset, w - offset):
            
            ssd = summed_ssd[r + offset, c + offset, :] + summed_ssd[r - offset - 1, c - offset - 1, :] - summed_ssd[r + offset, c - offset - 1, :] - summed_ssd[r - offset - 1, c + offset, :]
            
            ind = np.argmin(ssd)
            
            offset_c = ind % window_size
            offset_r = (ind - offset_c) // window_size
            
            offset_y = offset_r - offset
            offset_x = offset_c - offset
            
            u[r, c] = offset_x
            v[r, c] = offset_y
            
            if (u[r, c] ** 2 + v[r, c] ** 2) ** 0.5 > max_of:
                max_of = (u[r, c] ** 2 + v[r, c] ** 2) ** 0.5
                
    for r in range(offset, h - offset, 10):
        for c in range(offset, w - offset, 10):
        
            scale = ((u[r, c] ** 2 + v[r, c] ** 2) ** 0.5) / max_of
            dist = np.array([u[r, c], v[r, c]]) / max_of
            color = find_color_value(dist, scale)
            output_img = draw_arrow(float(r), float(c), float(r - u[r, c]), float(c - v[r, c]), output_img, color = color)
    
    output_img = output_img[20 : h - 20, 20 : w - 20, :]
    
    return output_img
    
#transform(TypeSpecialize(checks=False))
def calc_summed_ssd(input_img1, input_img2, y, x):

    h = input_img1.shape[0]
    w = input_img2.shape[1]

    dif_img = np.zeros(input_img1.shape)
    clip_img1 = np.zeros(input_img1.shape)
    clip_img2 = np.zeros(input_img1.shape)
    
    if y > 0:
        clip_img1[0:h - y, :, :] = input_img1[y:h, :, :]
    else:
        clip_img1[-y:h, :, :] = input_img1[0:h + y, :, :]
    
    if x > 0:
        clip_img2[:, 0:w - x, :] = clip_img1[:, x:w, :]
    else:
        clip_img2[:, -x:w, :] = clip_img1[:, 0:w + x, :]
        
    dif_img = input_img2 - clip_img2
    
    dif_img = dif_img * dif_img
    
    dif_summed_color = np.zeros([input_img1.shape[0], input_img1.shape[1]])
    
    for i in range(dif_img.shape[2]):
        
        dif_summed_color = dif_summed_color + dif_img[:, :, i]
        
    area = np.zeros(dif_summed_color.shape)
    
    area[1, :] = dif_summed_color[1, :]
    
    for r in range(2, input_img1.shape[0]):
        area[r, :] = area[r - 1, :] + dif_summed_color[r, :]
        
    for c in range(2, input_img1.shape[1]):
        area[:, c] = area[:, c - 1] + area[:, c]
    
    return area     
    
def test(n = None):
    input_img1 = util.image_filename('opt_flow1.png')
    input_img2 = util.image_filename('opt_flow2.png')
    ans = util.test_image_pipeline_filename(optical_flow_ssd, (input_img1, input_img2,), n, grayscale = False, name = 'optical_flow')
    return util.combine_tests([ans])

def main():
    if Display:
        cap = cv2.VideoCapture(0)
        out = cv2.VideoWriter('output.avi', -1, 2.0, (600, 440))
        count = 0
        
        while(count < 20):
            ret, frame1 = cap.read()
            ret, frame2 = cap.read()
            
            frame1 = skimage.img_as_float(frame1)
            frame2 = skimage.img_as_float(frame2)
            
            output_img = optical_flow_ssd(frame1, frame2)
            out.write(output_img)
            
            cv2.imshow('frame', output_img)
            count = count + 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                break
    else:
        test()

if __name__ == '__main__':
    main()
