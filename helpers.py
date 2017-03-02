import math
import numpy as np
import cv2

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines_v1(img, lines, color=[255, 0, 0], thickness=1, **kwargs):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img

def draw_lines_v2(img, lines, color=[255, 0, 0], thickness=1, **kwargs):

    x_segment = img.shape[1] / 24.

    left_x_segment = x_segment * 11.
    right_x_segment = x_segment * 13.

    x_left, y_left, x_right, y_right = [], [], [], []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # cv2.line(img, (x1, y1), (x2, y2), color, thickness)

            # because axis start top left, left-hand lane lines have a negative gradient; right-hand positive.
            try:
                dy_dx = (y2 - y1) / (x2 - x1)
            except ZeroDivisionError:
                dy_dx = 0

            # left side. alocate line by position on grid and gradient
            if x1 < left_x_segment and x2 < left_x_segment and dy_dx < 0:

                x_left += [x1, x2]
                y_left += [y1, y2]

            elif x1 > right_x_segment and x2 > right_x_segment and dy_dx > 0:

                x_right += [x1, x2]
                y_right += [y1, y2]

    # fit left line with np.polyfit
    if x_left:
        m, b = np.polyfit(np.asarray(x_left), np.asarray(y_left), 1)
        start, end = get_line_coords(img, m, b)
        cv2.line(img, start, end, color=[255, 0, 0], thickness=10)

    # fit right line with np.polyfit
    if x_right:
        m, b = np.polyfit(np.asarray(x_right), np.asarray(y_right), 1)
        start, end = get_line_coords(img, m, b)
        cv2.line(img, start, end, color=[255, 0, 0], thickness=10)

    return img

def draw_lines_v3(img, lines, color=[255, 0, 0], thickness=1, **kwargs):

    debug = False
    deviation = 0.1

    # get historical m values
    history = kwargs.get('history')
    previous_m_left = history.get('left').get('m')[-1]
    previous_m_right = history.get('right').get('m')[-1]

    left_lines, right_lines = [], []

    for line in lines:
        for x1, y1, x2, y2 in line:

            # because axis start top left, left-hand lane lines have a negative gradient; right-hand positive.
            m, b = np.polyfit(np.asarray([x1, x2]), np.asarray([y1, y2]), 1)

            # if dy_dx is within 'deviation' of previous m then include line in calculating this m
            # dealing with negative gradients so be careful with conditionals...
            # could also try including lines based on previous 'b' values too
            if m > previous_m_left * (1 + deviation) and m < previous_m_left * (1 - deviation):
                left_lines.append((x1, y1, x2, y2))

            elif m < previous_m_right * (1 + deviation) and m > previous_m_right * (1 - deviation):
                right_lines.append((x1, y1, x2, y2))

    if debug:
        for x1, y1, x2, y2 in left_lines + right_lines:
            cv2.line(img, (x1, y1), (x2, y2), color=[255,0,0], thickness=10)

    left_m, left_b = draw_line(img, left_lines, 'left', history)
    right_m, right_b = draw_line(img, right_lines, 'right', history)

    # calcuate intersection of lane lines and add to history
    x = (right_b - left_b) / (left_m - right_m)
    y = (left_m * x) + left_b
    history.get('intersection').append((x, y))

    if debug:
        cv2.circle(img, center=(int(x), int(y)), radius=10, color=[0,0,255], thickness=1)

    return img

def draw_line(img, lines, side, history):

    if len(lines) <= 2:
        # bit of a hack but works ok
        # print("Setting lines from previous frame on the {}-hand lane".format(side))
        m = history.get(side).get('m')[-1]
        b = history.get(side).get('b')[-1]
    else:
        ms, bs = [], []
        # get average 'm' and 'b' values, weighted by the length of the line
        for x1, y1, x2, y2 in lines:
            m, b = get_avg_mb(x1, y1, x2, y2)
            ms += m
            bs += b

        # take the avg of the middle half
        ms = sorted(ms)
        slice = int(len(ms) / 4)
        m = sum(ms[slice:3 * slice]) / (2 * slice)
        # m = ms[int(slice*2)]
        # update history
        history.get(side).get('m').append(m)

        bs = sorted(bs)
        slice = int(len(bs) / 4)
        b = sum(bs[slice:3 * slice]) / (2 * slice)
        # b = bs[int(slice*2)]
        # update history
        history.get(side).get('b').append(b)

        # n = 2
        # m_list = history.get('left').get('m')[-n:]
        #
        # # new m is weighted average of previous n ms - this doesn't really make much of a difference
        # # was trying to make the line a bit smoother but this doesn't work too well
        # m = np.average(m_list, weights=range(1, len(m_list) + 1) )

    start, end = get_line_coords(img, m, b)

    cv2.line(img, start, end, color=[255, 0, 0], thickness=10)

    return m, b

def get_line_coords(img, m, b):

    # get x start position by setting y to length of frame
    x_start = int((b - img.shape[0]) / -m)
    y_start = img.shape[0]

    y_segment = img.shape[0] / 24.

    # get end co-ordinates by just extrapolating to the same length of the mask region
    x_end = int((b - y_segment * 15) / -m)
    y_end = int(y_segment * 15)

    return (x_start, y_start), (x_end, y_end)

def get_avg_mb(x1, y1, x2, y2):

    # get line length
    x_len = abs(x2 - x1)
    y_len = abs(y2 - y1)
    line_length = math.sqrt(x_len ** 2 + y_len ** 2)

    # fit with np.polyfit
    m, b = np.polyfit(np.asarray([x1, x2]), np.asarray([y1, y2]), 1)

    return [m] * int(line_length), [b] * int(line_length),

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, using=draw_lines_v1, **kwargs):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    using(line_img, lines, **kwargs)
    return line_img

# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def colour2white(img):

    img = img.copy()

    x_segment = img.shape[1] / 24.
    y_segment = img.shape[0] / 24.

    # only evaluate patch of image due to processing times..
    for y in range( int(y_segment * 18), int(y_segment * 22) ):
        for x in range( int(x_segment * 5), int(x_segment * 10) ):

            rgb_min = min(img[y][x])
            rgb_max = max(img[y][x])

            rgb_range = int(rgb_max) - int(rgb_min)

            # if values are greater than 60 points of each other then convert to white
            if rgb_range > 60:
                img[y][x] = [255, 255, 255]

    return img
