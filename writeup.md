#**Finding Lane Lines on the Road: Reflection**

###1. Description

My pipeline consisted of 7 steps.

Step 1: Apply colour2white function. This detects 'non-greyscale' colours on the frame and converts them to white.
The theory being that any 'non-greyscale' colour has been intentionally placed on the road so should be considered a
lane marking of some sort.

Step 2: Apply standard grayscale function.
Step 3: Apply Gaussian blur with a kernel_size=5 to suppress noise and spurious gradients
Step 4: Apply canny edge detector with low_threshold=50, high_threshold=150 to convert image to a selection of lines.
Step 5: Apply a region of interest with 4 point polygon encompassing the shape of the road
Step 6: Detect applicable lines using the HoughLinesP method, with values rho=1, theta=np.pi/180, threshold=15, min_line_len=20, max_line_gap=40
Step 7: Overlay the detected lines onto the original image, hopefully over the lane markings on the road.

In order to draw a single line on the left and right lanes, I modified the draw_lines_v1() (renamed to draw_lines_v2)
function by calculating the gradient of each line, along with the position of the line on the page, and assigned the
line as belonging to the left-hand lane if it has a gradient of less than 0 and it's on the left hand side of the frame;
 and to the right-hand lane if it has a gradient of greater than 0 and sits on the right side of the frame.

Once the lines have been assigned to the left or right lane, I plotted a line of best fit through the end points of
each line to create a single line for each lane, increased the thickness and reduced the opacity to easily see the
lane markings when overlaying the lines onto the original image.

###2. Shortcomings


I initially reduced the the min_line_length to capture cats eyes and small segments of line at the bottom of the mask.
This works well when you don't have too much noise on the track, however when you introduce things like shadows this
approach doesn't work.

Additionally draw_lines_v2 function has no way of discriminating other line markings on the road, such as chevrons,
direction arrows, text etc.. which would drastically screw the line of best fit.

It also struggles to capture lines in bright sunshine against a 'greyish' surface where the contrast is minimal and
grayscaling has a negative effect, however decreasing the threshold for line detection increases the number of 'noisy'
lines included, further skewing the line of best fit.

Other cars on the road. Cars changing lanes covering the markings.

###4. Improvements

I modified  draw_lines_v2 further (renamed to  draw_lines_v3) and included a 'history' component.

It's safe to assume the the line in previous frame would have a similar gradient to the line in the current frame,
so I used the previous gradient to filter out any lines that lay outside of 10% of the previous gradient. A similar
approach could also have be used for the 'b' in y = mx + b.

For each line, I then created a list of gradients, where each gradient was multiplied by the length of the line to
reduce the chance of outliers skewing the final calculation.

The final calculation for 'm' was the average of the middle 2 quarters of the gradient list.

The same approach was used for calculating 'b'.

Lines were then plotted using these values for m and b and overlaid onto the original image.

If no lines were detected 'm' and 'b' were set to the previous values.

###5. Shortcoming of Improvements

Initial values for 'history' are hard-coded – a more sophisticated method for setting these values should be used,
along with some 'confidence' factor in the detection of lines, as the pipeline can fall apart if it starts filtering
out 'good' lines because the historical values are bad.

It still really struggled to pickup on the white lines on the right hand side when in bright sunshine against a
greyish surface (~ frames 100-110) so it fell back onto the historical calculations to render the line. This is a bit
of a hack and not a robust method, so I would invest more time pre-processing the image to sharpen up the contrast on
frames like this.

The change in m and b values can be quite severe causing a jerky overlay on challenge.mp4. I tried smoothing this out
by calculating a weighted moving average for m and b but this didn't have much effect.