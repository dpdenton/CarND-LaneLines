import os
import numpy as np
import matplotlib.pyplot as plt


class ImagePipeline(object):

    PIPELINE_DIR = 'pipeline_img'
    SEGMENT = 24

    def __init__(self, show_img=False, save_img=False):

        self.show_img = show_img
        self.save_img = save_img
        self.vertices = np.array([])
        self._pipes = []

        self.reset_history()

    def reset(self):

        self._pipes = []

        return self

    def reset_history(self):
        self.history = {
            'intersection': [(0,0)],
            'left': {
                'm': [-0.7],
                'b': [0],
            },
            'right': {
                'm': [0.7],
                'b': [-100],
            }}

    def add_pipe(self, fn, *args, **kwargs):

        self._pipes.append((fn, args, kwargs))

        return self

    def replace_pipe(self, fn, *args, **kwargs):
        for idx, pipe in enumerate(self._pipes):
            if pipe[0] == fn:
                self._pipes[idx] = (fn, args, kwargs)

        return self

    def flow(self):

        for pipe in self._pipes:

            fn, args, kwargs = pipe
            kwargs['img'] = self.image
            self.image = fn(*args, **kwargs)

            if self.show_img:
                print("Calling {}() with arg {}, and kwargs {}".format(fn.__name__, args, kwargs))
                plt.imshow(self.image, cmap='Greys_r')

            if self.save_img:
                path = self.PIPELINE_DIR.strip('/') + "/{}/{}.jpg".format(self.image_name, fn.__name__)
                directory = os.path.dirname(path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.imsave(path, arr=self.image, cmap='Greys_r')

        return self

    def set_image(self, image, image_name=None):

        self.image_name = image_name if image_name is not None else 'pipeline_img'
        self.image = image
        self.original_image = image

        return self

    def set_vertices(self, use_intersection=False):

        # break up image into 24 x 24 segments
        x_segment = self.original_image.shape[1] / 24.
        y_segment = self.original_image.shape[0] / 24.

        if use_intersection:
            coord_1 = self.history.get('intersection')[-1]
            coord_2 = self.history.get('intersection')[-1]
        else:
            # if vertices have already been set and not using 'dynamic' vertices then nothing further required
            if self.vertices.size:
                return self
            coord_1 = (10 * x_segment, y_segment*15)
            coord_2 = (14 * x_segment, y_segment*15)

        self.vertices = np.array([
            [
                # apears to be (x, y) points in the 'order' of bottom left, top left, top right, bottom right
                (0 * x_segment, 24 * y_segment),
                coord_1,
                coord_2,
                (24 * x_segment, 24 * y_segment)
            ]], dtype=np.int32
        )

        return self

# example usage
from helpers import *

pipeline = ImagePipeline(show_img=False, save_img=False)

def process_image(image):

    # set pipeline
    pipeline.reset()
    pipeline.set_image(image)
    pipeline.set_vertices(use_intersection=False)

    # add pipes
    pipeline \
        .add_pipe(colour2white) \
        .add_pipe(grayscale) \
        .add_pipe(gaussian_blur, kernel_size=5) \
        .add_pipe(canny, low_threshold=50, high_threshold=150) \
        .add_pipe(region_of_interest, vertices=pipeline.vertices) \
        .add_pipe(hough_lines, rho=1, theta=np.pi/180, threshold=15,
                  min_line_len=20, max_line_gap=40, history=pipeline.history, using=draw_lines_v3) \
        .add_pipe(weighted_img,  initial_img=pipeline.original_image, α=0.8, β=1., λ=0.) \

    # start flow
    pipeline.flow()

    return pipeline.image