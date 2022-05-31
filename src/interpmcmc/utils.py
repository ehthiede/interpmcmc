import numpy as np


def min_image(distance_vec, dmn_width):
    return distance_vec - np.round_(distance_vec / dmn_width) * dmn_width
