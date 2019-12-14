import numpy as np
from scipy.ndimage.filters import convolve
from scipy import signal
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from imageio import imread
import os

MIN_RESOLUTION = 16

ROWS = 0

REPRESENTATION_ERROR = "Representation code not exist. please use 1 or 2"

RGB = 2

FILE_PROBLEM = "File Problem"

GREYSCALE = 1

MAX_INTENSITY = 255


def realpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def representation_check(representation):
    """
    check if representation code is valid
    :param representation: representation code
    """
    if representation not in [GREYSCALE, RGB]:
        print(REPRESENTATION_ERROR)
        exit()


def normalize_0_to_1(im):
    """
    normalize picture
    :param im: image in range 0-255
    :return: image in range [0,1]
    """
    if im.dtype != np.float64:
        im = im.astype(np.float64)
        im /= MAX_INTENSITY
    return im


def read_image(filename, representation):
    """
    This function returns an image, make sure the output image is represented by a matrix of type
    np.float64 with intensities (either grayscale or RGB channel intensities) normalized to the range [0, 1].
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    :return: an image
    """
    im = None
    try:
        im = imread(filename)
    except Exception:  # internet didnt have specific documentation regarding the exceptions this func throws
        print(FILE_PROBLEM)
        exit()
    representation_check(representation)
    im = normalize_0_to_1(im)
    if representation == GREYSCALE:
        return rgb2gray(im)
    return im.astype(np.float64)


def reduce(im, filter):
    im = convolve(im, filter)
    im = convolve(im, filter.T)
    return im[:im.shape[0]:2, :im.shape[1]:2]


def expand(im, filter):
    new_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    new_im[:new_im.shape[0]:2, :new_im.shape[1]:2] = im
    new_im = convolve(new_im, 2 * filter)
    new_im = convolve(new_im, 2 * filter.T)
    return new_im


def create_filter(size):
    base_filter = np.array([1, 1])
    filter = np.array([1, 1])
    for i in range(size - 2):
        filter = signal.convolve(filter, base_filter)
    return filter / np.sum(filter)


def build_gaussian_pyramid(im, max_levels, filter_size):
    filter = create_filter(filter_size).reshape(1, filter_size)
    gaussian_pyr = [im]
    for i in range(max_levels - 1):
        curr_level = reduce(gaussian_pyr[i], filter)
        if curr_level.shape[0] < MIN_RESOLUTION or curr_level.shape[1] < MIN_RESOLUTION:
            break
        gaussian_pyr.append(curr_level)
    return gaussian_pyr, filter


def build_laplacian_pyramid(im, max_levels, filter_size):
    gaussian_pyr, filter = build_gaussian_pyramid(im, max_levels, filter_size)
    laplacian_pyr = []
    for i in range(len(gaussian_pyr) - 1):
        laplacian_pyr.append(gaussian_pyr[i] - expand(gaussian_pyr[i + 1], filter))
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr, filter


def stretch(im):
    max_val = np.max(im)
    im -= np.min(im)
    return im / max_val


def laplacian_to_image(lpyr, filter_vec, coeff):
    lpyr *= coeff
    for i in range(1, len(lpyr)):
        lpyr[-i - 1] += expand(lpyr[-i], filter_vec)
    return lpyr[0]


def render_pyramid(pyr, levels):
    res = np.zeros((pyr[0].shape[0], int(pyr[0].shape[1] * (2 - 0.5 ** (levels - 1)))))
    last_col = 0
    for i in range(levels):
        m, n = pyr[i].shape
        res[:m, last_col:last_col + n] = stretch(pyr[i])
        last_col += n
    return res


def display_pyramid(pyr, levels):
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    L_1, filter_vec_im = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L_2, _ = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    G_m, filter_vec_mask = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    L_out = []
    for i in range(len(L_1)):
        L_out[i] = G_m[i] * L_1[i] + (1 - G_m[i]) * L_2[i]
    coeff = [1] * len(L_out)
    return laplacian_to_image(L_out, filter_vec, coeff).clip(min=0, max=1)


def blending_example(im1, im2, mask, ):
    im1 = read_image(realpath(im1), 2)
    im2 = read_image(realpath(im2), 2)
    mask = read_image(realpath(mask), 1).astype(np.bool)
    blended = np.empty(im1.shape+(3,))
    for i in range(blended.shape[2]):
        blended[:,:,i] = pyramid_blending(im1,im2,mask,,,)



if __name__ == '__main__':
    pyr = build_laplacian_pyramid(read_image('monkey.jpg', 1), 5, 3)[0]
    display_pyramid(pyr, 4)
