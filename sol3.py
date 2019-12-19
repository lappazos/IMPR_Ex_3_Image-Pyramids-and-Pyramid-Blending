import numpy as np
from scipy.ndimage.filters import convolve
from scipy import signal
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from imageio import imread
import os

EVEN_INDEX = 2

MIN_RESOLUTION = 16

ROWS = 0

REPRESENTATION_ERROR = "Representation code not exist. please use 1 or 2"

RGB = 2

FILE_PROBLEM = "File Problem"

GREYSCALE = 1

MAX_INTENSITY = 255


def realpath(filename):
    """
    :param filename
    :return: full path of file name
    """
    return os.path.join(os.path.dirname(__file__), filename)


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
    im = normalize_0_to_1(im)
    if representation == GREYSCALE:
        return rgb2gray(im)
    return im.astype(np.float64)


def reduce(im, filter):
    """
    Blur & sub-sample
    :param im: im to reduce
    :param filter: filter to blur with using convolution
    :return: reduced im
    """
    im = convolve(im, filter)
    im = convolve(im, filter.T)
    return im[:im.shape[0]:EVEN_INDEX, :im.shape[1]:EVEN_INDEX]


def expand(im, filter):
    """
    pad with zeroes & blur
    :param im: im to expand
    :param filter: filter to blur with using convolution
    :return: expanded im
    """
    new_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    new_im[:new_im.shape[0]:EVEN_INDEX, :new_im.shape[1]:EVEN_INDEX] = im
    new_im = convolve(new_im, 2 * filter)
    new_im = convolve(new_im, 2 * filter.T)
    return new_im


def create_filter(size):
    """
    give gaussian filter of given size
    :param size: filter size
    :return: row vector
    """
    if size == 1:
        return np.array([1]).reshape((1, 1))
    base_filter = np.array([1, 1])
    filter = np.array([1, 1])
    for i in range(size - 2):
        filter = signal.convolve(filter, base_filter)
    return (filter / np.sum(filter)).reshape((1, size))


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    construct a Gaussian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size:  the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
            in constructing the pyramid filter
    :return: pyr: a standard python array with maximum length of max_levels, where each element
            of the array is a grayscale image,
            filter_vec: row vector of shape (1, filter_size) used for the pyramid construction
    """
    filter = create_filter(filter_size).reshape(1, filter_size)
    gaussian_pyr = [im]
    for i in range(max_levels - 1):
        curr_level = reduce(gaussian_pyr[i], filter)
        if curr_level.shape[0] < MIN_RESOLUTION or curr_level.shape[1] < MIN_RESOLUTION:
            break
        gaussian_pyr.append(curr_level)
    return gaussian_pyr, filter


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    construct a Laplacian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size:  the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
            in constructing the pyramid filter
    :return: pyr: a standard python array with maximum length of max_levels, where each element
            of the array is a grayscale image,
            filter_vec: row vector of shape (1, filter_size) used for the pyramid construction
    """
    gaussian_pyr, filter = build_gaussian_pyramid(im, max_levels, filter_size)
    laplacian_pyr = []
    for i in range(len(gaussian_pyr) - 1):
        laplacian_pyr.append(gaussian_pyr[i] - expand(gaussian_pyr[i + 1], filter))
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr, filter


def stretch(im):
    """
    stretch the values of each pyramid level to [0, 1]
    :param im: im to stretch
    :return: stretched image
    """
    max_val = np.max(im)
    minimum = np.min(im)
    im -= minimum
    return im / (max_val - minimum)


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    e reconstruction of an image from its Laplacian Pyramid
    :param lpyr: a standard python array, where each element of the array is a grayscale image
    :param filter_vec: row vector of shape (1, filter_size)
    :param coeff: python list
    :return: reconstructed image
    """
    for i in range(len(lpyr)):
        lpyr[i] = lpyr[i] * coeff[i]
    for i in range(1, len(lpyr)):
        lpyr[-i - 1] += expand(lpyr[-i], filter_vec)
    return lpyr[0]


def render_pyramid(pyr, levels):
    """
    create rendered_pyramid image
    :param pyr: either a Gaussian or Laplacian pyramid
    :param levels: number of levels to present in the result
    :return: single black image in which the pyramid levels of the given pyramid pyr are stacked horizontally
    """
    res = np.zeros((pyr[0].shape[0], int(pyr[0].shape[1] * (2 - 0.5 ** (levels - 1)))))
    last_col = 0
    for i in range(levels):
        m, n = pyr[i].shape
        f = stretch(pyr[i])
        res[:m, last_col:last_col + n] = f
        last_col += n
    return res


def display_pyramid(pyr, levels):
    """
    display the stacked pyramid image
    :param pyr: either a Gaussian or Laplacian pyramid
    :param levels: number of levels to present in the result
    :return:
    """
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
     pyramid blending
    :param im1: input grayscale image to be blended
    :param im2: input grayscale image to be blended
    :param mask: a boolean mask containing True and False representing which parts
            of im1 and im2 should appear in the resulting im_blend. Note that a value of True corresponds to 1,
            and False corresponds to 0.
    :param max_levels: is the max_levels parameter when generating the Gaussian and Laplacian pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd scalar that represents a squared filter) which
            defining the filter used in the construction of the Laplacian pyramids of im1 and im2
    :param filter_size_mask: is the size of the Gaussian filter(an odd scalar that represents a squared filter) which
            defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: grayscale image
    """
    L_1, filter_vec_im = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L_2, _ = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    G_m, filter_vec_mask = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    L_out = []
    for i in range(len(L_1)):
        L_out.append(G_m[i] * L_1[i] + (1 - G_m[i]) * L_2[i])
    coeff = [1] * len(L_out)
    return laplacian_to_image(L_out, filter_vec_im, coeff).clip(min=0, max=1)


def blending_example(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
     pyramid blending and display
    :param im1: input RGB image to be blended
    :param im2: input RGB image to be blended
    :param mask: a boolean mask containing True and False representing which parts
            of im1 and im2 should appear in the resulting im_blend. Note that a value of True corresponds to 1,
            and False corresponds to 0.
    :param max_levels: is the max_levels parameter when generating the Gaussian and Laplacian pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd scalar that represents a squared filter) which
            defining the filter used in the construction of the Laplacian pyramids of im1 and im2
    :param filter_size_mask: is the size of the Gaussian filter(an odd scalar that represents a squared filter) which
            defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: RGB image
    """
    im1 = read_image(realpath(im1), 2)
    im2 = read_image(realpath(im2), 2)
    mask = read_image(realpath(mask), 1).astype(np.int).astype(np.bool)
    blended = np.empty(im1.shape)
    for i in range(blended.shape[2]):
        blended[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask, max_levels, filter_size_im,
                                            filter_size_mask)
    plt.figure()
    _, quadrants = plt.subplots(2, 2)
    quadrants[0, 0].imshow(im1, cmap=plt.get_cmap("gray"), interpolation='nearest')
    quadrants[0, 1].imshow(im2, cmap=plt.get_cmap("gray"), interpolation='nearest')
    quadrants[1, 0].imshow(mask, cmap=plt.get_cmap("gray"), interpolation='nearest')
    quadrants[1, 1].imshow(blended, cmap=plt.get_cmap("gray"), interpolation='nearest')
    plt.show()
    return im1, im2, mask, blended


def blending_example1():
    """
    dispalt example 1
    :return: im1, im2, mask, blended
    """
    return blending_example(os.path.join('externals', 'girl_im.jpg'), os.path.join('externals', 'fire.jpg'),
                            os.path.join('externals', 'eyes_mask.jpg'), 4, 3, 3)


def blending_example2():
    """
    display example 2
    :return: im1, im2, mask, blended
    """
    return blending_example(os.path.join('externals', 'abbey_road.jpg'), os.path.join('externals', 'piano.jpg'),
                            os.path.join('externals', 'legs_mask.jpg'), 3, 5, 3)
