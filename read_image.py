import math
import operator
import pickle
from typing import Union

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core._multiarray_umath import ndarray
from scipy.ndimage.measurements import label
from scipy.signal import convolve2d
from skimage.io import imsave
from skimage.transform import resize
from data_loader import augment_for_predict
from neural_net_model import vote_predict
import data_loader as dl


def find_lines(edges):
    lines = cv2.HoughLines(edges, 0.8, np.pi / 180, int(edges.shape[0] * 0.8))
    lines = lines[:, 0, :]

    if __level < 1:
        print('number of Hough lines:', len(lines))

    line_map = np.zeros(edges.shape)
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        if abs(math.degrees(theta) % 90) > 15:
            continue

        cv2.line(line_map, (x1, y1), (x2, y2), 1, 2)

    imdisplay(line_map, 2, title='Hough line_map')
    return line_map


def find_corners_of_largest_polygon(img):
    """Finds the 4 extreme corners of the largest contour in the image."""
    dest = img.copy()
    c = cv2.findContours(dest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, h = c[-2:]  # Find contours
    dest = cv2.cvtColor(dest, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(dest, contours, -1, (255, 0, 0))
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
    polygon = contours[0]  # Largest image
    cv2.drawContours(dest, [polygon], -1, (0, 255, 0), thickness=3)
    imdisplay(dest, 3, title="contours")

    # Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
    # Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

    # Bottom-right point has the largest (x + y) value
    # Top-left has point smallest (x + y) value
    # Bottom-left point has smallest (x - y) value
    # Top-right point has largest (x - y) value
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    # Return an array of all 4 points using the indices
    # Each point is in its own array of one coordinate
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def get_edges_image(img, d=True):
    """Returns a map of edge pixels"""
    th3 = clean_image(img)

    edges = cv2.Canny(th3, 0, 1, apertureSize=3)
    imdisplay(edges, 0, title='canny1')

    if d:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        imdisplay(edges, 0, title='canny2')
    return edges


def clean_image(img, blur=7):
    """returns black and white image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imdisplay(gray, 1, title='gray')
    gray_blur = cv2.medianBlur(gray, blur)
    th3 = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 3)
    imdisplay(gray_blur, 1, title='gray_blur')
    imdisplay(th3, 1, title='th')
    return th3


def distance_between(p1, p2):
    """Returns the scalar distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def infer_grid(img, size=9):
    """Infers 81 cell grid from a square image."""
    squares = []
    side = img.shape[:1]
    side = side[0] / size

    # Note that we swap j and i here so the rectangles are stored in the list reading left-right instead of top-down.
    for j in range(size):
        for i in range(size):
            p1 = (i * side, j * side)  # Top left corner of a bounding box
            p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
            squares.append((p1, p2))
    return squares


def cut_from_rect(img, rect, pad):
    """Cuts a rectangle from an image using the top left and bottom right points."""
    return img[int(rect[0][1] + pad):int(rect[1][1] - pad), int(rect[0][0] + pad):int(rect[1][0] - pad)]


def crop_and_warp(img, crop_rect):
    """Crops and warps a rectangular section from an image into a square of similar size."""

    # Rectangle described by top left, top right, bottom right and bottom left points
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

    # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    # Get the longest side in the rectangle
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])

    # Describe a square with side of the calculated length, this is the new perspective we want to warp to
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    m = cv2.getPerspectiveTransform(src, dst)

    # Performs the transformation on the original image
    warp = cv2.warpPerspective(img, m, (int(side), int(side)))
    imdisplay(warp, 2, title='extracted and warped sudoku from the above image')

    return warp


def crop_and_warp_revrse(img_dest, crop_rect, part):
    """Crops and warps a rectangular section from an image into a square of similar size."""

    # Rectangle described by top left, top right, bottom right and bottom left points
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

    # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    # Get the longest side in the rectangle
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])

    # Describe a square with side of the calculated length, this is the new perspective we want to warp to
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    m = cv2.getPerspectiveTransform(dst, src)

    # Performs the transformation on the original image
    warp = cv2.warpPerspective(part, m, (img_dest.shape[1], img_dest.shape[0]))
    imdisplay(warp, 0, title='reversed')
    reversed_image = np.uint8(img_dest * (warp == 0)[:, :, 0, np.newaxis] + warp)
    cv2.polylines(reversed_image, np.int_([src]), True, (0, 255, 0), 2)

    imdisplay(reversed_image, 4, title='reversed2')

    return reversed_image


debug_stage = ''


def imdisplay(im, level, fit_color=False, title=None):
    """
    displays an image represented by numpy array
    :param title: the name of the plot and file
    :param fit_color: should the image be normalized
    :param im: a numpy array of the image
    """
    if level < __level:
        return
    if fit_color:
        im -= im.min()
        if im.max() != 0:
            im = np.divide(im, im.max(), dtype=np.float)
    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    # plt.ion()
    plt.show()
    # plt.pause(1)
    imsave(f"read_image/{debug_stage}{title}.jpeg", im)


def extract_digit(img, rect, cut_padding=2, margin=8):
    """Extracts a digit (if one exists) from a Sudoku square."""
    digit = cut_from_rect(img, rect, cut_padding)  # Get the digit box from the whole square

    digit = cut_main_shape(digit)

    if np.count_nonzero(digit) == 0:
        return np.zeros((28, 28))

    digit = digit - digit.min()
    digit = digit / digit.max()
    m = 28 - margin * 2
    digit = resize(digit * 255, (m, m))

    s = 28

    digit1 = np.full((s, s), 0)
    x = margin
    y = margin
    digit1[x:x + digit.shape[0], y:y + digit.shape[1]] = digit
    return digit1


def cut_main_shape(digit):
    """cuts around the biggest shape in the image"""
    rect1 = get_largest_shape(1 - digit)
    if rect1 is not None and np.sum(rect1) > 5:
        # rect1 = digit
        rect2 = (first_non_zero_2d(rect1, axis=1), first_non_zero_2d(rect1, axis=0)), \
                (last_non_zero_2d(rect1, axis=1), last_non_zero_2d(rect1, axis=0))
        digit1 = cut_from_rect(digit, rect2, -5)  # Get the digit box from the whole square
        if (digit1.shape[0] + digit1.shape[1]) > 30:
            digit = digit1
    return digit


def get_shape_lengths(labeled, ncomponents):
    lengths = []
    for i in range(1, ncomponents + 1):
        start = first_non_zero_2d(labeled == i, axis=1)
        end = last_non_zero_2d(labeled == i, axis=1)
        length = end - start
        lengths.append([i, length])
        if start == end:
            continue
        # print(f'i={i},start={start},end={end}')
        # imdisplay(labeled==i)
    return np.array(lengths)


def get_longest_line(labeled, lengths):
    if len(lengths) == 0:
        return None
    j = np.argmax(lengths.T[1])
    i = int(lengths.T[0][j])
    line1 = labeled == i
    return line1.astype(int)


def get_largest_shape(board):
    labeled, ncomponents = label(np.logical_not(board))
    imdisplay(labeled, 0, title="shapes")
    lengths = get_shape_lengths(labeled, ncomponents)

    return get_longest_line(labeled, lengths)


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def last_non_zero_2d(line1, axis):
    np_max = np.max(last_nonzero(line1, axis=axis, invalid_val=np.NINF))
    if np.isinf(np_max):
        return line1.shape[axis]

    return int(np_max)


def first_non_zero_2d(line1, axis):
    np_min: Union[ndarray, int, float, complex] = np.min(first_nonzero(line1, axis=axis, invalid_val=np.inf))
    if np.isinf(np_min):
        return 0
    return int(np_min)


def extract_puzzle(file_path='', puzzle_size=9):
    """cuts 81 digit images from a file"""
    img = cv2.imread(file_path)
    imdisplay(img, 2, title='origin')
    imgr = (resize(img, (480, 640),mode='reflect') * 255).astype(np.uint8)
    imdisplay(imgr, 2, title='origin_resized')

    edges = get_edges_image(imgr)

    corners = find_corners_of_largest_polygon(edges)
    global debug_stage
    debug_stage = 'cropped_'
    cropped = crop_and_warp(imgr, corners)
    cropped = cropped[2:-3, 2:-3]
    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    imdisplay(cropped_gray, 1, title='gray1')
    cropped_gray = cv2.adaptiveThreshold(cropped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 3)
    imdisplay(cropped_gray, 1, title='gray2')

    edges = get_edges_image(cropped, d=True)
    shape = find_lines(edges)
    if shape is not None:
        kernel = np.ones((3, 3), np.uint8)
        shape = cv2.dilate(shape, kernel, iterations=2) > 0
        imdisplay(shape, 1, title='shape')

        kernel = np.ones((3, 3), np.uint8)
        shape = convolve2d(shape, kernel, mode='same', boundary='symm')
        cropped_gray[shape > 0] = 255
    imdisplay(cropped_gray, 3, title='gray3')
    cropped_gray = cv2.medianBlur(cropped_gray, 3)
    cropped_gray = cropped_gray - cropped_gray.min()
    cropped_gray = cropped_gray / cropped_gray.max()
    cropped_gray = 1 - cropped_gray


    imdisplay(cropped_gray, 3, title='gray4')

    kernel = np.ones((3, 3), np.uint8)

    squares = infer_grid(cropped, puzzle_size)

    digits = np.array([extract_digit(cropped_gray, rec, cut_padding=6, margin=0) for rec in squares])
    plot_many_images(digits)
    return digits, (imgr, cropped, squares, corners)


def plot_many_images(images, titles='', rows=9, columns=9):
    """Plots each image in a given list as a grid structure. using Matplotlib."""
    if __level > 4:
        return
    plt.title('Extracted elements from sudoku')
    for i, image in enumerate(images):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image, 'gray')
        plt.xticks([]), plt.yticks([])  # Hide tick marks
    plt.axis('off')
    plt.savefig(f"./read_image/digits.jpeg")
    plt.show()


__level = 0


def read_digits(digits, nn, min_pixels=30,):
    model_file = open(nn, 'rb')
    nn = pickle.load(model_file)
    # reshape = digits.reshape(81, 784, 1)
    s = digits.shape[0]
    puzzle = np.zeros((s,), dtype=np.int8)
    for i,d1 in enumerate(digits):
        if d1.max() == 0 or np.count_nonzero(d1 > 0.1) < min_pixels:
            continue
        d1 = d1 / d1.max()

        augmented = augment_for_predict(dl.zoom_out(d1, 1), 2)

        d = np.array(vote_predict(nn, augmented))

        puzzle[i] = d

    return puzzle.reshape(int(s**0.5), -1)


def load_display():
    X=np.zeros((10,28,28),dtype=np.uint8)
    for i in range(1,10):
        im = cv2.imread(f"display/{i}.jpeg")
        X[i] =cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return X

def write_solution(img, cropped, squares, corners, board, solution):
    X = load_display()
    for i in range(9):
        for j in range(9):
            if board[i, j] > 0:
                continue

            s = solution[i, j]

            d1 = X[s]
            d1 = dl.zoom_out(d1, 2)
            rect = cut_from_rect(cropped, squares[i * 9 + j], 2)
            d1 = resize(d1, tuple(rect.shape[0:2]))
            z = np.zeros((*d1.shape, 3))
            d2 = np.stack([d1] * 3, axis=-1)
            z[:, :, 2] = d1
            d1 = z
            d1 = (d1 * 255).astype(np.uint8)
            rect[:] = rect * (1 - d2) + d2 * d1
    return crop_and_warp_revrse(img, corners, cropped)
