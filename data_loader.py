"""data loader for nn training and some data augmentations functions"""
import gzip
import pickle
import time

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
from scipy.signal import convolve2d
from skimage import filters
from skimage.measure import regionprops
from skimage.transform import resize
from sklearn.model_selection import train_test_split


def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (28, 28)) for x in tr_d[0]]
    validation_inputs = [np.reshape(x, (28, 28)) for x in va_d[0]]
    test_inputs = [np.reshape(x, (28, 28)) for x in te_d[0]]
    start = time.time()
    tr_data = augment_data(list(zip(training_inputs, tr_d[1])))
    print("done tr")
    va_data = augment_data(list(zip(validation_inputs, va_d[1])))
    print("done va")
    te_data = augment_data(list(zip(test_inputs, te_d[1])))
    print("done te")
    end = time.time()
    print("time taken: ", end - start)
    training_data = [(np.reshape(x, (784, 1)), vectorized_result(y)) for (x, y) in tr_data]
    validation_data = [(np.reshape(x, (784, 1)), y) for (x, y) in va_data]
    test_data = [(np.reshape(x, (784, 1)), y) for (x, y) in te_data]
    return training_data, list(validation_data), test_data


def load_printed_data():
    X, Y = load_printed_data_clean()
    (X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.30, random_state=42)
    tr_data = augment_data2(list(zip(X_train, Y_train)))
    te_data = augment_data2(list(zip(X_test, Y_test)))
    training_data = [(np.reshape(x, (784, 1)), vectorized_result(y)) for (x, y) in tr_data]
    test_data = [(np.reshape(x, (784, 1)), vectorized_result(y)) for (x, y) in te_data]
    return training_data, test_data


def load_printed_data_clean():
    data = pd.read_csv('image_data.csv')
    data = data.loc[data['y'] != 10]
    X, Y = [], data['y']
    data = data.drop(columns=['y'])
    for i in range(data.shape[0]):
        flatten_image = data.iloc[i].values[1:]
        image = np.reshape(flatten_image, (28, 28))
        X.append(image)
    X, Y = np.array(X), np.array(Y)
    return X, Y


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[int(j)] = 1.0
    return e


def blur_img(image, percentage):
    filter1 = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]])
    filter1 = filter1 / filter1.max()
    filter2 = np.zeros((3, 3))
    filter2[1, 1] = 1
    filter1 = filter1 * percentage + filter2 * (1 - percentage)
    b = convolve2d(image, filter1, mode='same', boundary='symm')
    if np.any(b):
        b = b / np.max(b)

    return b


def shift_image(image, vertical_change, horizontal_change):
    shifted_image = shift(image, [vertical_change, horizontal_change], cval=0, mode="constant")
    if np.any(shifted_image < 0):
        shifted_image[shifted_image < 0] = 0
    if np.any(shifted_image != 0):
        shifted_image = shifted_image / np.max(shifted_image)
    return shifted_image


def add_gaussian_noise(image):
    gauss = np.random.normal(0, 0.0001 ** 0.5, image.shape)
    gauss = gauss.reshape(image.shape)
    noisy_im = image + gauss
    return noisy_im


def add_salt_pepper_noise(image):
    s_vs_p = 0.25
    amount = 0.06
    out = np.copy(image)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    salt_indices = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(salt_indices)] = 0.8
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    pepper_indices = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(pepper_indices)] = 0.2
    return out.reshape((28, 28))


def add_poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_im = np.random.poisson(image * vals) / float(vals)
    return noisy_im


def add_speckle_noise(image):
    row, col = image.shape
    gauss = np.random.randn(row, col) * 0.3
    gauss = gauss.reshape(row, col)
    noisy = image + image * gauss
    return noisy


def augment_data(data):
    augmented_data = [(x, y) for (x, y) in data]
    for (x, y) in data:
        augmented_data.append((add_salt_pepper_noise(x), y))
        augmented_data.append((add_gaussian_noise(x), y))
        augmented_data.append((blur_img(x, 0.7), y))
        augmented_data.append((add_poisson_noise(x), y))
        augmented_data.append((add_speckle_noise(x), y))
        for direction in [(3, 0), (0, -3), (-3, 3), (0, 4)]:
            augmented_data.append((shift_image(x, direction[0], direction[1]), y))
    return np.array(augmented_data)


def zoom_out(image, size):
    if size > 1:
        s = int(max(image.shape) * size)
        digit1 = np.full((s, s), 0, dtype=image.dtype)
        x = (s - image.shape[0]) // 2
        y = (s - image.shape[1]) // 2
        digit1[x:x + image.shape[0], y:y + image.shape[1]] = image
        return resize(digit1, (28, 28))
    else:
        s = int(max(image.shape) * size)
        x = (image.shape[0] - s) // 2
        y = (image.shape[1] - s) // 2
        digit1 = image[x:x + s, y:y + s]
        return resize(digit1, (28, 28))


def stretch_x(image, size):
    if size < 0:
        image = image[::-1]
        size = -size
    if size < 1:
        return stretch_y(image, 1 / size)
    s = int(max(image.shape) * size)
    digit1 = np.full((s, 28), 0, dtype=image.dtype)
    x = (s - image.shape[0]) // 2
    digit1[x:x + image.shape[0], :] = image
    return resize(digit1, (28, 28))


def stretch_y(image, size):
    return stretch_x(image.T, size).T


def add_surrounding_noise(im):
    threshold_value = filters.threshold_otsu(im)
    labeled_foreground = (im > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, im)
    center_of_mass = properties[0].centroid
    noisy = np.copy(im)
    for i in range(max(0, int(center_of_mass[0]) - 8), min(27, int(center_of_mass[0]) + 9)):
        for j in range(max(0, int(center_of_mass[1]) - 8), min(27, int(center_of_mass[1]) + 9)):
            if im[i, j] < 0.1:
                noisy[i, j] = np.random.uniform(0.0, 0.05)
    return noisy


def erode(image):
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=3)
    eroded[eroded < 0] = 0
    eroded = eroded / eroded.max()
    return eroded


def dilate(image):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=3)
    dilated[dilated < 0] = 0
    dilated = dilated / dilated.max()
    return dilated


def augment_data2(data):
    augmented_data = []
    for (x, y) in data:
        augmented_data.append((stretch_x(x, 0.6), y))
        # augmented_data.append((stretch_x(x, 0.5), y))
    temp1 = []
    for (x, y) in augmented_data:
        # temp1.append((zoom_out(x, 1.45), y))
        temp1.append((zoom_out(x, 1.7), y))
        temp1.append((zoom_out(x, 2), y))
    augmented_data += temp1
    temp2 = []
    for (x, y) in augmented_data:
        for direction in [(5, 0), (-5, 0), (0, 5), (0, -5), (5, 5), (5, -5), (-5, 5), (-5, -5),
                          (2, 3), (2, -3), (-2, 3), (-2, -3), (3, 2), (3, -2), (-3, 2), (-3, -2),
                          (3, 0), (-3, 0), (0, 3), (0, -3), (3, 3), (3, -3), (-3, 3), (-3, -3)]:
            temp2.append((shift_image(x, direction[0], direction[1]), y))
    augmented_data += temp2
    temp3 = []
    for (x, y) in augmented_data:
        temp3.append((add_salt_pepper_noise(x), y))
        # temp3.append((add_poisson_noise(x), y))
        temp3.append((add_surrounding_noise(x), y))
        x_temp = add_salt_pepper_noise(x)
        temp3.append((add_gaussian_noise(x_temp), y))
        x_temp2 = add_salt_pepper_noise(x)
        temp3.append((add_surrounding_noise(x_temp2), y))
    augmented_data += temp3
    return np.array(augmented_data)


def augment_data3(data, multiple=2, noise_chance=0.2, shift=7):
    augmented_data = [(x, y) for (x, y) in data]
    for (x, y) in data:
        for i in range(multiple):
            x1 = x.reshape((28, 28))
            x1 = augment(x1, noise_chance, shift)
            augmented_data.append((x1, y))
            # imdisplay(x1,5)

    return augmented_data


def augment(x1, noise_chance, shift):
    if np.random.random() < noise_chance:
        x1 = add_salt_pepper_noise(x1)
    if np.random.random() < noise_chance:
        x1 = add_gaussian_noise(x1)
    if np.random.random() < noise_chance:
        x1 = blur_img(x1, np.random.random())
    # if np.random.random() < noise_chance:
    #     x1 = add_poisson_noise(x1)
    if np.random.random() < noise_chance:
        x1 = add_speckle_noise(x1)
    # if np.random.random() < noise_chance:
    #     x1 = zoomOut(x1, random.uniform(0.8,1.5))
    # x1 = zoomOut(x1, random.uniform(0.8,1))
    d = np.random.randint(-shift, shift + 1, 2, np.int)
    x1 = shift_image(x1, *d)
    return x1


def augment_for_predict(image, variation_mul=2):
    variations = [image.reshape((784, 1))]
    for zoom in [0.8, 0.9, 1.1, 1.2]:
        variations.append(zoom_out(image, zoom).reshape((784, 1)))
    temp = []
    for x in variations:
        for direction in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            temp.append(shift_image(x.reshape((28, 28)), direction[0], direction[1]).reshape((784, 1)))
        for blur_val in [0.6, 0.7, 0.8]:
            temp.append(blur_img(x.reshape((28, 28)), blur_val).reshape((784, 1)))
    variations += temp
    temp = []
    for x in variations:
        for i in range(variation_mul):
            temp.append(augment(x.reshape((28, 28)), 0.5, 3).reshape(784, 1))
    variations += temp
    return variations
