from termcolor import colored
import tensorflow as tf
import os
import sys
import platform

machine_type = platform.uname()[0]
if machine_type == 'Linux':
    path_sep = '/'
else:
    path_sep = '\win'
path_sep = path_sep[0]
MASK_path = os.getcwd() + path_sep + 'utils' + path_sep + 'mask' + path_sep + 'mask_binary_mod.bmp'


def refactor_path(absolute_path, dataset_path, path_sep):
    relative_path = []
    for path in absolute_path:
        refactored_path = dataset_path+'images'+path_sep+path.split('/')[7]+path_sep+path.split('/')[8]+path_sep+path.split('/')[9]
        relative_path.append(refactored_path)
    return relative_path


def parse_function(filename):
    # Get the image as raw bytes.
    image_name = tf.read_file(filename)
    # Decode the raw bytes so it becomes a tensor with type.
    splitter = filename.split('.')
    type_file = splitter[len(splitter)-1]
    if type_file == 'bmp':
        image = tf.image.decode_bmp(image_name, channels=3)
    elif type_file == 'jpg':
        image = tf.image.decode_jpeg(image_name, channels=3)
    elif type_file == 'png':
        image = tf.image.decode_png(image_name, channels=3)
    else:
        TypeError:"input_image is not supported, must be jpg, png, or bmp"
    # The type is now uint8 but we need it to be float.
    image = tf.to_float(image)
    # image = tf.cast(image, dtype="float32")

    return image


def data_normalization(image):
    mean = [66.00343274112447, 97.59649670930388, 23.549425435852218]
    std_dev = [45.10411941567958, 47.92432784804989, 23.598328247056116]
    mean = tf.reshape(mean, [1, 1, 3])
    std_dev = tf.reshape(std_dev, [1, 1, 3])
    img_m = image - mean
    image = img_m / std_dev

    return image


# normalization should be applied to the entire dataset


def data_resize(image):
    resized_image = tf.image.resize_images(image, size=(96, 144))

    return resized_image


def data_masking(image):
    mask_name = tf.read_file(MASK_path)
    mask = tf.image.decode_bmp(mask_name, channels=3)
    mask = tf.to_float(mask)
    mask = tf.image.resize_images(mask, size=(96, 144))
    masked_img = tf.multiply(image, mask)

    return masked_img
