from termcolor import colored
import tensorflow as tf
import os
import sys
import platform

machine_type = platform.uname()[0]
if machine_type == 'Windows':
    path_sep = '\win'
else:
    path_sep = '/'
path_sep = path_sep[0]
MASK_path = os.getcwd() + path_sep + 'utils' + path_sep + 'mask' + path_sep + 'mask_binary_mod.bmp'


def refactor_path(absolute_path, dataset_path, path_sep):
    relative_path = []
    for path in absolute_path:
        refactored_path = dataset_path+'images'+path_sep+path.split('/')[7]+path_sep+path.split('/')[8]+path_sep+path.split('/')[9]
        relative_path.append(refactored_path)
    return relative_path


def parse_function(filename, weight, speed, log_length):
    # Get the image as raw bytes.
    image_name = tf.read_file(filename)
    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.image.decode_bmp(image_name, channels=3)

    # The type is now uint8 but we need it to be float.
    image = tf.to_float(image)
    # image = tf.cast(image, dtype="float32")

    return image, weight, speed, log_length


def data_normalization(image, weight, speed, log_length):
    mean = [66.00343274112447, 97.59649670930388, 23.549425435852218]
    std_dev = [45.10411941567958, 47.92432784804989, 23.598328247056116]
    mean = tf.reshape(mean, [1, 1, 3])
    std_dev = tf.reshape(std_dev, [1, 1, 3])
    img_m = image - mean
    image = img_m / std_dev

    return image, weight, speed, log_length


# normalization should be applied to the entire dataset


def data_resize(image, weight, speed, log_length):
    resized_image = tf.image.resize_images(image, size=(96, 144))

    return resized_image, weight, speed, log_length


def data_masking(image, weight, speed, log_length):
    mask_name = tf.read_file(MASK_path)
    mask = tf.image.decode_bmp(mask_name, channels=3)
    mask = tf.to_float(mask)
    mask = tf.image.resize_images(mask, size=(96, 144))
    masked_img = tf.multiply(image, mask)

    return masked_img, weight, speed, log_length


def compute_loss(prediction, label, train_log_length_, operation='L2'):
    if operation == 'L2':
        return tf.divide(tf.squared_difference(prediction, label), train_log_length_)
    elif operation == 'Subtraction':
        return tf.divide(tf.subtract(prediction, label), train_log_length_)
    elif operation == 'L1':
        return tf.divide(tf.abs(tf.subtract(prediction, label)), train_log_length_)
    else:
        raise ValueError('Please specify loss function (L2, L1, Subtraction)')


def print_progress(count, total, cnt, overall, time_, count_log, loss, loss_):
    percent_complete = float(count) / total
    overall_complete = float(cnt) / (overall - 1)

    sec = time_ % 60
    mint = int(time_ / 60) % 60
    hr = int(time_ / 3600) % 60
    loss = str(loss)
    loss_ = str(loss_)
    msg = "\r Time_lapsed (hr:mm:ss) --> {0:02d}:{1:02d}:{2:02d} ,   loss: {3:s}   Log Progress: {4:.1%},     Overall Progress:{5:.1%}," \
          " completed {6:d} out of 185 logs <--> Initial loss: {7:s} ".format(hr, mint, sec, loss, percent_complete, overall_complete, count_log, loss_)
    sys.stdout.write(msg)
    sys.stdout.flush()


def validation_progress(log_cnt, num_logs, time_, loss, accuracy_loc):
    log_cnt += 1
    overall_complete = float(log_cnt) / num_logs
    sec = int(time_) % 60
    mint = int(time_ / 60) % 60
    hr = int(time_ / 3600) % 60
    loss = str(loss)
    msg = "\r Validation_Time (hr:mm:ss) --> {0:02d}:{1:02d}:{2:02d} ,   Avg_loss: {3:s}   Avg_accuracy: {4:.2%}   Overall Progress:{5:.1%}," \
          " completed {6:d} out of {7:d} logs".format(hr, mint, sec, loss, accuracy_loc, overall_complete, log_cnt, num_logs)
    sys.stdout.write(colored(msg, 'green'))
    sys.stdout.flush()
