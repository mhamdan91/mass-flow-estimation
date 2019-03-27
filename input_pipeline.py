from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np          # dealing with arrays

tfe = tf.contrib.eager
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
layers = tf.keras.layers
tf.enable_eager_execution(config=config)
tf.executing_eagerly()
print(tf.__version__)

HOME_DIR = '/home/moe/PycharmProjects/step_step/'
MASK_path = '/home/moe/Desktop/bambo_Logs/mask/mask_binary_mod.bmp'
MAIN_dir = '/home/moe/Desktop/bambo_Logs/Data/simple_dataset/regression/sorts/'
data_files_dir = '/home/moe/PycharmProjects/step_step/Cleaned_code/data_files/'


def parse_function(filename, weight, speed, log_length):
    # Get the image as raw bytes.
    image_name = tf.read_file(filename)
    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.image.decode_bmp(image_name, channels=3)

    # The type is now uint8 but we need it to be float.
    image = tf.to_float(image)

    return image, weight, speed, log_length


def data_normalization(image, weight, speed, log_length):
    mean_std = np.load(data_files_dir + 'std_mean_60_dataset.npy')
    mean = [mean_std[0], mean_std[2], mean_std[4]]
    std_dev = [mean_std[1], mean_std[3], mean_std[5]]
    mean = tf.reshape(mean, [1, 1, 3])
    std_dev = tf.reshape(std_dev, [1, 1, 3])
    img_m = image - mean
    image = img_m / std_dev

    return image, weight, speed, log_length


# normalization should be applied to the entire dataset


def data_masking(image, weight, speed, log_length):
    mask_name = tf.read_file(MASK_path)
    mask = tf.image.decode_bmp(mask_name, channels=3)
    mask = tf.to_float(mask)
    masked_img = tf.multiply(image, mask)

    return masked_img, weight, speed, log_length


Batch_size = 8
Buffer_size = 8
Epochs = 70
train_dataset = tf.data.Dataset.from_tensor_slices((images_path_r, labels_r, speeds_r, log_length_r))
train_dataset = train_dataset.map(parse_function, num_parallel_calls=8)
train_dataset = train_dataset.map(data_normalization, num_parallel_calls=8)
train_dataset = train_dataset.map(data_masking, num_parallel_calls=8)
train_dataset = train_dataset.batch(Batch_size)
train_dataset = train_dataset.repeat(Epochs)
train_dataset = train_dataset.prefetch(Batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((images_path_t, labels_t, speeds_t, log_length_t))
test_dataset = test_dataset.map(parse_function, num_parallel_calls=8)
test_dataset = test_dataset.map(data_normalization, num_parallel_calls=8)
test_dataset = test_dataset.map(data_masking, num_parallel_calls=8)
test_dataset = test_dataset.batch(Batch_size)
test_dataset = test_dataset.repeat(Epochs)
test_dataset = test_dataset.prefetch(Batch_size)

