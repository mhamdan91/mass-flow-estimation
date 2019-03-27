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


'''
Provide your dataset that needs to be consumed

'''
# dataset_dir = MAIN_dir + '2017_08_04_1043_Flow_Test_30_587_regression/'

labels_r = []
images_path_r = []
speeds_r = []
log_length_r = []

labels_t = []
images_path_t = []
speeds_t = []
log_length_t =[]

# raw_label = np.load("predicted_labels_208.npy")

elv_speed_train = np.load('train_sorts_speeds.npy')
elv_speed_test = np.load('test_sorts_speeds.npy')

lengths_t = [251, 527, 433, 361]
lengths_r = [511, 520, 147, 448, 553, 254, 622] 
weight_t = [619, 0, 626, 555]
weight_r = [640, 0, 5.7, 623, 0, 636, 623]
# elv = np.concatenate((elv_speed_2[:208], elv_speed))

log_length =[]
for index, subset_ in enumerate(sorted(os.listdir(MAIN_dir))):
    subset_path = MAIN_dir + subset_ + '/'
    for idx, log in enumerate(sorted(os.listdir(subset_path))):
        log_path = subset_path + log+'/'
        image_files = sorted(os.listdir(log_path))
        print(log_path)
        for i, image_ in enumerate(image_files):
            if index == 0:
                images_path_t.append(log_path+image_)
                labels_t.append(np.float32([weight_t[idx]]))
                speeds_t.append([elv_speed_test[i]])
                log_length_t.append(np.float32(lengths_t[idx]))
            else:
                images_path_r.append(log_path+image_)
                labels_r.append(np.float32([weight_r[idx]]))
                speeds_r.append([elv_speed_train[i]])
                log_length_r.append(np.float32(lengths_r[idx]))
        if idx == 1: # get 2 logs only 
            break
labels_r = np.asarray(labels_r)
labels_t = np.asarray(labels_t)

speeds_r = np.asarray(speeds_r) 
speeds_t = np.asarray(speeds_t) 
print(np.shape(labels_r), np.shape(images_path_r), np.shape(speeds_r), np.shape(log_length_r))




'''
The following three functions are used to read data, normalize images, and apply a certain mask to images.
you can write your own function to do what you need
'''

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



'''
This code segment is responsible for consuming the dataset.
Dataset should be fed in the form of tensor slices i.e numpy arrays of matching dimensions
'''
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

