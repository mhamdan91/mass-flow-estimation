from __future__ import absolute_import, division, print_function
from model import RES_9ER_CAM as Res
from utils import cam_functions as cf
from termcolor import colored
import subprocess
import tensorflow as tf
import os                   # work with directories
import numpy as np          # dealing with arrays
import matplotlib.pyplot as plt
import time
import  cv2
import platform
from skimage.transform import resize
import warnings

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
layers = tf.keras.layers
tf.enable_eager_execution(config=config)
tf.executing_eagerly()
print(tf.__version__)

machine_type = platform.uname()[0]
if machine_type == 'Windows':
    path_sep = '\win'
else:
    path_sep = '/'
path_sep= path_sep[0]
MAIN_dir = os.getcwd() + path_sep
checkpoint_path = MAIN_dir +'checkpoint'+path_sep
model_path = MAIN_dir + 'model'+path_sep
input_sample = 'sample.bmp'
input_path = MAIN_dir + 'images'+path_sep + 'input' + path_sep
output_path = MAIN_dir + 'images'+path_sep + 'output' + path_sep


def visualize_mod(img, cam, filename):
    fig, ax = plt.subplots(nrows=1, ncols=3)

    plt.subplot(121)
    plt.axis("off")
    imgplot = plt.imshow(img)

    cam = (cam * -1.0) + 1.0
    cam_heatmap = np.array(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET))
    # plt.subplot(143)
    plt.axis("off")

    plt.subplot(122)
    plt.axis("off")

    cam_heatmap = cam_heatmap / 255.0

    fin = (img * 0.7) + (cam_heatmap * 0.3)
    imgplot = plt.imshow(fin)

    plt.savefig(filename, dpi=600)
    plt.close(fig)


def cam_vis(in_img_name, out_img_name):
    in_path = input_path + in_img_name
    out_path = output_path + out_img_name
    sample_image = cf.data_normalization(cf.data_resize(cf.parse_function(in_path)))
    sample_image = tf.reshape(sample_image, [1, 96, 144, 3])
    image = tf.convert_to_tensor(sample_image)

    # Instantiate the model and configure tensorbaord and checkpoints
    data_format = 'channels_last'
    model = Res.Res9ER(data_format=data_format, include_top=True, pooling=None, classes=1)

    checkpoint_name = 'Gradcam_RES_9ER'
    model.load_weights(checkpoint_path+checkpoint_name)

    warnings.filterwarnings('ignore')
    start = time.time()
    prediction_raw = []

    step = tf.train.get_or_create_global_step()
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(image)
        cost, layer_conv = model(image, conv_out=True)
        prediction_raw.append(cost)

    # this is for guided-backprop -- not used
    gb_grad = tape.gradient(cost, image)

    target_conv_layer_grad = tape.gradient(cost, layer_conv)
    del tape
    conv_first_grad = tf.exp(cost) * target_conv_layer_grad
    # second_derivative
    conv_second_grad = tf.exp(cost) * target_conv_layer_grad * target_conv_layer_grad

    # triple_derivative
    conv_third_grad = tf.exp(cost)[0] * target_conv_layer_grad * target_conv_layer_grad * target_conv_layer_grad
    global_sum = np.sum(tf.reshape(layer_conv[0], (-1, conv_first_grad[0].shape[2])), axis=0)

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum.reshape((1, 1, conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num / alpha_denom

    weights = np.maximum(conv_first_grad[0], 0.0)

    alphas_thresholding = np.where(weights, alphas, 0.0)

    alpha_normalization_constant = np.sum(np.sum(alphas_thresholding, axis=0), axis=0)
    alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant,
                                                      np.ones(alpha_normalization_constant.shape))
    alphas /= alpha_normalization_constant_processed.reshape((1, 1, conv_first_grad[0].shape[2]))

    deep_linearization_weights = np.sum(tf.reshape((weights * alphas), (-1, conv_first_grad[0].shape[2])), axis=0)

    # print deep_linearizat0=eglmoprstvion_weights
    grad_CAM_map = np.sum(deep_linearization_weights * layer_conv[0], axis=2)

    # Passing through ReLU
    cam = np.maximum(grad_CAM_map, 0)
    cam = cam / np.max(cam)  # scale 0 to 1.0

    cam = resize(cam, (480, 720))

    img = plt.imread(in_path)
    img2 = resize(img, (480, 720))
    # plt.imshow(img2)

    visualize_mod(img2, cam, out_path)
    print(colored('successfully generated a cam', 'blue'))

    if machine_type == 'Linux':
        subprocess.Popen(['xdg-open', output_path])
    elif machine_type == 'Windows':
        os.startfile(output_path)
