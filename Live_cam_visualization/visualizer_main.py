from __future__ import absolute_import, division, print_function
from model import RES_9ER_CAM as Res
from utils import cam_functions as cf
from termcolor import colored
from moviepy.video.io.bindings import  mplfig_to_npimage
import subprocess
import tensorflow as tf
import os                   # work with directories
import numpy as np          # dealing with arrays
import matplotlib.pyplot as plt
import time
import  cv2
import pickle as pk
import platform
from skimage.transform import resize
import warnings
# import keyboard
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



def open_file(path):
    if machine_type == "Windows":
        os.startfile(path)
    elif machine_type == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
        



def cam_vis(dir_path, pickle_path):
    cost_arr = []
    draw_cost = []
    idx_arr = []
    pickle_in = open(pickle_path, 'rb')
    pickle_data = pk.load(pickle_in)
    pickle_in.close()
    speed = pickle_data['speed']
    weight = pickle_data['weight'][0]
    t = 1/7.5
    acm_prd = 0
    images = sorted(os.listdir(dir_path))
    dir_len = len(images)
    LBKG = 0.453
    OPENCV = True
    if OPENCV:
        fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(7.2, 4.8))
    else:
        fig, (ax, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    ax2.axis([0, dir_len+10, -0.2, 8])
    t2 = ax2.text(5, 7.5, 'Prediction: {0:3f}'.format(0))
    t3 = ax2.text(dir_len - 150, 7.5, 'Acm Prediction: {0:3f}'.format(0),color='blue')
    t4 = ax2.text(dir_len - 150, 7,   'Ground Truth: {0:3f}'.format(0), color='green')
    t5 = ax2.text(5, 7, 'Completion: {0:.1%}'.format(0))
    plt.xlabel('Number of images in the run')
    plt.ylabel('Weight (lbs)')
    cfact = 0.162
    cnt = 0
    # Instantiate the model and configure tensorbaord and checkpoints
    data_format = 'channels_last'
    model = Res.Res9ER(data_format=data_format, include_top=True, pooling=None, classes=1)

    checkpoint_name = 'Gradcam_RES_9ER'
    model.load_weights(checkpoint_path + checkpoint_name)
    warnings.filterwarnings('ignore')
    start = time.time()
    prediction_raw = []

    step = tf.train.get_or_create_global_step()
    for idx, img in enumerate(images):
        in_path = os.path.join(dir_path, img)
        sample_image = cf.data_normalization(cf.data_resize(cf.parse_function(in_path)))
        with tf.device('/gpu:0'):
            sample_image = tf.reshape(sample_image, [1, 96, 144, 3])
            image = tf.convert_to_tensor(sample_image)
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

            if OPENCV:
                img = cv2.imread(in_path)
            else:
                img = plt.imread(in_path)

            img = resize(img, (480, 720))
            cost = cost.numpy()[0][0]
            cost = cost * speed[idx] * t
            cost_arr.append(cost)
            idx_arr.append(idx)
            if cost < 0:
                cost = np.abs(cost)/10
            # elif idx > 5:
            #     cost = cost + cfact

            acm_prd += cost
            draw_cost.append(cost)


            cam = (cam * -1.0) + 1.0
            cam_heatmap = np.array(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET))
            cam_heatmap = cam_heatmap / 255.0

            fin = (img * 0.7) + (cam_heatmap * 0.3)
            print(idx)

            if OPENCV:

                # print(np.shape(plot_curve), np.shape(fin))
                ax2.plot(idx_arr, draw_cost, color='red', linestyle='dashed', linewidth=0.8)
                t2.set_text('Prediction: {0:.3f}'.format(cost))
                t3.set_text('Acm Prediction: {0:.3f}'.format(acm_prd))
                t4.set_text('Ground Truth: {0:.1f}'.format(weight))
                t5.set_text('Completion: {0:0.1%}'.format((idx + 1) / dir_len))

                plot_curve = resize(mplfig_to_npimage(fig), (320, 480))
                stacked = np.hstack((resize(fin, (320, 480)), plot_curve, resize(img,(320, 480))))
                # cv2.namedWindow('frame')
                # cv2.resizeWindow('frame', 1024, 1024)
                cv2.imshow('frame', stacked)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                with tf.device('/cpu:0'):
                    ax.imshow(img)
                    ax1.imshow(fin)
                    ax2.plot(idx_arr, draw_cost, color='red', linestyle='dashed', linewidth=0.8)
                    t2.set_text('Prediction: {0:.3f}'.format(cost))
                    t3.set_text('Acm Prediction: {0:.3f}'.format(acm_prd))
                    t4.set_text('Ground Truth: {0:.1f}'.format(weight))
                    t5.set_text('Completion: {0:0.1%}'.format((idx+1)/dir_len))
                    plt.pause(.001333)
                    plt.draw()

                    ## Uncomment this section to exit on key stroke
                    # try:  # used try so that if user pressed other than the given key error will not be shown
                    #     if keyboard.is_pressed('q'):  # if key 'q' is pressed
                    #         print('You Pressed A Key!')
                    #         break  # finishing the loop
                    #     else:
                    #         pass
                    # except:
                    #     break

    # plt.show()aa
    if OPENCV:
        cv2.destroyAllWindows()
    print('Processing time:', time.time()-start)

    # print(colored('successfully generated a cam', 'blue'))

