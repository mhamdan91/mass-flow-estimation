from __future__ import absolute_import, division, print_function
from tqdm import tqdm
from termcolor import colored
from utils import functions
from utils import volume_mass_predictor as vm_prd
from utils.validate import validate_model
from utils.train import train
import warnings
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import platform
warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)  # disable to see tensorflow warnings
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
tf.executing_eagerly()
print(tf.__version__)

'''
THIS CODE SNIPPETS HANDLES PATH SETUP
SETUP BASICS START
'''
machine_type = platform.uname()[0]

if machine_type == 'Linux':
    path_sep = '/'
else:
    path_sep = '\win'
path_sep = path_sep[0]
MAIN_dir = os.getcwd() + path_sep
checkpoint_path = MAIN_dir + 'checkpoints' + path_sep
data_files_path = MAIN_dir + 'data_files' + path_sep
mean_path = data_files_path + 'std_mean_60_dataset.npy'
dataset_path = MAIN_dir + 'dataset' + path_sep
tensorboard_path = MAIN_dir + 'tensorboard' + path_sep

'''
SETUP BASICS END-- CHANGE ACCORDINGLY IF NEEDED
'''


def load_dataset(batch_size):
    pickle_in_train = open(dataset_path + "training_Path_sample_dataset.pickle", "rb")
    training_data = pk.load(pickle_in_train)
    pickle_in_train.close()

    pickle_in_validate = open(dataset_path + "validation_Path_sample_dataset.pickle", "rb")
    validation_data = pk.load(pickle_in_validate)
    pickle_in_validate.close()
    pickle_in_test = open(dataset_path + "testing_Path_sample_dataset.pickle", "rb")
    testing_data = pk.load(pickle_in_test)
    pickle_in_test.close()

    training_relative = training_data['img_path']  # image in training subset
    training_w = np.float32(np.reshape(training_data['weight'], [-1, 1]))  # weight of material i.e. ground truth
    training_s = np.reshape(training_data['speed'], [-1, 1])  # elevator speed
    training_l = np.float32(training_data['log_length'])  # log length i.e. number of images in a log
    training_v = np.reshape(training_data['volume'], [-1, 1])  # instant volume recorded by the stereo camera
    training_img = functions.refactor_path(training_relative, dataset_path, path_sep)
    train_data = [training_img, training_w, training_s, training_l, training_v]

    validation_relative = validation_data['img_path']
    validation_w = np.float32(np.reshape(validation_data['weight'], [-1, 1]))
    validation_s = np.reshape(validation_data['speed'], [-1, 1])
    validation_l = np.float32(validation_data['log_length'])
    validation_v = np.reshape(validation_data['volume'], [-1, 1])
    validation_img = functions.refactor_path(validation_relative, dataset_path, path_sep)
    validation_data = [validation_img, validation_w, validation_s, validation_l, validation_v]

    testing_relative = testing_data['img_path']
    testing_w = np.float32(np.reshape(testing_data['weight'], [-1, 1]))
    testing_s = np.reshape(testing_data['speed'], [-1, 1])
    testing_l = np.float32(testing_data['log_length'])
    testing_v = np.reshape(testing_data['volume'], [-1, 1])
    testing_img = functions.refactor_path(testing_relative, dataset_path, path_sep)
    test_data = [testing_img, testing_w, testing_s, testing_l, testing_v]


    # Intantiate TF DATASET API -- START OF DATA PIPELINE
    train_dataset = tf.data.Dataset.from_tensor_slices((training_img, training_w, training_s, training_l))
    train_dataset = train_dataset.map(functions.parse_function, num_parallel_calls=8)
    train_dataset = train_dataset.map(functions.data_resize, num_parallel_calls=8)
    train_dataset = train_dataset.map(functions.data_normalization, num_parallel_calls=8)
    train_dataset = train_dataset.map(functions.data_masking, num_parallel_calls=8)
    train_dataset = train_dataset.batch(batch_size)
    # train_dataset = train_dataset.repeat(Epochs)
    train_dataset = train_dataset.prefetch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((testing_img, testing_w, testing_s, testing_l))
    test_dataset = test_dataset.map(functions.parse_function, num_parallel_calls=8)
    test_dataset = test_dataset.map(functions.data_resize, num_parallel_calls=8)
    test_dataset = test_dataset.map(functions.data_normalization, num_parallel_calls=8)
    test_dataset = test_dataset.map(functions.data_masking, num_parallel_calls=8)
    test_dataset = test_dataset.batch(batch_size)
    # test_dataset = test_dataset.repeat(Epochs)
    test_dataset = test_dataset.prefetch(batch_size)

    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_img, validation_w, validation_s, validation_l))
    validation_dataset = validation_dataset.map(functions.parse_function, num_parallel_calls=8)
    validation_dataset = validation_dataset.map(functions.data_resize, num_parallel_calls=8)
    validation_dataset = validation_dataset.map(functions.data_normalization, num_parallel_calls=8)
    validation_dataset = validation_dataset.map(functions.data_masking, num_parallel_calls=8)
    validation_dataset = validation_dataset.batch(batch_size)
    # test_dataset = test_dataset.repeat(Epochs)
    validation_dataset = validation_dataset.prefetch(batch_size)

    return train_dataset, test_dataset, validation_dataset, train_data, test_data, validation_data


def run_and_visualize_signal(batch_size, model, summary_writer, target_dataset, target_data, logs_N, target_subset):
    # Generate predicted signal with proper information and save it
    # Obtain a signal
    signals = validate_model(batch_size, model, 1, logs_N, summary_writer, target_dataset, write_summary=False, return_losses=False)

    signal = []
    signalz = []
    signalz = []
    onehot_signals = []
    speed_signal = []
    volume_signal = []
    cnt = 0
    target_name = 'sample_signal.npy'
    instant_mass = []
    sum_x = 0
    for sig in signals:
        for si in sig:
            for s in si:
                sm = np.float32(np.squeeze(s.numpy()))
                signal.append(sm)
                speed_signal.append(target_data[2][cnt])
                volume_signal.append(target_data[4][cnt])
                x = vm_prd.model_pos(np.float(target_data[4][cnt])) * target_data[2][cnt]
                instant_mass.append(x)
                sum_x += x
                onehot_signals.append(sm)
                cnt += 1
        signalz.append([target_data[1][cnt - 1], target_data[3][cnt - 1], signal[:len(signal)],
                        speed_signal[:len(speed_signal)], instant_mass[:len(instant_mass)], volume_signal[:len(volume_signal)]])
        signal.clear()
        speed_signal.clear()
        volume_signal.clear()
        instant_mass.clear()

    np.save(data_files_path + target_name, signalz)
    # Visualize Predicted signal
    subset_names = ['train', 'test', 'validation']
    title = ''
    for name in subset_names:
        if name == target_subset:
            title = 'Visualization of ' + name + ' logs'
            break
    lb2kg = 0.453592
    # NOT DOING SUBPLOTS because maximum output plots are 2
    for i in range(len(signalz)):
        gt = np.squeeze(signalz[i][0]) * lb2kg
        prd = np.sum(signalz[i][2]) * lb2kg
        vprd = np.sum(signalz[i][4]) * lb2kg
        plt.figure(i+1)
        if machine_type == 'Linux':
            vol_sig = signalz[i][4]
        else:
            vol_sig = (np.reshape(signalz[i][4], [len(signalz[i][4])]))
        plt.plot(vol_sig, '--', color='red')
        plt.plot(signalz[i][2], '--', lineWidth=2)
        relative = ''
        if gt > 0:
            ACC = (1 - np.abs(gt - prd) / gt)*100
            vACC = (1 - np.abs(gt - vprd) / gt)*100
        else:
            # average total weight = 598*lb2kg = ~271 kg-- relative accuracy
            ACC = (1 - np.abs((271+gt) - (271+prd)) / (271+gt))*100
            vACC = (1 - np.abs((271+gt) - (271+vprd)) / (271+gt))*100
            relative = 'Relative '
        gt_kg = '{:.2f}'.format(gt)
        Prd_kg = '{:.2f}'.format(prd)
        vPrd_kg = '{:.2f}'.format(vprd)
        ACC1 = np.float('{:.2f}'.format(ACC))
        vACC1 = np.float('{:.2f}'.format(vACC))
        bottom, top = plt.ylim()
        left, right = plt.xlim()
        lshift = (0.2*len(signalz[i][2]))
        bshift = (.22*top)
        plt.title(title)
        plt.legend(('Volume-Based Prediction Signal', 'DNN-Based Prediction Signal'), shadow=False, loc='upper right', handlelength=1, fontsize=8)
        print(colored('\nRun {0:} - Ground Truth:{1:} DNN Prediction:{2:} Accuracy:{3:.2%} '.format(i+1, gt_kg, Prd_kg, ACC1/100), 'red'))
        plt.xlim(left-lshift, right)
        plt.text(abs(.5*left)-lshift, top-bshift, 'Ground Truth:' + str(gt_kg) + 'Kgs' +
                 '\nDNN-Based Prediction:' + str(Prd_kg) + 'Kgs' +
                 '\n'+relative+'Accuracy:' + str(ACC1) + '%' +
                 '\nVolume-Based Prediction:' + str(vPrd_kg) + 'Kgs' +
                 '\n'+relative+'Accuracy:' + str(vACC1) + '%',
                 {'color': 'k', 'fontsize': 8, 'bbox': dict(boxstyle="square", fc="w", ec="k", pad=0.5, alpha=0.3)})
        if i == len(signalz)-1:
            plt.show(i+1)


def predictor(network_size=None, batch_size=8, train_mode=0, epochs=10, visualize=True, target_subset='test'):
    if batch_size > 215 or batch_size < 1:
        batch_size = 8
    if train_mode > 2 or train_mode < 0:
        train_mode = 0
    if visualize > 1 or visualize < 0:
        visualize = 1

    train_dataset, validation_dataset, test_dataset, train_data, validation_data, test_data = load_dataset(batch_size)
    summary_writer = tf.contrib.summary.create_file_writer(tensorboard_path)
    # network_size = None  # Set to default RES-9ER
    logsN = len(os.listdir(dataset_path+'images'+path_sep+'training'+path_sep))
    data_format = 'channels_last'
    if network_size == 16:
        from models import RES_16E as Res
        model = Res.Res16E(data_format=data_format, include_top=True, pooling=None, classes=1)
    elif network_size == 9:
        from models import RES_9E as Res
        model = Res.Res9E(data_format=data_format, include_top=True, pooling=None, classes=1)
    else:
        from models import RES_9ER as Res
        model = Res.Res9ER(data_format=data_format, include_top=True, pooling=None, classes=1)
    # Instantiate the model and configure tensorbaord and checkpoints
    print(colored('model was successfully constructed!', 'blue'))

    # LOAD checkpoint if you wish to test results
    if (train_mode == 0 and visualize) or train_mode == 1:
        if network_size == 9:
            checkpoint_name = 'RES_9E'
        elif network_size == 16:
            checkpoint_name = 'RES_16E'
        else:
            checkpoint_name = 'RES_9ER'
        model.load_weights(checkpoint_path + checkpoint_name)
        print(colored('checkpoint was successfully loaded!', 'blue'))

    if train_mode != 0:
        train(epochs, batch_size, model, summary_writer, train_dataset, validation_dataset, MAIN_dir, checkpoint_path, path_sep, logsN)

    if visualize:
        if target_subset == 'train':
            target_dataset = train_dataset
            target_data = train_data
            logs_N = logsN

        elif target_subset == 'validation':
            target_dataset = validation_dataset
            target_data = validation_data
            logs_N = 1
        else:
            target_dataset = test_dataset
            target_data = test_data
            logs_N = 1

        run_and_visualize_signal(batch_size, model, summary_writer, target_dataset, target_data, logs_N, target_subset)


