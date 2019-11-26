# from utils import Sample_main

import visualizer_main
import tensorflow as tf
import argparse
import os
import platform

tf.logging.set_verbosity(tf.logging.ERROR)  # disable to see tensorflow warnings
machine_type = platform.uname()[0]

if machine_type == 'Windows':
    path_sep = '\win'
else:
    path_sep = '/'
path_sep = path_sep[0]

relative_path = os.getcwd()
run_path = relative_path.split(path_sep+'Live_cam_visualization')[0]+path_sep+'dataset'+path_sep+\
           'images'+path_sep+'validation'+path_sep+'2017_08_04_1128_Flow_Test_35_576'+path_sep
pickle_path = relative_path.split(path_sep+'Live_cam_visualization')[0]+path_sep+'dataset'+path_sep+\
              'validation_Path_sample_dataset.pickle'

# print(run_path)
def cam(in_run=run_path, in_pickle= pickle_path, view_mode=False):
    visualizer_main.cam_vis(in_run, in_pickle, view_mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--run_dir', default=run_path, type=str, help= '(Full path to desired run -- default set to '
                                                                       './dataset/images/validation/2017_08_04_1128_Flow_Test_35_576')
    parser.add_argument('-p', '--pickle_dir', default=pickle_path, type=str, help= '(Full path to pickle file -- default set to '
                                                                       './dataset/validation_Path_sample_dataset.pickle')
    parser.add_argument('-v', '--view_mode', default=False, type=bool, help='Use openCV to visualize if True, else use matplotlib, default set to False')
    args = parser.parse_args()
    cam(args.run_dir, args.pickle_dir, args.view_mode)

# In case referenced by other modules


if __name__ == '__main__':
    main()
