# from utils import Sample_main

import gradcam_main
import numpy as np
import tensorflow as tf
import argparse
import os
tf.logging.set_verbosity(tf.logging.ERROR)  # disable to see tensorflow warnings


def cam(in_path='sample.bmp', out_path = 'sample.png',):
    gradcam_main.cam_vis(in_path, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_image', default='sample.bmp', type=str, help= '(Full name of the input image -- default set to sample.bmp')
    parser.add_argument('-o', '--output_image', default='sample.png', type=str,   help='Full name of output image (should be .png) -- default set to '
                                                                                       'input_image.png')
    args = parser.parse_args()
    if args.input_image != 'sample.bmp' and args.output_image == 'sample.png':
        out_name = args.input_image
        out_name = out_name.replace('bmp', 'png')
    else:
        out_name = args.output_image
        out_name = out_name.replace('bmp', 'png')
    cam(args.input_image, out_name)

# In case referenced by other modules


if __name__ == '__main__':
    main()
