# from utils import Sample_main
import mass_flow
import numpy as np
import tensorflow as tf
import argparse
import os
tf.logging.set_verbosity(tf.logging.ERROR)  # disable to see tensorflow warnings

def predictor(Network_size=None, Batch_size = 8, train_mode=0, epochs_= 10, visualize=1, logs=2):
    if logs == 0:
        dataset = 'train'
    elif logs == 1:
        dataset = 'validation'
    else:
        dataset = 'test'
    Sample_main.predictor(Network_size, Batch_size, train_mode , epochs_, visualize, dataset)
    # print(Network_size,Batch_size, train_mode, epochs_,visualize)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network_size', default=None, type=int, help= '(9: RES9E, 16:RES16E) -- default set to: RES9ER')
    parser.add_argument('-b', '--batch_size',default=8, type=int,help='(between 1<=b<=215 (smallest log length=215). depends on GPU/CPU ram capacity -- '
    'default set to: 8 ')
    parser.add_argument('-t', '--train_mode', default=0, type=int, help='0: No training, 1: continue with existing checkpoint, '
    '2: train from scratch) -- set to default: 0 ')
    parser.add_argument('-e', '--training_epochs', default=10, type=int, help='-- default set to 10')
    parser.add_argument('-v', '--visualize', default=1, type=int, help='(0, No visualization, 1: validate and visualize log signal) -- defualt set to: 1 ')
    parser.add_argument('-l', '--logs', default=2, type=int, help='(Logs to visualize--> 0, train logs, 1: validate logs, 2: test logs) -- defualt set to: 2 ')
    args = parser.parse_args()
    predictor(args.network_size, args.batch_size, args.train_mode, args.training_epochs, args.visualize, args.logs)

# In case referenced by other modules
if __name__ == '__main__':
    main()
