
# Mass Estimation from Images with Sparse ground Truth using DNN
### Implementation
The code provided herein is implemented in TF1.12 and compatible with eager mode.

## Run
To run the code, simply run the estimate.py with specifying the following arguments directly in terminal.

###Args
* '-n', '--network_size', default=None, type=int, help= '(9: RES9E, 16:RES16E) -- default set to: RES9ER'
* '-b', '--batch_size',default=8, type=int,help='(between 1<=b<=511 (log size=511). depends on GPU/CPU ram capacity -- default set to: 8 '
* '-t', '--train_mode', default=0, type=int, help='0: No training, 1: continue with existing checkpoint, 2: train from scratch) -- set to default: 0 '
* '-e', '--training_epochs', default=10, type=int, help='-- default set to 10'
* '-v', '--visualize', default=1, type=int, help='(0, No visualization, 1: validate and visualize log signal) -- defualt set to: 1 '
* '-l', '--logs', default=2, type=int, help='(Logs to visualize--> 0, train logs, 1: validate logs, 2: test logs) -- defualt set to: 2 '
### Example use  -- this runs in training mode with existing checkpoints then visualize the predicted signal of the test log/s
 - python3 estimate.py -t 1 


#### Note:
* Test accuracy of test log using RES9_ER should give an accuracy of 99.45% and if trained with option 1 for 1 epoch (i.e. python3 estimate.py -t 1 -e 1), accuracy can top 99.67%. This attached code is tested with TF1.12 and compabatible with linux and windows machines. Also, make sure to include/install all TF dependencies as per used in the code.
* when training, checkpoints for certain accuracies are automatically saved in checkpoints folder inside generated checkpoints folder

### Aditional Note
Gradcam code is provided seperately in the Gradcam_visualization folder, navigate to the Readme file in that folder for instructions on usage.

## Author
Muhammad K.A.Hamdan
