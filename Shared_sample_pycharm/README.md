# Since the dataset size is too large to be attached as a supplimentary file (Size limit 100MB). 
# We provide code for a sample log that should work in the same manner as the entire dataset would. 


## To run the code, simply run the estunate.py with specifying the following arguments directly in terminal.

###Args
'-n', '--network_size', default=None, type=int, help= '(9: RES9E, 16:RES16E) -- default set to: RES9ER'
'-b', '--batch_size',default=8, type=int,help='(between 1<=b<=511 (log size=511). depends on GPU/CPU ram capacity -- default set to: 8 '
'-t', '--train_mode', default=0, type=int, help='0: No training, 1: continue with existing checkpoint, 2: train from scratch) -- set to default: 0 '
'-e', '--training_epochs', default=10, type=int, help='-- default set to 10'
'-v', '--visualize', default=1, type=int, help='(0, No visualization, 1: validate and visualize log signal) -- defualt set to: 1 '
### Example use  -- this runs in training mode with existing checkpoints then visualize the predicted signal
python3 estimate.py -t 1 

## For convenience, we also provide an implementation in Jupter notebook (Sample_main.ipyb)
you can specify the options for the notebook from within manually

## Validating using RES9EE should give an accuracy of 97.94% for the log and if trained with option 1, accuracy can top 99.98%.

### We will share the complete version with corrosponding codes and utils (Processing .py files, gradcam code, etc) via a github link after the blind review process is over.

# This attached code is tested on TF1.12 and compabatible with linux and windows machines. Also, make sure to include/install all TF dependencies as per used in the code.
Best,