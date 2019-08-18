# GradCAM Visualization of Mass Estimation From Images
### Implementation
The code provided herein is implemented in TF1.12 and compatible with **Eager mode**.

## To run the code
Simply run the **live_visualization.py** with specifying the following arguments directly in terminal.
If you would like to visualize a paritcular run, make sure to provide the directory of run to the flags (runs are found in the  *./dataset/images* folder)

### Args
* '-i', '--input_dir', default='dataset/images/validation/2017_08_04_1128_Flow_Test_35_576', type=str, help= '(Full path of the run directory -- default set to validation dir'
*  '-p', '--pickle_dir', default=pickle_path, type=str, help= '(Full path to pickle file -- default set to '
                                                                       './dataset/validation_Path_sample_dataset.pickle'
                                                                       
### Simplist use  
This runs the live_cam with **validation/2017_08_04_1128_Flow_Test_35_576** run located in *_/dataset/images/_* 
 - python3 live_visualization.py

#### Output
This code produces an CAM video and live plotting of predictions for select run

##### Acknowledgements
Part of the code was adapted from:
TF session implementation [here](https://github.com/adityac94/Grad_CAM_plus_plus).



# Author
Muhammad K.A. Hamdan
