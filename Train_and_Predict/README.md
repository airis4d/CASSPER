## Fresh Training
### 1:
Put the mrc files for training in the `mrc_files` folder and labels in the folder- `Protein1/labels`. Note that mrc files and labels have same root name. 
### 2:
Run the command `./cassper_train.sh Protein1`. The .sh is followed by the folder name. In this folder `Protein1` is the folder.   

## Prediction using trained model. 
#### Step1
To predict the segmented labels using the trained model, enter the command **`./cassper_predict.sh Protein1`** in the terminal. The predicted images will be saved in the folder- `Protein1/Predict_labels`. 

The trained model will be saved in the folder -`Protein1/TSaved1`. If we want to use another pretrained model, just replace the TSaved folder. Pretrained models correspoding to different proteins mentioned in the paper  and the **cross model** trained with 15 proteins can be downloaded from the link:

#### Step 2
To get the centre of the particles in star format, run the code `python star_from_labels,py -i Protein1/Predict_labels -o Protein1/star_coordinates`.The star files will be saved in folder `Protein1/star_coordinates`.
