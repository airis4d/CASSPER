## Folder and file Structure
```
builders/  
frontends/  
models/  
mrc_files/ 
utils/
Protein1/  
    └───labels/  
    └───class_dict.csv  
Protein2/  
    └───labels/  
    └───class_dict.csv  
predict2.py  
ProtiPr.py  
ProtiSem.cpython-36m-x86_64-linux-gnu.so  
PSPred.py  
PSTest.py  
PSTrain.py  
scale_write_img.py  
test2.py  
train2.py  
Train8.sh  
```  

## Fresh Training
### 1: Prepration of supplimentary files for Training
Put the mrc files for training in the `mrc_files/` folder and labels in the respective protein folder, say `Protein1/labels/`. Note that mrc files and labels should have the same root name. 
### 2: Training
Run the command `./cassper_train.sh Protein1`. The .sh is followed by the folder name. Default, `Protein1` is the folder name.   

## Prediction using trained model

### Extraction of coordinates directly.
#### Step 1
Run `./get_radius.sh folder/` and adjust the track bars to get the radius and erode iteration value. Erode is done to diconnect the particles which are stacked together, if any. The key 'q' has to be pressed after each image and the radius and erode values will be displayed in the terminal after two images 
#### Step 2
Run `python predict_coordinates.py --mrc mrc_files --rads radius --erode erode_val` and the coordinates are saved as `.star` files in `Protein1/Star` and `.box` coordinates are saved in `Protein1/Box`

## Prediction using cross model or pretrained model
The trained model will get saved in the folder -`Protein1/TSaved`. If we want cross model or a  pretrained model, just replace the TSaved folder. Pretrained models correspoding to different proteins mentioned in the paper  and the **cross model** trained with 15 proteins can be downloaded from the link:[Pretrained_and_cross_cassper_models](https://drive.google.com/drive/folders/1Vi4N8RSObD6Oa_pCRcyZ2MS8WzbDT-7b?usp=sharing "Google Drive").

### Prediction of labels and extract the coordinates (Debug mode- if the visualization of prediction is needed). 
#### Step 1 
To predict the segmented labels using the trained model, enter the command **`./predict_labels.sh Protein1`** in the terminal. The predicted images will be saved in the folder- `Protein1/Predict_labels`. 
#### Step 2
To get the centre of the particles in `.star` format, run the code `python extract_coordinates_from_labels.py -i Protein1/Predict_labels -o Protein1/star_coordinates`. `Protein1/star_coordinates` is the location of the folder to which the coordinates are to be saved.  The coorderdinates will be saved in this folder in *.star* and *.box*format
A demo video showing the working of the code can be seen [here](https://youtu.be/wxdpRDVdJZY) .




