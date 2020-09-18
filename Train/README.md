Fresh Training
1: Prepration of supplimentary files for Training
Put the mrc files for training in the mrc_files/ folder and labels in the respective protein folder, say Protein1/labels/. Note that mrc files and labels should have the same root name.

2: Training
Run the command ./cassper_train.sh Protein1. The .sh is followed by the folder name. Default, Protein1 is the folder name.

Prediction using trained model.
Step1
To predict the segmented labels using the trained model, enter the command ./cassper_predict.sh Protein1 in the terminal. The predicted images will be saved in the folder- Protein1/Predict_labels.

Step 2
To get the centre of the particles in .star format, run the code python star_from_labels.py -i Protein1/Predict_labels -o Protein1/star_coordinates. Protein1/star_coordinates is the location of the folder to which the coordinates are to be saved. The coorderdinates will be saved in this folder in .star format A demo video showing the working of the code can be seen here .
