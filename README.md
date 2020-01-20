# CASSPER
## A Semantic Segmentation based Particle Picking Algorithm for Single Particle Cryo-Electron Microscopy

This repository contains the following:
## 1. CASSPER Labelling Tool
The Labelling Tool is used to generate segmented labels for fresh training of CASSPER. The tool and sample mrc files are given in the sub folder. All the labels used for the study is also provided. The instructions and Demo video for operating the labelling tool is also given in the subfolder.

## 2. Fresh Training
The mrc files and segmented labels are needed for fresh training. The detailed description is given in the subfolder-**Train_and_Predict**

## 3. Prediction using trained model
If prediction without training is to be done, the folder TSaved, containing the saved weights, should be added into the respective protein folder in **Train_and_Predict** folder.

The saved weights used by CASSPER can be found [here](https://drive.google.com/drive/folders/1Vi4N8RSObD6Oa_pCRcyZ2MS8WzbDT-7b?usp=sharing "Google Drive").

Please remember to run the commands from the respective sub directories itself. 

## Installation
A python3 virtual environment with libraries mentioned in `requirements.txt` is to be set up. 
The code cassper.sh will create a pthon3 virtual environment and install required libraries. 
