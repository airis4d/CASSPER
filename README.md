


# CASSPER 
**C**ryo-EM **A**utomatic **S**emantic **S**egmentation based **P**article pick**ER**

###### *A Semantic Segmentation based Particle Picking Algorithm for Single Particle Cryo-Electron Microscopy*

https://zenodo.org/badge/latestdoi/231048773

This repository contains the following:
## 1. CASSPER Labelling Tool
The Labelling Tool is used to generate segmented labels for fresh training of CASSPER. The tool and sample mrc files are given in the sub folder. All the labels used for the study is also provided. The instructions and Demo video for operating the labelling tool is also given in the subfolder.

## 2. CASSPER Training and Prediction code 
The mrc files and segmented labels are needed for fresh training. The detailed description is given in the subfolder-**Train_and_Predict**

## 3. CASSPER Pre-trained weights 

The **TSaved** folder containing the trained weights for different proteins obtained from CASSPER can be found [here](https://drive.google.com/drive/folders/1Vi4N8RSObD6Oa_pCRcyZ2MS8WzbDT-7b?usp=sharing "Google Drive").   
If prediction without training is to be done, the folder **TSaved**, containing the saved weights, should be added into the respective protein folder in **Train_and_Predict** folder.

### Instructions about using CASSPE
The detailed instructions about labelling, training and prediction are mentioned in the README files of the respective subfolders.
Instructions about labelling tool: https://github.com/airis4d/CASSPER/tree/master/Labelling_Tool and Training and prediction:https://github.com/airis4d/CASSPER/tree/master/Train_and_Predict

### Setting up CASSPER

In Linux, just run setup.sh to set up the virtual environment for CASSPER. In that case, you may skip the following procedure.

CASSPER runs on Python 3.6+. We recommend running it from within
a virtual environment.

#### Creating a virtual environment for CASSPER

##### Set up a virtual environment using pip and Virtualenv

If you are familiar with `virtualenv`, you can use it to create 
a virtual environment.

For Python 3.6, create a new environment
with your preferred virtualenv wrapper, for example:

* [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) (Bourne-shells)
* [virtualfish](https://virtualfish.readthedocs.io/en/latest/) (fish-shell)


Either follow instructions [here](https://virtualenv.pypa.io/en/stable/installation/) or install via
`pip`.
```bash
$ pip install virtualenv
```

Then, create a `virtualenv` environment by creating a new directory for a Python 3.6 virtualenv environment
```bash
$ virtualenv --python=python3.6 cassper
```
where `python3.6` is a valid reference to a Python 3.6 executable.

Activate the environment
```bash
$ source cassper/bin/activate
```

#### Install the required python packages

*Note: make sure that the environment is activated throughout the installation process.
When you are done, deactivate it using* 
`source deactivate`, *or* `deactivate` 
*depending on your version*.

In the project root directory, run the following to install the required packages.
Note that this commands installs the packages within the activated virtual environment.

```bash
$ pip install -r requirements.txt
```
## Running CASSPER
**Please remember to activate this virtual environment each time you run the codes and run the codes from respective sub-directories itself.** 

## **Publication**
This folder contains Particle stacks ( in **.star** format) obtained using crYOLO, CASSPER and Gautomatch for four proteins discussed in the paper https://www.nature.com/articles/s42003-021-01721-1. The 2D images used for 3D reconstruction is also marked and shown for all cases in the folder **Publication/2D_images** 
