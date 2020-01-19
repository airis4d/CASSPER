# CASSPER Labelling Tool
![label generator](images/image.png).  

A demo video showing the labeling tool functioning can be found here: https://youtu.be/h5dpQzWUEvE .  
### Usage

#### Step 1:
Copy the mrc files into the folder **mrc_directory**

#### Step 2:
Enter the following command in the terminal.
`python label_generator.py -i mrc_directory -o output_directory`
* `mrc_directory` specifies the location of the folder containing all the mrc files
* `output_directory` specifies the location to which the labels are to be stored.

#### Step 3
Adjust the `blur_sigma` and `contrast` trackbars until the protein and ice particles are clearly visible. 
Now adjust the `threshold` and `contour_area` trackbars to label only the desired particles.
#### Mouse control

Button | Description | 
--- | --- |
Left | Draw box while labeling ice
Left | To click on the four corners of the carbon contamination


#### Keyboard Shortcuts

Shortcut | Description | 
--- | --- |
<kbd>i</kbd> | color ice |
<kbd>c</kbd> | color carbon contamination |
<kbd>f</kbd> | save the current label |
<kbd>q</kbd> | ignore the current micrograph |
<kbd>Space</kbd> | save the current ice patch while labeling ice |
<kbd>ESC</kbd> | Finish labeling ice patches |
<kbd>ESC</kbd> | Finish labeling while carbon contamination annotation |


#### Step 4: 
Annotate the ice and carbon contaminations
![color ice](images/ice.png)



