import cv2
import os
import numpy as np
import argparse

def callback(x):
    pass

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="directory of the Predicted Labels")

args = vars(ap.parse_args())

dir1 = args['input']

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# create trackbars 
cv2.createTrackbar('radius','image',50,200,callback)
cv2.createTrackbar('erosion','image',0,10, callback)
radss=[]
eros=[]
j=0
for fln in os.listdir(dir1):
    img = cv2.imread(os.path.join(dir1,fln))
    if img is None:
        continue
        
    while(1):
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
           break 

        # get current positions of the trackbars
        radius = cv2.getTrackbarPos('radius','image')
        erode = cv2.getTrackbarPos('erosion','image')
        
        img[np.any(img != [0, 0, 255], axis=-1)]=[0,0,0]
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(gray_frame,0,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(6,6))
        thresh1 = cv2.erode(thresh1, kernel,iterations=erode)
        circle_frame=cv2.cvtColor(thresh1,cv2.COLOR_GRAY2BGR)
        
        contours,_ = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c)>800 and cv2.contourArea(c)<80000:
                (x,y),_ = cv2.minEnclosingCircle(c)
                center = (int(x),int(y))
                cv2.circle(circle_frame, center, radius, (0, 255, 0), 3)   
        cv2.imshow('image',circle_frame)
    j+=1

    radss.append(radius)
    eros.append(erode)
    #print("Your required Radius is",radius)
    #print("Your required erosion count is",erode)
    if j==2:
        break

print("Your required Radius is",np.int(np.mean(radss)))
print("Your required erosion count is",np.int(np.mean(eros)))
cv2.destroyAllWindows()
