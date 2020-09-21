from PIL import Image,ImageEnhance
import numpy as np,csv
import cv2,os,argparse
import mrcfile,time
from scipy import ndimage
from functools import partial
import itertools
import pandas as pd
import multiprocessing
pool = multiprocessing.Pool()
from matplotlib import pyplot 

st=time.time()

# Codes for star from labels and pr_curve

from PIL import Image,ImageEnhance
import numpy as np,csv
import cv2,os,argparse
import mrcfile,time
from scipy import ndimage
from functools import partial
import itertools
import pandas as pd
import multiprocessing
pool = multiprocessing.Pool()
from matplotlib import pyplot 

w=260


def bbox_from_point(point):
    return(point[0]-w/2,point[1]-w/2,point[0]+w/2,point[1]+w/2)
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou







# Using argparse to define the input and output directory.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="directory of the predicted labels")
ap.add_argument("-g", "--groundtruth",default='star/',required=False,help="directory in which the ground truth star file is stored")

ap.add_argument("-t", "--iou_thresh",type=float,default=0.3,required=False,help="iou Threshold")

#args = vars(ap.parse_args())
args = ap.parse_args()

pred_dir=args.input
gt_dir=args.groundtruth
#out_dir=args.output

iou_thresh=args.iou_thresh



print("\n***** PR Curve Generation *****")
print("SS Images -->", args.input)
print("GT Folder -->", args.groundtruth)




precision_list=[]
recall_list=[]
actual_list=[]
pred_list=[]
TP_sum=0
FP_sum=0
FN_sum=0
f1_sum=0
f2_sum=0
total_gt=0
total_pred=0
precision_sum =0
recall_sum =0
mean_iou_sum =0
total_grond=0
images_with_particles=0
for fln in (os.listdir(pred_dir)):
    
    pred_df=pd.read_csv(os.path.join(pred_dir,fln),skiprows=9,header=None,sep='\t')
    pred_cord=pred_df[[0,1]]
    pred_tuples = [tuple(x) for x in pred_cord.values]
     
    gt_df=pd.read_csv(os.path.join(gt_dir,fln),skiprows=9,header=None,sep='\t')
    gt_cord=gt_df[[0,1]]
    gt_tuples = [tuple(x) for x in gt_cord.values]
   
    pred_found_index=[]
    ground_matching_index=[]
    len_gt_tuples=0
    len_pred_tuples=0
    MEAN_IOU=0
    TP=0
    
    
    pred_bbox = list(map(bbox_from_point, pred_tuples))
    gt_bbox = list(map(bbox_from_point, gt_tuples))
    total_gt+=len(gt_cord)
    total_pred+=len(pred_cord)
    len_gt_tuples=len_gt_tuples+len(gt_cord)
    len_pred_tuples=len_pred_tuples+len(pred_bbox)
    for i,boxes in enumerate(gt_bbox):
        actual_list.append(1)
        bbox=np.array([boxes]*len(pred_bbox))
        iou=list(map(bb_intersection_over_union,bbox,pred_bbox))
        if(iou):
            max_iou=np.max(iou)
        else:
            pred_list.append(0)
            continue
        if max(iou)>iou_thresh:
            pred_found_index.append(np.argmax(iou))
            ground_matching_index.append(i)
                
            del pred_bbox[np.argmax(iou)]
            del pred_tuples[np.argmax(iou)]
            TP=TP+1
            MEAN_IOU += max_iou
            total_grond=total_grond+1
            pred_list.append(1)
        else:
            pred_list.append(0)
    if len(gt_tuples) > 0:
        FN = len_gt_tuples - len(pred_found_index)
        #print(FN,len_cry_tuples,len(pred_found_index))
        FP = len_pred_tuples - len(ground_matching_index)
        if (TP + FP) == 0:
            precision = 0
        else:
            precision = 1.0 * TP / (TP + FP)

            recall = 1.0 * TP / (TP + FN)
        if (precision + recall) == 0:
            F1 = 0
            F2 = 0
        else:
            F1 = 2.0 * precision * recall / (precision + recall)
            F2 = 5.0 * precision * recall / (4 * precision + recall)
        f1_sum += F1
        f2_sum += F2
        precision_sum += precision
        recall_sum += recall
        mean_iou_sum += MEAN_IOU / (TP + 0.001)
        images_with_particles = images_with_particles + 1
        TP_sum += TP
        FP_sum += FP
        FN_sum += FN       
        print(fln,len(gt_cord),len(pred_cord),TP,FN,FP)       

f1_avg = f1_sum / images_with_particles
f2_avg = f2_sum / images_with_particles
precision_avg = precision_sum / images_with_particles
recall_avg = recall_sum / images_with_particles
mean_iou_sum = mean_iou_sum / images_with_particles 
total_precision=1.0 * TP_sum / (TP_sum + FP_sum)   
total_recall=1.0 * TP_sum / (TP_sum + FN_sum)        
with open('results.txt', "a") as boxfile:                    
    boxwriter = csv.writer(boxfile, delimiter="\t", quotechar="|", quoting=csv.QUOTE_NONE)
    #boxwriter.writerow([fln,iou_val])
    boxwriter.writerow(['TP:',TP_sum,'FP:',FP_sum,'FN:',FN_sum,'Precision:',total_precision,'Recall:',total_recall])
print('TP:',TP_sum,'FP:',FP_sum,'FN:',FN_sum,'Precision:',total_precision,'Recall:',total_recall)            







