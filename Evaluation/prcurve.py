from PIL import Image,ImageEnhance
import numpy as np,csv
import cv2,os,argparse
import mrcfile,time
from scipy import ndimage
from functools import partial
import itertools
import pandas as pd
from matplotlib import pyplot 



#to get a bounding box from a single point
def getbox_from_point(pt):
    return(pt[0]-w/2,pt[1]-w/2,pt[0]+w/2,pt[1]+w/2)

# compute the area of intersection between two boxes
def bbox_iou(boxx, boxy):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxx[0], boxy[0])
    yA = max(boxx[1], boxy[1])
    xB = min(boxx[2], boxy[2])
    yB = min(boxx[3], boxy[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxxArea = abs((boxx[2] - boxx[0]) * (boxx[3] - boxx[1]))
    boxyArea = abs((boxy[2] - boxy[0]) * (boxy[3] - boxy[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxxArea + boxyArea - interArea)

    # return the intersection over union value
    return iou





# Program to get the radius of the particle by adjusting the track bar
def radius_circle():
    c_frame=circle_frame.copy()
    for i in c_list:
        cv2.circle(c_frame, tuple(i), int(manual_picked_radius), (0, 255, 0), 3)
    cv2.namedWindow(source_window,cv2.WINDOW_NORMAL)
    cv2.imshow( source_window, c_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Call back function for trakbar
def radius_callback(val):
    global manual_picked_radius
    manual_picked_radius =val
    radius_circle()



def primary_sorting(i):
    print("primary_sorting done")
    i1,i2,i3,i4=i[1]-(radius*2),i[0]-(radius*2),i[1]+(radius*2),i[0]+(radius*2)
    if i1<0 and i2<0:
        center=(i1+radius*2,i1+radius*2)
        i1=0
        i2=0
    elif i3 > thresh1.shape[0] and i4 > thresh1.shape[1]:
        i3= thresh1.shape[0]
        i4 = thresh1.shape[1]
        center=(radius*2,radius*2)
    elif i1<0:
        center=(i1+radius*2,i1+radius*2)
        i1=0    
    elif i2<0:
        center=(i2+radius*2,i2+radius*2)
        i2=0
    elif i3 > thresh1.shape[0]:
        i3= thresh1.shape[0]
        center=(radius*2,radius*2)
    elif i4 > thresh1.shape[1]:
        i4 = thresh1.shape[1]
        center=(radius*2,radius*2)
    else:
        center=(radius*2,radius*2)	

    
    th1=thresh1[i1:i3,i2:i4]
    
    c_frame=circle_frame[i1:i3,i2:i4]
    mask = np.zeros_like(th1)
    mask2 = np.zeros_like(th1)
    mask3 = np.zeros_like(c_frame)
    mask=cv2.circle(mask, center=center, radius=radius,color=(255),thickness=-1)
    mask2=cv2.circle(mask2, center=center, radius=radius+10,color=(255),thickness=-1)
    mask3=cv2.circle(mask3, center=center, radius=radius+10,color=[255,255,255],thickness=-1)
    col_mask= np.bitwise_and(c_frame,mask3)
    result = np.bitwise_and(th1,mask)
    result2 = np.bitwise_and(th1,mask2)
    col_a=np.where(np.all(col_mask== [0,255,0] , axis=-1))
    col_b=np.where(np.all(col_mask== [0,255,255] , axis=-1))
    inn_positions=np.nonzero(result)
    out_positions=np.nonzero(result2)
    
    contours, hierarchy = cv2.findContours(result,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    l=len(contours)
    a,b= (inn_positions[0].max()-inn_positions[0].min()),(inn_positions[1].max()-inn_positions[1].min())
    c,d= (out_positions[0].max()-out_positions[0].min()),(out_positions[1].max()-out_positions[1].min())
    if (len(col_a[0]) or len(col_b[0]))>0 :
        pass
    elif (c or d) > (radius*2.1):
        pass
    elif l==1 and ((c and d) <(radius*2)):
        return(tuple([i[0],i[1]]))
    else:
        
        cv2.circle(frame, i, radius, (255, 0, 0), 3)
        ans={}
        centerx=range(0+radius,th1.shape[1]-radius,5)
        centery=range(0+radius,th1.shape[0]-radius,5)
        rad=range(radius,5,5)
        for cx,cy in itertools.product(centerx,centery):
            mask00 = np.zeros_like(th1)
            mask20=mask.copy()
            mask00=cv2.circle(mask00, center=(cx,cy), radius=radius, color=(255,255,255), thickness=-1)
            mask20=cv2.circle(mask20, center=(cx,cy), radius=radius+5, color=(255,255,255), thickness=-1)
            result00 = np.bitwise_and(th1,mask00)
            tot_ellip=np.sum(mask00 == [255])
            tot_ellip_white1=np.sum(result00 == [255])
            per_white=(tot_ellip_white1/tot_ellip)*100
            result002 = np.bitwise_and(th1,mask20)
            tot_ellip_white2=np.sum(result002 == [255])
            diff=tot_ellip_white2-tot_ellip_white1
        
            if diff<10 and per_white>10:
                ans[per_white]=(cx,cy)
            
        
        if len(ans.keys())>0:
            k=ans[max(ans.keys())]
            return (tuple([i2+k[0],i1+k[1]]))
        else:
            pass
            

def min_circle(cont):
    contours_poly = cv2.approxPolyDP(cont, 3, True)
    center, _= cv2.minEnclosingCircle(contours_poly)
    return(int(center[0]),int(center[1]))

def min_rect_circle(cont):
    contours_poly = cv2.approxPolyDP(cont, 3, True)
    center, _= cv2.minEnclosingCircle(contours_poly)
    rect=cv2.minAreaRect(cont)
    box=np.int0(cv2.boxPoints(rect))
    mn,mx=np.amin(box,axis=0),np.amax(box,axis=0)
    diff=mx-mn
    if np.all(diff<(2*radius+40)):
        return(int(center[0]),int(center[1]))
    else:
        pass


def eliminate_near(fields):
    fields=np.array(fields,dtype=np.int32)
    i=np.arange(len(fields))
    print('inside eliminate',len(fields))
    diff=fields.reshape(len(fields),1,2)-fields 
    D=np.sqrt((diff**2).sum(2))
    D=np.array(D,dtype=np.float64)
    D[np.triu_indices(D.shape[0])]=np.inf
    print(D.shape)
    re = np.where(D< radius)
    print(re)
    out=np.array(list(zip(re[0],re[1])),dtype=np.int32)
    print(out)
    outmin=np.unique(np.min(out,axis=1))
    return(outmin)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



# Using argparse to define the input and output directory.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="directory of the predicted labels")
ap.add_argument("-g", "--groundtruth",default='star/',required=False,help="directory in which the ground truth star file is stored")
ap.add_argument("-o", "--output",default='out/',help="directory in which the output is stored")
ap.add_argument("-t", "--iou_thresh",type=float,default=0.3,required=False,help="iou Threshold")
ap.add_argument("-r", "--radius",type=int,default=105, required=False,help="Radius of the particle")
ap.add_argument("-p", "--points",type=int,default=10, required=False,help="Number of points in pr curve")
ap.add_argument("-w", "--width",type=int,default=260, required=False,help="Width of bounding box")
ap.add_argument("-e", "--erod_iter",type=int,default=6, required=False,help="directory in which the output is stored")
ap.add_argument("-m", "--manual_radius",type=str2bool,default=False, required=False,help="Should we pick manually the radius of the particle")


#args = vars(ap.parse_args())
args = ap.parse_args()

input_dir=args.input
gt_dir=args.groundtruth
out_dir=args.output
erode_iter=args.erod_iter
iou_thresh=args.iou_thresh
rad=args.radius
pts=args.points
w=args.width
min_radius=1
max_radius=200





prec_final=[]
recall_final=[]

if args.manual_radius:
    j=0
    for fln in (os.listdir(input_dir)):
    
        frame=cv2.imread(os.path.join(input_dir,fln))
        
        
        frame[np.any(frame != [0, 0, 255], axis=-1)]=[0,0,0]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(gray_frame,0,255,cv2.THRESH_BINARY)
        canvas = np.zeros(thresh1.shape[:2],np.uint8)
        contrs, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thresh1 = cv2.drawContours(canvas, contrs, -1 ,(255),-1)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(11,11))
        #thresh1 = cv2.erode(thresh1, kernel,iterations=12)
        circle_frame=cv2.cvtColor(thresh1,cv2.COLOR_GRAY2BGR)
        circle_frame=frame.copy()
        #preprocessing the image. To exclude the boundary and to ignore the ice and carbon contamination predictions
        #print('frame shape',frame.shape)
    

    
        if j==0:
            contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cont_array=np.array([c for c in contours])
            c_ = np.array([cv2.contourArea(contour) for contour in contours])
            c_full_list=cont_array[(c_>200) & (c_<50000)]
         
            c_list=list(map(lambda x: min_circle(x),c_full_list))
            c_list=[x for x in c_list if x is not None]  
      
            source_window='adjust_radius'
            cv2.namedWindow(source_window,cv2.WINDOW_NORMAL)
            cv2.createTrackbar('Adjust the radius to enclose the particle: ', source_window, min_radius, max_radius,radius_callback)
            radius_callback(min_radius)
        j+=1
    radius=manual_picked_radius
else:
    radius=rad



print("\n***** PR Curve Generation *****")
print("SS Images -->", args.input)
print("GT Folder -->", args.groundtruth)
print("output Folder -->", args.output)
print("Radius -->", radius)
print("Erode Iteration -->", erode_iter)
print("Points in PR Curve -->", pts)
print("Width of bounding box-->", w)
print("IOU Threshold-->", iou_thresh)


cont_range=np.linspace(2,(3.14*radius*radius)/2.4,pts)
#cont_range=[2]
for cont_thresh in cont_range:

    precision_list=[]
    recall_list=[]
    actual_list=[]
    pred_list=[]
    TP_sum=0
    FP_sum=0
    FN_sum=0
    f1_sum=0
    f2_sum=0
    precision_sum =0
    recall_sum =0
    mean_iou_sum =0
    total_grond=0
    images_count=0
    for fln in (os.listdir(input_dir)):
    
        frame=cv2.imread(os.path.join(input_dir,fln))
        circle_frame=frame.copy()
        #preprocessing the image. To exclude the boundary and to ignore the ice and carbon contamination predictions
        #print('frame shape',frame.shape)
    
    
        frame[np.any(frame != [0, 0, 255], axis=-1)]=[0,0,0]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(gray_frame,0,255,cv2.THRESH_BINARY)
        canvas = np.zeros(thresh1.shape[:2],np.uint8)
        contrs, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thresh1 = cv2.drawContours(canvas, contrs, -1 ,(255),-1)

        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
        kernel=np.ones((5,5),np.uint8)
        thresh1 = cv2.erode(thresh1, kernel,iterations=erode_iter)

        
        x=thresh1.shape[1]
        y=thresh1.shape[0]
        thresh1[0:(int(radius/2)+2),0:x][:]=[0]
        thresh1[0:y,0:(int(radius/2)+2)][:]=[0]
        thresh1[(y-(int(radius/2)+2)):y,0:x][:]=[0]
        thresh1[0:y,((x-int(radius/2))+2):x][:]=[0]
        contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cont_array=np.array([c for c in contours])
        c_ = np.array([cv2.contourArea(contour) for contour in contours])
        c_full_list=cont_array[(c_>(cont_thresh)) & (c_<500000)]
        c_list=(list(map(lambda x: min_rect_circle(x),c_full_list)))
        cass_tuples=[x for x in c_list if x is not None]
        cass_all=cass_tuples.copy()
        gt_df=pd.read_csv(os.path.join(gt_dir,fln.replace('.png','.star')),skiprows=9,header=None,sep='\t')
        gt_cord=gt_df[[0,1]]
        gt_tuples = [tuple(x) for x in gt_cord.values]
        
        pred_found_index=[]
        ground_matching_index=[]
        len_cry_tuples=0
        len_pred_tuples=0
        MEAN_IOU=0
        TruePos=0
    
    
        cass_bbox = list(map(getbox_from_point, cass_tuples))
        cry_bbox = list(map(getbox_from_point, gt_tuples))
        
        len_cry_tuples=len_cry_tuples+len(gt_cord)
        len_pred_tuples=len_pred_tuples+len(cass_bbox)
        for i,boxes in enumerate(cry_bbox):
            actual_list.append(1)
            bbox=np.array([boxes]*len(cass_bbox))
            iou=list(map(bb_intersection_over_union,bbox,cass_bbox))
            if(iou):
                max_iou=np.max(iou)
            else:
                pred_list.append(0)
                continue
            if max(iou)>iou_thresh*0.1:

                pred_found_index.append(np.argmax(iou))
                ground_matching_index.append(i)
                
                del cass_bbox[np.argmax(iou)]
                del cass_tuples[np.argmax(iou)]
                TruePos=TruePos+1
                MEAN_IOU += max_iou
                total_grond=total_grond+1
                pred_list.append(1)
            else:
                pred_list.append(0)
        if len(gt_tuples) > 0:
                FalseNeg = len_cry_tuples - len(pred_found_index)
                #print(FN,len_cry_tuples,len(pred_found_index))
                FalsePos = len_pred_tuples - len(ground_matching_index)


                if (TruePos + FalsePos) == 0:
                    precision = 0
                else:
                    precision = 1.0 * TruePos / (TruePos + FalsePos)

                recall = 1.0 * TruePos / (TruePos + FalseNeg)
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
                mean_iou_sum += MEAN_IOU / (TruePos + 0.001)
                images_count = images_count + 1
                TP_sum += TruePos
                FP_sum += FalsePos
                FN_sum += FalseNeg
        list_2 = [i for n, i in enumerate(gt_tuples) if n not in ground_matching_index]
        list_3 = [i for n, i in enumerate(cass_all) if n not in cass_tuples]
        thresh1=cv2.cvtColor(thresh1,cv2.COLOR_GRAY2BGR)
        for line in list_2:
            cv2.circle(thresh1, (int(line[0]),int(line[1])), radius, (0, 255, 0), 3)
            cv2.circle(frame, (int(line[0]),int(line[1])), radius, (0, 255, 0), 3)
        #for line in list_3 :
        #    cv2.circle(thresh1, (int(line[0]),int(line[1])), radius, ( 255, 0,255), 3)
        #    cv2.circle(frame, (int(line[0]),int(line[1])), radius, ( 255, 0,255), 3)
        cv2.imwrite(os.path.join(out_dir,fln),thresh1)            
        cv2.imwrite(os.path.join(out_dir,fln.replace('.png','fr.png')),frame)            
                

    f1_avg = f1_sum / images_with_particles
    f2_avg = f2_sum / images_with_particles
    precision_avg = precision_sum / images_count
    recall_avg = recall_sum / images_count
    mean_iou_sum = mean_iou_sum / images_count 


    total_precision=1.0 * TP_sum / (TP_sum + FP_sum)   
    total_recall=1.0 * TP_sum / (TP_sum + FN_sum)        
    with open('results.txt', "a") as boxfile:                    
        boxwriter = csv.writer(boxfile, delimiter="\t", quotechar="|", quoting=csv.QUOTE_NONE)
        #boxwriter.writerow([fln,iou_val])
        boxwriter.writerow(['TP:',TP_sum,'FP:',FP_sum,'FN:',FN_sum,'Precision:',total_precision,'Recall:',total_recall])
    print('cont',cont_thresh,'TP:',TP_sum,'FP:',FP_sum,'FN:',FN_sum,'Precision:',total_precision,'Recall:',total_recall)            
    if not prec_final:
        prec_final.append(total_precision)
        recall_final.append(total_recall)   
    else:
        if total_precision>prec_final[-1]:
            prec_final.append(total_precision)
            recall_final.append(total_recall)   
        else:
            continue
print(prec_final)
print(recall_final)
    
# AUC Calculation
precision_array = np.array(prec_final)
recall_array = np.array(recall_final)

    # Interpolate first point
recall_array = np.insert(recall_array, len(recall_array), 0)
precision_array = np.insert(
        precision_array, len(precision_array), precision_array[len(precision_array) - 1]
    )

    # Sort
sorted_index = np.argsort(recall_array)[::-1]
precision_array = precision_array[sorted_index]
recall_array = recall_array[sorted_index]

area = 0
for i in range(1, len(recall_array)):
    val = (recall_array[i - 1] - recall_array[i]) * precision_array[i]
    area = area + val

print('auc',area)
#pyplot.plot(recall_array, precision_array, marker='.', label='Logistic')
# axis labels
#pyplot.xlabel('Recall')
#pyplot.ylabel('Precision')
# show the legend
#pyplot.legend()
# show the plot
#pyplot.show()    
sp=time.time()

import time
timestr = time.strftime("%Y%m%d-%H%M%S")
np.savez(((args.input.split('/')[0])+timestr+'.npz'),area,recall_array,precision_array)

print((sp-st)/60,'minutes')    	



