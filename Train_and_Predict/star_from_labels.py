from PIL import Image,ImageEnhance
import numpy as np,csv
import cv2,os,argparse
import mrcfile,time
from scipy import ndimage
from functools import partial
import itertools

st=time.time()

# Program to get the radius of the particle by adjusting the track bar
def radius_circle():
    c_frame=circle_frame.copy()
    for i in c_list:
        cv2.circle(c_frame, tuple(i), int(radius), (0, 255, 0), 3)
    cv2.namedWindow(source_window,cv2.WINDOW_NORMAL)
    cv2.imshow( source_window, c_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Call back function for trakbar
def radius_callback(val):
    global radius
    radius =val
    radius_circle()



def primary_sorting(i):
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
    if np.all(diff<(2*radius+20)):
        return(int(center[0]),int(center[1]))
    else:
        pass


def eliminate_near(fields):
    fields=np.array(fields,dtype=np.int32)
    i=np.arange(len(fields))
    diff=fields.reshape(len(fields),1,2)-fields 
    D=np.sqrt((diff**2).sum(2))
    D=np.array(D,dtype=np.float64)
    D[np.triu_indices(D.shape[0])]=np.inf
    re = np.where(D< radius)  
    out=np.array(list(zip(re[0],re[1])),dtype=np.int32)
    outmin=np.unique(np.min(out,axis=1))
    return(outmin)

# Using argparse to define the input and output directory.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,help="directory of the predicted labels")
ap.add_argument("-o", "--output", required=True,help="directory in which the output png is stored")
args = vars(ap.parse_args())

input_dir=args['input']
out_dir=args['output']

j=0
min_radius=1
max_radius=200

for fln in (os.listdir(input_dir)):
    cool_list=[]
    mess_list=[]

    frame=cv2.imread(os.path.join(input_dir,fln))
    circle_frame=frame.copy()
    #preprocessing the image. To exclude the boundary and to ignore the ice and carbon contamination predictions
    #print('frame shape',frame.shape)
    
    
    frame[np.any(frame != [0, 0, 255], axis=-1)]=[0,0,0]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray_frame,0,255,cv2.THRESH_BINARY)
    
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
    x=thresh1.shape[1]
    y=thresh1.shape[0]
    thresh1[0:radius+2,0:x][:]=[0]
    thresh1[0:y,0:radius+2][:]=[0]
    thresh1[(y-radius+2):y,0:x][:]=[0]
    thresh1[0:y,(x-radius+2):x][:]=[0]
    contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cont_array=np.array([c for c in contours])
    c_ = np.array([cv2.contourArea(contour) for contour in contours])
    c_full_list=cont_array[(c_>(radius**1.4)) & (c_<50000)]
    
    c_list=(list(map(lambda x: min_rect_circle(x),c_full_list)))
    c_list=[x for x in c_list if x is not None]
    if radius<100:
        final_list=(list(map(lambda x: primary_sorting(x),c_list)))
        final_list=[x for x in final_list if x is not None]  
    else:
        final_list=c_list
    bef_len=len(final_list)
    out_list=eliminate_near(final_list)
    for ele in sorted(out_list, reverse = True):  
        del final_list[ele] 
    with open(os.path.join(out_dir,fln.replace('.png','.star')), "w") as boxfile:
        boxwriter = csv.writer(
        boxfile, delimiter="\t", quotechar="|", quoting=csv.QUOTE_NONE
        )
        boxwriter.writerow([])
        boxwriter.writerow(["data_"])
        boxwriter.writerow([])
        boxwriter.writerow(["loop_"])
        boxwriter.writerow(["_rlnCoordinateX #1 "])
        boxwriter.writerow(["_rlnCoordinateY #2"])
        boxwriter.writerow(["_rlnClassNumber #3"])
        boxwriter.writerow(["_rlnAnglePsi #4"])
        boxwriter.writerow(["_rlnAutopickFigureOfMerit  #5"])
        for line in final_list:
            boxwriter.writerow(['{0:.6f}'.format(line[0]),'{0:.6f}'.format(line[1]), -999, '{0:.6f}'.format(-999), '{0:.6f}'.format(-999)])
            cv2.circle(frame, line, radius, (0, 255, 0), 3)
    
    print(fln,bef_len,len(out_list),len(final_list),'completed')
sp=time.time()
print((sp-st)/60,'minutes')

