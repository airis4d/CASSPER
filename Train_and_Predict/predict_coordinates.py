import os,time,cv2, sys, math
import tensorflow as tf
import argparse,csv
import numpy as np
import CASS_scale as sw
from scipy import ndimage
from functools import partial
import itertools
from utils import utils, helpers
from builders import model_builder
import time
from ProtiSem import *

import os, glob, math, cv2, time
import numpy as np
from joblib import Parallel, delayed
nprocs =6 


def get_shape(img):
    while True:
        try:
           w,h,l=np.shape(np.asarray(img))
           break
        except:
           print(np.shape(np.asarray(img)))
           print("Failed to read shape trying again")
           pass                           #Sometims they do not read shape correctly in first attempt!
    return(w,h,l)



def nmzImg(img):
	mx_img=np.max(img)
	mn_img=np.min(img)
	return (254*((img-mn_img)/(mx_img-mn_img))).astype(np.uint8 )



def normalize(im):
    max_mrc=np.max(im)
    min_mrc=np.min(im)
    img_original=(254*((im-min_mrc)/(max_mrc-min_mrc))).astype(np.uint8 )
    return(img_original)
    

def save_img3(in1,in2,in3,owidth=512,oheight=512):
    w,h=np.shape(in1)
    rgbA=np.zeros((w,h,3), 'uint8')
    rgbA[..., 0]=np.copy(in1)
    rgbA[..., 1]=np.copy(in2)
    rgbA[..., 2]=np.copy(in3)
    rgbA=apply_CLAHE(rgbA,gzw=8,gzh=8,cl=0)
    Watershed(rgbA)
    img=Image.fromarray(rgbA)
    return np.array(Resize_img(img,w2=owidth,h2=owidth))
    #img=AdptTh(img)
    #img.save(dest+outfl)
    #img.close()
    

def Resize_img(image,w2=0,h2=0):
    #w,h,l=get_shape(image)
    #if w2==0:
    #    w2=w
    #if h2==0:
    #    h2=h
    imResize=image.resize((w2,h2), Image.NEAREST) #ANTIALIAS)
    return (np.array(imResize))

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
    
    c_frame=output_image[i1:i3,i2:i4]
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
        
        #cv2.circle(frame, i, radius, (255, 0, 0), 3)
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







def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser=argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None, required=False, help='The image you want to predict on. ')
parser.add_argument('--checkpoint_path', type=str, default=None, required=False, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--mrc', type=str, default="mrc_files", required=False, help='The path to the mrcfiles.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default="FRRN-B", required=False, help='The model you are using')
parser.add_argument('--dataset', type=str, default="Protein1", required=False, help='The dataset you are using')
parser.add_argument('--eliminate', type=str2bool, default=False, help='Whether eliminate near particles?')
parser.add_argument('--sorting', type=str2bool, default=False, help='Whether primary sorting has to done?')
parser.add_argument('--rads', type=int, default=60, help='Radius of the particles to be selected')
parser.add_argument('--erode', type=int, default=0, help='No. of erode iterations to be done to disconnect joint particles if any')
parser.add_argument('--threshold_val', type=int, default=None, required=False,help='Threshold value of minimum contour area')



args=parser.parse_args()

class_names_list, label_values=helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
num_classes=len(label_values)

if args.checkpoint_pth is not None:
    chechkpoint_pth=args.chechkpoint_pth
else:
    checkpoint_path=os.path.join(args.dataset,"TSaved/BestFr_InceptionV4_model_FRRN-B_F1.ckpt")
 
radius=args.rads    
    
if args.threshold_val is not None:
    contr_min=args.threshold_val
else:
    contr_min=radius**1.4

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Radius -->", args.radius)
print("Erode Iteration -->", args.erode)
print("Num Classes -->", num_classes)
print("Image -->", args.image)
print("Chekpoint -->",checkpoint_path)

def normalize(im):
    max_mrc=np.max(im)
    min_mrc=np.min(im)
    img_original=(254*((im-min_mrc)/(max_mrc-min_mrc))).astype(np.uint8 )
    return(img_original)



st=time.time()

# Initializing network
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

net_input=tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output=tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network, _=model_builder.build_model(args.model, net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=args.crop_width,
                                        crop_height=args.crop_height,
                                        is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, checkpoint_path)
des=args.dataset
radius=args.rads
mrcpth=args.mrc


try:
	shutil.rmtree(des+"Results")
except:
	pass

Star_dir = os.path.join(des,"Star")
Box_dir = os.path.join(des,"Box")
if not os.path.exists(Star_dir):
    os.mkdir(Star_dir)
if not os.path.exists(Box_dir):
    os.mkdir(Box_dir)


def process_image(flname):
	im=read_anymrcfile(flname) 
	fln=flname.split('/')[-1]
	fl=os.path.splitext(fln)[0]
	origw,origh=np.shape(im)

	img=Resize_img(Image.fromarray(im),w2=int((np.min(im.shape))/2),h2=int((np.min(im.shape))/2))
	img=np.array(img)	
	img=np.copy(cv2.equalizeHist(img))
	d=np.max((np.shape(img)))
	d=d//640
	img=np.copy(cv2.bilateralFilter(img,d=d,sigmaColor=12, sigmaSpace=12))
	img=nmzImg(img)
	if np.max(img)>0:
	# Contrast stretching
		imag=CutMedianTH(Image.fromarray(img))
		imag=nmzImg(imag)
		imgx=np.asarray(imag)
		imgx=nmzImg(imgx)#save_mximg(imgx,filename="fig4.png")
		cx=np.std(imgx)
		imgx=cv2.adaptiveThreshold(imgx,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,			cv2.THRESH_BINARY,255,-0.1*cx)
		output_width=512  # For training image is compressed to 512x512

		np.savez(os.path.join(des+"Result/",fln.replace(".mrc",".npz")),(save_img3(in1=np.asarray(imgx),in2=np.asarray(imag),in3=np.asarray(img),owidth=output_width)),origw,origh)

		return(fln)
	
if (args.image==""):

        path = os.path.join(mrcpth,'*.mrc')
        files=glob.glob(path)
        X_train=[]
        X_train.extend(Parallel(n_jobs=nprocs)(delayed(process_image)(im_file) for im_file in files))



nparr=(os.listdir(des+"Result/"))
for fln in nparr:
	print(nparr,fln)
	arry=np.load(des+"Result/"+fln)
	resized_image,origw,origh=arry['arr_0'],arry['arr_2'],arry['arr_2']
	input_image=np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0
	output_image=sess.run(network,feed_dict={net_input:input_image})
	output_image=np.array(output_image[0,:,:,:])
	output_image=helpers.reverse_one_hot(output_image)
	output_image=np.where(output_image!=0,50,output_image)
	output_image=np.where(output_image==0,1,output_image)
	output_image=np.where(output_image==50,0,output_image)
	output_image=normalize(output_image)
	output_image=Resize_img(Image.fromarray(output_image),origh,origw)
	ret,thresh1 = cv2.threshold(output_image,0,255,cv2.THRESH_BINARY)
	x=thresh1.shape[1]
	y=thresh1.shape[0]
	thresh1[0:radius+2,0:x][:]=[0]
	thresh1[0:y,0:radius+2][:]=[0]
	thresh1[(y-radius+2):y,0:x][:]=[0]
	thresh1[0:y,(x-radius+2):x][:]=[0]
	contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cont_array=np.array([c for c in contours])
	c_ = np.array([cv2.contourArea(contour) for contour in contours])
	c_full_list=cont_array[(c_>contr_min) & (c_<500000)]
	c_list=(list(map(lambda x: min_rect_circle(x),c_full_list)))
	c_list=[x for x in c_list if x is not None]
	if args.sorting:
		final_list=(list(map(lambda x: primary_sorting(x),c_list)))
		final_list=[x for x in final_list if x is not None]  
	else:
		final_list=[x for x in c_list if x is not None]  
	if args.eliminate:
		out_list=eliminate_near(final_list)
		for ele in sorted(out_list, reverse = True):  
			del final_list[ele]
	
	#final_list=[x for x in c_list if x is not None]  
	with open(os.path.join(des+"Star/",fln.replace('.npz','.star')), "w") as starfile:
		starwriter = csv.writer(
		starfile, delimiter="\t", quotechar="|", quoting=csv.QUOTE_NONE)
		starwriter.writerow([])
		starwriter.writerow(["data_"])
		starwriter.writerow([])
		starwriter.writerow(["loop_"])
		starwriter.writerow(["_rlnCoordinateX #1 "])
		starwriter.writerow(["_rlnCoordinateY #2"])
		starwriter.writerow(["_rlnClassNumber #3"])
		starwriter.writerow(["_rlnAnglePsi #4"])
		starwriter.writerow(["_rlnAutopickFigureOfMerit  #5"])
		for line in final_list:
			starwriter.writerow(['{0:.6f}'.format(line[0]),'{0:.6f}'.format(line[1]), -999, '{0:.6f}'.format(-999), '{0:.6f}'.format(-999)])
	with open(os.path.join(des+"Box/",fln.replace('.npz','.box')), "w") as boxfile:
		boxwriter = csv.writer(
		boxfile, delimiter="\t", quotechar="|", quoting=csv.QUOTE_NONE)
		for line in final_list:
			boxwriter.writerow([line[0]-radius ,line[1]-radius,radius,radius])
shutil.rmtree(des+"Results")
