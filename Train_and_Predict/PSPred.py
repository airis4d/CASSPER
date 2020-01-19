from ProtiSem import *
import os,shutil,sys
argv=sys.argv[1:5]
if argv==[]:
	print("---------------------------------------------------------------------------------------------------------------------------------------------")
	print("You can call this function to create prediction png images of mrc files")
	print("The parameters are:")
	print("Destination Folder for created images as first parameter (default is current directory) ")
	print("The location where the mrc files are kept - the default is same as the destination folder/mrc/")
	print("Author: Ninan Sajeeth Philip - sajeethphilip2gmail.com")
	print("---------------------------------------------------------------------------------------------------------------------------------------------")
	print("Command is "+sys.argv[0]+" Destination_for_predict_files folder  Folder_of_mrc_files")
	print("---------------------------------------------------------------------------------------------------------------------------------------------")
	print("")
	des="./"
	mrcpth="./mrc/"
else:
	des=argv[0]
	mrcpth=des+"mrc/"
	if len(argv)>1:
		mrcpth=argv[1]
try:
	os.mkdir( des+"Predict/" )
except:
	print("")


def write_heatMap(img,outfl,dest,owidth):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(img)
    rgb_img = np.delete(rgba_img, 3, 2)
    #print(rgb_img.shape)
    img=Image.fromarray(rgb_img, 'RGB')
    img=Resize_imgAspect(img,owidth)
    img.save(dest+outfl)
def nmzImg(img):
	mx_img=np.max(img)
	mn_img=np.min(img)
	return (254*((img-mn_img)/(mx_img-mn_img))).astype(np.uint8 )

mrc=(os.listdir(mrcpth))
cont=np.size(mrc)
cnt=1
dest=des+"Predict/"
print("Creating Prediction data from "+str(cont)+" mrc files in folder "+mrcpth+" to -> "+dest)
for fln in (mrc):
	#fln='HCN1apo_0002_2xaligned.mrc'
	fltyp=os.path.splitext(fln)[1]
	fl=os.path.splitext(fln)[0]
	#fl=".".join(fl)
	if (fltyp==".jpg"):
		im=Image.open(des+"labels/"+fln)
		rgb_im=np.asarray(im.convert('RGB'))
		os.remove(des+"labels/"+fln)
		rgb_im[rgb_im>=170]=255
		rgb_im[rgb_im<170]=0
		print(np.unique(rgb_im))

		fln=fl+'.png'
		im= Image.fromarray(rgb_im)
		im.save(des+"labels/"+fln)
		im.close()

	try:
		img=read_anymrcfile(mrcpth+fln.replace(".png",".mrc"))   # We are looking for mrc files in the folder
	except:
		print('Corrupt mrc, ignoring')
		continue
	save_imageSz(np.asarray(img),fl)
	#img=Resize_img(Image.fromarray(img),w2=1024,h2=1024)	#save_mximg(img,filename="fig1.png")
	#img=np.uint8(img)
	img=np.copy(cv2.equalizeHist(img))
	#imf=ndimage.gaussian_filter(img,3)
	#img=np.uint8(imf)
	#img=np.copy(cv2.bilateralFilter(img,d=13,sigmaColor=5, sigmaSpace=5))
	d=np.max((np.shape(img)))
	d=d//640
	img=np.copy(cv2.bilateralFilter(img,d=d,sigmaColor=12, sigmaSpace=12))
	img=nmzImg(img)
	#clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(30,30))
	#img = clahe.apply(img)
	#img=np.copy(cv2.equalizeHist(img))
	save_mximg(img,filename="fig3.png")
	if np.max(img)>0:
		# Contrast stretching
		imag=CutMedianTH(Image.fromarray(img))
		imag=nmzImg(imag)
		imgx=np.asarray(imag)
		imgx=nmzImg(imgx)#save_mximg(imgx,filename="fig4.png")
		cx=np.std(imgx)
		imgx=cv2.adaptiveThreshold(imgx,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,			cv2.THRESH_BINARY,255,-0.1*cx)
		output_width=512  # For training image is compressed to 512x512
		#output_height=np.shape(imgx)[0]
		write_img3(in1=np.asarray(imgx),in2=np.asarray(imag),in3=np.asarray(img),outfl=fl+".png",dest=dest+"/",owidth=output_width)
		#write_heatMap(imgx,outfl=fl+".png",dest=dest+"/",owidth=output_width)

