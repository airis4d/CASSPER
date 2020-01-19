from PIL import Image,ImageEnhance
import numpy as np
import cv2,os,argparse
import mrcfile
from scipy import ndimage

def concat_display(im1,im2):
    img_corrected = cv2.hconcat([im1, im2])
    imgd=np.zeros((int((image.shape[0])/7),(image.shape[1])*2),np.uint8)
    imgd=cv2.cvtColor(imgd,cv2.COLOR_GRAY2BGR)
    imgd[:]=[255,255,255]
    img_disp=cv2.vconcat([imgd, img_corrected])

    cv2.putText(img_disp, " After adjusting the slidebar, press 'f' key to SAVE the current label ", (80, 100 ), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,0),5)
    cv2.putText(img_disp, " If carbon contamination is to be marked, press 'c' key", (80, 200 ), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,0),5)
    cv2.putText(img_disp, " To label ice, press 'i' key.", (80, 300 ), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,0),5)
    cv2.putText(img_disp, " Press Esc and space key while coloring ice or carbon contamination", (80, 400), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,0),5)
    cv2.putText(img_disp, " To ignore the current image, press the 'q' key.", (80, 500 ), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,0),5)
    cv2.namedWindow(file1,cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(file1,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow(file1,img_disp)


def other_part(name,points,img):


    window_image = name
    cv2.namedWindow(window_image, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_image,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_image, img)


    def next_pickPoint(event, x, y, flags, param):
        seed = None

        if event != cv2.EVENT_LBUTTONDOWN | event != cv2.EVENT_RBUTTONDOWN:
            return

        if flags & cv2.EVENT_FLAG_LBUTTON:
            seed=(x,y)
            points.append(seed)

    cv2.setMouseCallback(window_image, next_pickPoint)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def color_ice(image,target):

    def callback_low(val):
        pass
    
    fln=file1
    im = image.copy()
    imtarget= target
    imPIL=Image.fromarray(imtarget)  
    imPILTAR=Image.fromarray(imtarget) 
    cv2.namedWindow(fln, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(fln,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    r=[]
    r = cv2.selectROIs(fln,im)

    for i in range(0,len(r)):    
        x=np.array(r[i])
        imCrop = im[int(x[1]):int(x[1]+x[3]), int(x[0]):int(x[0]+x[2])]
        imcroptar=imtarget[int(x[1]):int(x[1]+x[3]), int(x[0]):int(x[0]+x[2])]
        imCrop2=Image.fromarray(imCrop)

        contrast=ImageEnhance.Contrast(imCrop2)
        contr=contrast.enhance(6)
        contr=np.array(contr)
        contr=cv2.cvtColor(contr,cv2.COLOR_RGB2GRAY)

        blur = np.invert(cv2.bilateralFilter(contr,12,140,140))

        window_image = 'Source image'
        cv2.namedWindow(window_image,cv2.WINDOW_NORMAL)
        cv2.createTrackbar('thresh', window_image, 0,300, callback_low)

        while(1):
            blur1=blur.copy()
            imcroptar2=imcroptar.copy()
            imCrop2=imCrop.copy()
            ctr=cv2.getTrackbarPos('thresh', window_image)
            ret,blur2 = cv2.threshold(blur1,ctr,255,cv2.THRESH_BINARY)#+cv2.THRESH_OTSU)
            cv2.namedWindow(window_image, cv2.WINDOW_NORMAL)

            cv2.imshow(window_image,blur2)


            contours, hierarchy = cv2.findContours(blur2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            try:
                c=max(contours,key=cv2.contourArea)
            except:
                pass
            cv2.drawContours(imcroptar2,[c],0,(0,255,255),-1)
            #for c in contours:
            #    if cv2.contourArea(c)>5000:
            #        cv2.drawContours(imCrop2,[c],0,(0,255,255),1)
            k = cv2.waitKey(1) & 0xFF      
            if k==27: 
                cv2.destroyWindow(window_image)
                break

        imCropPILtar=Image.fromarray(imcroptar2)

        imPILTAR.paste(imCropPILtar,((int(x[0]),int(x[1]), int(x[0]+x[2]),int(x[1]+x[3]))))
    cv2.destroyAllWindows()
    return np.uint8(imPILTAR) 



def contr_enhance(im,ctr):
    contr=Image.fromarray(im)
    contr=contr.convert('RGB')
    contrast=ImageEnhance.Contrast(contr)
    contr=contrast.enhance(ctr*.1)
    return(np.array(contr)) 

def callback_1(val):
    global s_min
    s_min=val
    image_man()

def callback_2(val):
    global c_min
    c_min=val
    image_man()


def callback_3(val):
    global t_min
    t_min=val
    image_man()

def callback_4(val):
    global contr_min
    contr_min=val
    image_man()


def image_man():
    
    global img_corrected, contr_enh,image
    image=np.zeros_like(img_original)
    image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    image[:]=[255,0,0]
    
    #im1=img_original.copy()
    clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(16,16))
    clahe_equalized = clahe.apply(img_original)
    blurred= (cv2.bilateralFilter(clahe_equalized,s_min,350,350))

    contr_enh=contr_enhance(blurred,c_min)

    contr_gray=cv2.cvtColor(contr_enh,cv2.COLOR_BGR2GRAY)
    inv_img=np.invert(contr_gray)

    ret,th1 = cv2.threshold(inv_img,t_min,255,cv2.THRESH_BINARY)

    kernel=np.ones((2,2),np.uint8)
    th1=cv2.dilate(th1, kernel,iterations=1)

    contours,_ = cv2.findContours(th1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:

        if cv2.contourArea(c)>contr_min and cv2.contourArea(c)<80000:

            cv2.drawContours(contr_enh,[c],0,(0,0,255),2) 
            cv2.drawContours(image,[c],0,(0,0,255),-1)   


    
    img_corrected = cv2.hconcat([ contr_enh,image])
    white=np.zeros((int((image.shape[0])/7),(image.shape[1])*2),np.uint8)

    white=cv2.cvtColor(white,cv2.COLOR_GRAY2BGR)
    white[:]=[255,255,255]
    img_disp=cv2.vconcat([white, img_corrected])
    cv2.putText(img_disp, " After adjusting the slidebar, press 'f' key to SAVE the current label ", (80, 100 ), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,0),5)
    cv2.putText(img_disp, " If carbon contamination is to be marked, press 'c' key", (80, 200 ), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,0),5)
    cv2.putText(img_disp, " To label ice, press 'i' key.", (80, 300 ), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,0),5)
    cv2.putText(img_disp, " Press Esc and space key while coloring ice or carbon contamination", (80, 400), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,0),5)
    cv2.putText(img_disp, " To ignore the current image, press the 'q' key.", (80, 500 ), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,0),5)
    cv2.namedWindow(fname,cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(fname,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow(fname, img_disp)

s_min=12
s_max=50
c_min=12
c_max=20
t_min=200
t_max=255
contr_min=2000
contr_max=8000



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,help="directory of the input mrc")
ap.add_argument("-o", "--output", required=True,\
help="directory to which the output label is to be stored")
args = vars(ap.parse_args())


input_dir=args['input']
out_dir=args['output']   

print("***************************************************************************************************************************")
print("1. After adjusting the slidebar, press 'f' key to SAVE the current label. ")
print("2. If carbon contamination is to be marked, press 'c' key. ") 
print("3. To ignore the current image, press the 'q' key. ") 
print("4. To label ice, press 'i' key.")
print("Press Esc and space key while coloring ice or carbon contamination")
print("___________________________________________________________________________________________________________________________")



for fln in sorted(os.listdir(input_dir)):
    print(fln)
    if not fln.endswith(".mrc"):
    #if fln.rpartition('.')[-1]!='mrc' or os.path.isdir(os.path.join(input_dir,fln)):
        print(fln,"is not an mrc file")
        
        continue
    fname=fln.replace('.mrc','')    
    with mrcfile.open(os.path.join(input_dir,fln),permissive=True) as mrc:
        im=mrc.data
    	
    max_mrc=np.max(im)
    min_mrc=np.min(im)
    img_original=(254*((im-min_mrc)/(max_mrc-min_mrc))).astype(np.uint8 )
    



    img_original=ndimage.gaussian_filter(img_original,8)
    #img_original=cv2.resize(img_original,(3500,3500))
    cv2.namedWindow(fname,cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(fname,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.createTrackbar("blur sigma",fname,s_min,s_max,callback_1)
    cv2.createTrackbar("contrast",fname,c_min,c_max,callback_2)
    cv2.createTrackbar("threshold",fname,t_min,t_max,callback_3)
    cv2.createTrackbar("contour_area",fname,contr_min,contr_max,callback_4)

    callback_1(s_min)
    
    while(1):
        
        file1=fname+"_enhanced"
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            cv2.destroyAllWindows()
            cont_points=[]

            other_part(file1,cont_points,contr_enh)
            list1=(np.array((cont_points[0:4]),np.int32)).reshape((-1,1,2))
            list2=(np.array((cont_points[4:8]),np.int32)).reshape((-1,1,2))
            image=cv2.fillConvexPoly(np.array(image),list1,(0,255,0))
            image=cv2.fillConvexPoly(np.array(image),list2,(0,255,0))
            concat_display(contr_enh,image)

            continue

        if k == ord('i'):
            
            cv2.destroyAllWindows()
            image=color_ice(contr_enh,image)
            
            concat_display(contr_enh,image)

            continue

        if k==ord('f'):
            if fln.startswith("Falcon"):
                image = np.array(image,np.uint8)
                image[0:4096, 0:126][:]=[255,0,255]
                image[0:126, 0:4096][:]=[255,0,255]
                image[0:4096, 3964:4096][:]=[255,0,255]
                image[3964:4096, 0:4096][:]=[255,0,255]
            
            name1=fln.replace('.mrc','cont.png')
            name2=fln.replace('.mrc','.png')
            cv2.imwrite(os.path.join(out_dir,name1),contr_enh)
            cv2.imwrite(os.path.join(out_dir,name2),image)
            break
        if k==ord('q'):
            break



    cv2.destroyAllWindows()
