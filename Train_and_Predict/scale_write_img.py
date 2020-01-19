from PIL import Image
import os,shutil,sys,cv2,pickle
import numpy as np

def Resize_img(image,w2,h2):
    imResize=image.resize((w2,h2), Image.NEAREST) #ANTIALIAS)
    return  (imResize)



def scale_write_img(filename="",pth="./",des="",src=""):
		im=((Image.open(src)))
		j,k=pickle.load(open("./ProtiSEM_metadata/"+filename+'_OrgSz.p', 'rb'))

		#print(j,k)
		#im=cv2.resize(np.asarray(im),(k,j))
		#-----------------------------
		#im=Image.fromarray(im)
		im=Resize_img(im,w2=k,h2=j)
		im.save(pth+des+filename+".png")
		im.close()


