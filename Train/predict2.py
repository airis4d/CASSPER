import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import scale_write_img as sw

from utils import utils, helpers
from builders import model_builder

parser=argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None, required=False, help='The image you want to predict on. ')
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
args=parser.parse_args()

class_names_list, label_values=helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes=len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Image -->", args.image)

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
saver.restore(sess, args.checkpoint_path)
des=args.dataset
try:
	os.mkdir( des+"P_files/" )
except:
	print("Folder P_files/ already exist. Files will be appended to the folder")
st=time.time()

if (args.image==""):
	files=(os.listdir(des+"Predict/"))
	cont=np.size(files)
	print("Preparing for testing on "+str(np.int(cont))+"  files...")
	for fln in (files):
		dest=des+"P_files/"
		print("Testing image " + fln)

		loaded_image=utils.load_image(des+"Predict/"+fln)
		resized_image=cv2.resize(loaded_image, (args.crop_width, args.crop_width))
		input_image=np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0
		#input_image=np.expand_dims(np.float32(utils.load_image(val_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0

		output_image=sess.run(network,feed_dict={net_input:input_image})


		output_image=np.array(output_image[0,:,:,:])
		output_image=helpers.reverse_one_hot(output_image)

		out_vis_image=helpers.colour_code_segmentation(output_image, label_values)
		file_name=dest+os.path.splitext(fln)[0]
		#file_name=".".join(fl)


		cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
		sw.scale_write_img(filename=os.path.splitext(fln)[0],pth=des,des="Predict_labels/",src="%s_pred.png"%(file_name))
else:
	print("Testing image " + args.image)

	loaded_image=utils.load_image(args.image)
	resized_image=cv2.resize(loaded_image, (args.crop_width, args.crop_width))
	input_image=np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

	output_image=sess.run(network,feed_dict={net_input:input_image})


	output_image=np.array(output_image[0,:,:,:])
	output_image=helpers.reverse_one_hot(output_image)

	out_vis_image=helpers.colour_code_segmentation(output_image, label_values)
	file_name=utils.filepath_to_name(args.image)
	cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
	sw.scale_write_img(filename="%s_fpred.png"%(file_name),src="%s_pred.png"%(file_name))

run_time=time.time()-st

print("")
print("Finished in %s seconds" %run_time)
print("--------Bye-----------")
