#!/bin/bash
echo "----------------------------------------------------------"
echo "This is the master script to run the particle picking code"
echo "The code is Free under creative public licence"
echo "Author: Ninan Sajeeth Philip, ninansajeethphilip@gmail.com"
echo "Usage:"
echo "1. For preparing and training Freshly on protein:"
echo "$0 Protein_Folder/ Fresh new Input_model Network_model"
echo "2. For Fresh training on already prepared protein:"
echo "$0 Protein_Folder/ Fresh old Input_model Network_model"
echo "3. For continued training on already trained protein:"
echo "$0 Protein_Folder/ Cont old Input_model Network_model"
echo "4. For testing on already trained protein:"
echo "$0 Protein_Folder/ Test "
echo "5. For predicting new mrc files in PFolder/Pfiles.txt"
echo "$0 PFolder/ Predict "
echo "5. To predict files in PFolder/Pfiles.txt with ckpt"
echo "$0 PFolder/ Predict ckpt "
echo "Enjoy!"
echo "-----------------------------------------------------------"
#Specify paths here
anacondapath=`which python` #"/usr/anaconda3/bin"

#anacondapath="/home/radha/anaconda3/bin"
#defaultpth="/home/radha/santhom/work/bin/python"
mrc_files="mrc_files/"
#------------------Code starts- edit at your own risk-----------------------------
fend="ResNet101"
if [ "$4" != "" ]; then
fend=$4
fi
model="FC-DenseNet56"
if [ "$5" != "" ]; then  #FRRN-B
model=$5
fi
if [ "$1" == "" ]; then
P_data="P_data"
else
P_data=$1
fi
mkdir -p $P_data/train_labels/
mkdir -p $P_data/Final_labels/
mkdir -p $P_data/labels/
mkdir -p $P_data/Master_files/train/
mkdir -p $P_data/Master_files/val/
mkdir -p $P_data/Master_files/test/
mkdir -p $P_data/Master_files/train_labels/
mkdir -p $P_data/Master_files/val_labels/
mkdir -p $P_data/Master_files/test_labels/
mkdir -p $P_data/val_labels/
mkdir -p $P_data/train/
mkdir -p $P_data/val/
mkdir -p $P_data/Test/
mkdir -p $P_data/TSaved/
mkdir -p $P_data/Predict/
mkdir -p $P_data/test/
mkdir -p $P_data/test_labels/
mkdir -p $P_data/P_files/
mkdir -p $P_data/Predict_labels
rm  -f $P_data/train_labels/*
rm  -f $P_data/val_labels/*
rm  -f $P_data/train/*
rm  -f $P_data/val/*
rm  -f $P_data/test_labels/*
rm  -f $P_data/test/*
rm  -f $P_data/Predict/*
rm -rf ./tmp/*
mkdir -p ./tmp/
mkdir -p ./ProtiSEM_metadata/
if [ "$2" == "Fresh" ]; then
rm -f $P_data/TSaved/*

if [ "$3" == "new" ]; then
\rm ./tmp/
mkdir -p ./tmp/labels
for i in `cat $P_data/Trfiles.txt`; do
cp $P_data/labels/$i ./tmp/labels; done

rm  -f $P_data/Master_files/train/*
rm  -f $P_data/Master_files/val/*
rm  -f $P_data/Master_files/train_labels/*
rm  -f $P_data/Master_files/val_labels/*
mkdir -p ./tmp/png0
mkdir -p ./tmp/png1
mkdir -p ./tmp/png2
$anacondapath PSTrain.py ./tmp/ $mrc_files ./tmp/labels/
rm -r -f ./tmp/labels
mv ./tmp/train/* $P_data/Master_files/train/
mv ./tmp/val/* $P_data/Master_files/val/
mv ./tmp/train_labels/* $P_data/Master_files/train_labels/
mv ./tmp/val_labels/* $P_data/Master_files/val_labels/

fi
cp $P_data/Master_files/train/* $P_data/train/
cp $P_data/Master_files/val/* $P_data/val/
cp $P_data/Master_files/train_labels/* $P_data/train_labels/
cp $P_data/Master_files/val_labels/*   $P_data/val_labels/

#python splitJoin.py $P_data Master_files/train/ train/ Split  # Ensure that mrc files are used for determining image dimensions.
#python splitJoin.py $P_data Master_files/val/ val/ Split
#python splitJoin.py $P_data Master_files/train_labels/ train_labels/ Split
#python splitJoin.py $P_data Master_files/val_labels/ val_labels/ Split
#for i in `ls $P_data/train_labels/`; do slice-image -d $P_data/train_labels/  -f "jpg" $P_data'/train_labels/'$i 8; rm $P_data'/train_labels/'$i; done
#for i in `ls $P_data/val_labels/`; do slice-image -d $P_data/val_labels/  -f "jpg" $P_data'/val_labels/'$i 8; rm $P_data'/val_labels/'$i; done
#for i in `ls $P_data/train/`; do slice-image -d $P_data/train/  -f "jpg" $P_data'/train/'$i 8;rm $P_data'/train/'$i; done
#for i in `ls $P_data/val/`; do slice-image -d $P_data/val/  -f "jpg" $P_data'/val/'$i 8;rm $P_data'/val/'$i; done

python train2.py --dataset $P_data --frontend $fend --checkpoint_step 10 --brightness 0.5  --model $model --num_epochs 100 --h_flip 1 --v_flip 1 --rotation 8   --batch_size 1 --num_val_images 25 --continue_training 0
#python train.py --dataset $P_data --crop_height 512 --crop_width 512 --model GCN --num_epochs 10 --h_flip 1 --v_flip 1 --rotation 8  --batch_size 1 --num_val_images 256 --continue_training 0
fi
if [ "$2" == "Cont" ]; then

cp $P_data/Master_files/train/* $P_data/train/
cp $P_data/Master_files/val/* $P_data/val/
cp $P_data/Master_files/train_labels/* $P_data/train_labels/
cp $P_data/Master_files/val_labels/*   $P_data/val_labels/

#python splitJoin.py $P_data Master_files/train/ train/ Split  # Ensure that mrc files are used for determining image dimensions.
#python splitJoin.py $P_data Master_files/val/ val/ Split
#python splitJoin.py $P_data Master_files/train_labels/ train_labels/ Split
#python splitJoin.py $P_data Master_files/val_labels/ val_labels/ Split

#python train.py --dataset $P_data --crop_height 512 --crop_width 512 --model RefineNet --num_epochs 100 --num_val_images 4
python train2.py --dataset $P_data --frontend $fend --epoch_start_i 100  --checkpoint_step 10  --brightness 0.5 --crop_height 512 --crop_width  512 --model $model --num_epochs 300 --h_flip 1 --v_flip 1 --rotation 8  --batch_size 1 --num_val_images 25 --continue_training 1
#python train.py --dataset $P_data --crop_height 512 --crop_width 512 --model GCN --num_epochs 10 --h_flip 1 --v_flip 1 --rotation 8  --batch_size 1 --num_val_images 256 --continue_training 1
fi
if [ "$2" == "Test" ]; then
ckpt=$P_data"TSaved/BestFr_"$fend"_model_"$model".ckpt"
rm  -f $P_data/Test/*
if [ "$6" != "" ]; then
ckpt=$P_data/TSaved/$6
fi
echo "using $ckpt"

if [ "$3" == "new" ]; then
\rm ./tmp/

rm -f $P_data/Predict/*
mkdir -p ./tmp/labels/
ln -sf mrc_files ./tmp/mrc_files
for i in `cat $P_data/Tfiles.txt`; do
cp $P_data/labels/$i ./tmp/labels;done
$anacondapath PSTest.py $P_data mrc_files/  ./tmp/labels/
rm -r -f ./tmp/labels
rm -f $P_data/Test/*
rm -f $P_data/Master_files/test/*					# Remove the # to make the folders	 clean when new files are added.
rm -f $P_data/Master_files/test_labels/*
mv $P_data/test/* $P_data/Master_files/test/
mv $P_data/test_labels/* $P_data/Master_files/test_labels/
rm -f ./tmp/mrc_files
fi
#cp $P_data/Master_files/test/* $P_data/test/
#cp $P_data/Master_files/test_labels/* $P_data/test_labels/

python splitJoin.py $P_data Master_files/test/ test/ Split        # Ensure that mrc files are used for determining image dimensions.
python splitJoin.py $P_data Master_files/test_labels/ test_labels/ Split

python test2.py --dataset $P_data --model $model  --checkpoint_path $ckpt
python splitJoin.py $P_data Test/ Final_labels/
fi
if [ "$2" == "Predict" ]; then
if [ "$3" == "new" ]; then


rm  -f $P_data/Predict/*
mkdir -p ./tmp/$mrc_files/
for i in `cat $P_data/Pfiles.txt`; do
cp $mrc_files/$i ./tmp/$mrc_files; done

rm  -f $P_data/Master_files/test/*
rm  -f $P_data/Master_files/test_labels/*
mkdir -p $P_data/Predict/
mkdir -p $P_data/Master_files/Predict/
rm -f $P_data/Predict/*

$anacondapath PSPred.py $P_data ./tmp/$mrc_files
rm -r -f ./tmp/$mrc_files
rm -f $P_data/Master_files/Predict/*
mv $P_data/Predict/* $P_data/Master_files/Predict/
fi
#mv -f $P_data/Predict $P_data/Master_files/
mkdir -p $P_data/Predict/
cp $P_data/Master_files/Predict/* $P_data/Predict/
#python splitJoin.py $P_data Master_files/Predict/ Predict/ Split    # Ensure that mrc files are used for determining image dimensions.
ckpt=$P_data"TSaved/BestFr_"$fend"_model_"$model".ckpt"
if [ "$6" != "" ]; then
ckpt=$6 #$P_data/TSaved/$6
fi
rm  -f $P_data/P_files/*
python predict2.py --dataset $P_data --model $model --image "" --checkpoint_path $ckpt
#python splitJoin.py $P_data P_files/ Predict_labels/
fi

