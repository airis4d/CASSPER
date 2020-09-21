#!/bin/bash
source ./cassper/bin/activate
#ls ./mrc_files/ > ./Allfiles.txt
#shuf -n 2 "$1"/Pfiles.txt
#rm -f ./Allfiles.txt
#./Train8.sh "$1" Predict new InceptionV4 FRRN-B "$1"/TSaved/BestFr_InceptionV4_model_FRRN-B_F1.ckpt first_two
python radius_and_erode_count1.py -i "$1"/first_labels/
rm -rf "$1"/first_labels/

