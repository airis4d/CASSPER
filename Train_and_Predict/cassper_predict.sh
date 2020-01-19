#!/bin/bash
source ./cassper/bin/activate
ls ./mrc_files/ > "$1"/Pfiles.txt
./Train8.sh "$1" Predict new InceptionV4 FRRN-B "$1"/TSaved/BestFr_InceptionV4_model_FRRN-B_F1.ckpt
